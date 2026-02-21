from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models.fusion_gate import FusionGateMLP, average_fusion, build_fusion_features
from src.utils.metrics import metrics_row


REQUIRED_FEATURE_KEYS = [
    "episode_id",
    "y",
    "logit_vis",
    "logit_aud",
    "logit_rad",
    "q_vis",
    "q_aud",
    "q_rad",
    "mask",
]


def safe_sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _normalize_episode_id(x: object) -> str:
    s = str(x).strip()
    if s.isdigit():
        return str(int(s))
    return s


def _load_gate(path: Path, device: torch.device) -> FusionGateMLP:
    if not path.exists():
        raise FileNotFoundError(f"Missing fusion gate checkpoint: {path}")
    state = torch.load(path, map_location=device)
    if not isinstance(state, dict):
        raise RuntimeError(f"Unexpected checkpoint format in {path}")
    if "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    elif "model" in state and isinstance(state["model"], dict):
        state = state["model"]

    k_in = "net.0.weight"
    k_mid = "net.3.weight"
    if k_in not in state or k_mid not in state:
        raise RuntimeError(f"Fusion checkpoint missing expected keys ({k_in}, {k_mid}): {path}")
    w_in = state[k_in]
    w_mid = state[k_mid]
    if w_in.ndim != 2 or w_mid.ndim != 2:
        raise RuntimeError(f"Fusion checkpoint has invalid linear weight tensors: {path}")
    hidden1 = int(w_in.shape[0])
    input_dim = int(w_in.shape[1])
    hidden2 = int(w_mid.shape[0])

    model = FusionGateMLP(input_dim=input_dim, hidden1=hidden1, hidden2=hidden2, dropout=0.0).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def _load_static_alpha(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing static fusion weights: {path}")
    alpha = np.load(path).astype(np.float32).reshape(-1)
    if alpha.shape != (3,):
        raise ValueError(f"Expected static alpha shape (3,), got {alpha.shape} from {path}")
    return alpha


def _load_features(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing features cache: {path}")
    z = np.load(path, allow_pickle=False)
    missing = [k for k in REQUIRED_FEATURE_KEYS if k not in z.files]
    if missing:
        raise KeyError(f"Missing arrays in {path}: {missing}")
    out = {k: z[k] for k in REQUIRED_FEATURE_KEYS}
    n = int(out["y"].shape[0])
    for k in REQUIRED_FEATURE_KEYS:
        if out[k].shape[0] != n:
            raise ValueError(f"Array {k} has inconsistent first dimension {out[k].shape[0]} != {n}")
    if out["q_vis"].shape[1] != 5 or out["q_aud"].shape[1] != 4 or out["q_rad"].shape[1] != 3:
        raise ValueError(
            "Quality feature shapes must be q_vis=(N,5), q_aud=(N,4), q_rad=(N,3). "
            f"Got {out['q_vis'].shape}, {out['q_aud'].shape}, {out['q_rad'].shape}"
        )
    if out["mask"].shape[1] != 3:
        raise ValueError(f"Mask shape must be (N,3), got {out['mask'].shape}")
    return out


def _align_episodes(
    episodes: pd.DataFrame,
    feature_episode_ids: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    if "episode_id" not in episodes.columns:
        raise KeyError("episodes.csv missing required column: episode_id")
    eps = episodes.copy()
    eps["episode_id"] = eps["episode_id"].astype(str)
    feat_ids = np.asarray(feature_episode_ids).astype(str)
    feat_norm = np.array([_normalize_episode_id(x) for x in feat_ids], dtype=object)
    eps_norm = np.array([_normalize_episode_id(x) for x in eps["episode_id"].to_numpy()], dtype=object)

    idx_map = {eid: i for i, eid in enumerate(feat_norm)}
    if len(idx_map) != len(feat_ids):
        raise ValueError("Feature cache episode_id values are not unique after normalization.")

    missing_in_features = [str(eps["episode_id"].iloc[i]) for i in range(len(eps_norm)) if eps_norm[i] not in idx_map]
    if missing_in_features:
        raise ValueError(f"{len(missing_in_features)} episode_ids missing in feature cache.")

    order = np.array([idx_map[x] for x in eps_norm], dtype=np.int64)
    return eps, order


def _predict_gate(
    model: FusionGateMLP,
    x: np.ndarray,
    logits: np.ndarray,
    mask: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        xt = torch.from_numpy(x.astype(np.float32)).to(device)
        lt = torch.from_numpy(logits.astype(np.float32)).to(device)
        mt = torch.from_numpy(mask.astype(np.float32)).to(device)
        fused, w = model(xt, lt, mt)
    return torch.sigmoid(fused).cpu().numpy().reshape(-1), w.cpu().numpy()


def _predict_average(logits: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lt = torch.from_numpy(logits.astype(np.float32))
    mt = torch.from_numpy(mask.astype(np.float32))
    fused, w = average_fusion(lt, mt)
    return torch.sigmoid(fused).numpy().reshape(-1), w.numpy()


def _predict_static(logits: np.ndarray, mask: np.ndarray, alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.broadcast_to(alpha.reshape(1, 3), logits.shape).astype(np.float32).copy()
    a[mask <= 0.0] = -1e9
    a = a - a.max(axis=1, keepdims=True)
    exp_a = np.exp(a)
    w = exp_a / np.clip(exp_a.sum(axis=1, keepdims=True), 1e-6, None)
    fused = (w * logits).sum(axis=1)
    return safe_sigmoid(fused), w


def _ensure_mask_valid(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.float32)
    m = (m > 0.0).astype(np.float32)
    dead = np.where(m.sum(axis=1) <= 0.0)[0]
    if dead.size > 0:
        raise ValueError(f"Found {dead.size} episodes with all modalities missing in mask.")
    return m


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, default=Path("data/processed/episodes/episodes.csv"))
    parser.add_argument("--features", type=Path, default=Path("data/processed/episodes/features_cache.npz"))
    parser.add_argument("--gate-ckpt", type=Path, default=Path("results/checkpoints/fusion_gate.pt"))
    parser.add_argument("--gate-noq-ckpt", type=Path, default=Path("results/checkpoints/fusion_gate_noq.pt"))
    parser.add_argument("--static-alpha", type=Path, default=Path("results/checkpoints/fusion_static_alpha.npy"))
    parser.add_argument("--out", type=Path, default=Path("results/metrics_fusion.csv"))
    parser.add_argument("--ablation-out", type=Path, default=Path("results/ablation_table.csv"))
    parser.add_argument("--pred-out", type=Path, default=Path("results/preds_fusion.csv"))
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "all"])
    args = parser.parse_args()

    if not args.episodes.exists():
        raise FileNotFoundError(f"Episodes CSV not found: {args.episodes}")
    episodes_raw = pd.read_csv(args.episodes, dtype={"episode_id": str})
    z = _load_features(args.features)
    episodes, order = _align_episodes(episodes_raw, z["episode_id"])

    y = z["y"][order].astype(np.int64)
    lv = z["logit_vis"][order].astype(np.float32)
    la = z["logit_aud"][order].astype(np.float32)
    lr = z["logit_rad"][order].astype(np.float32)
    qv = z["q_vis"][order].astype(np.float32)
    qa = z["q_aud"][order].astype(np.float32)
    qr = z["q_rad"][order].astype(np.float32)
    mask = _ensure_mask_valid(z["mask"][order])

    x = build_fusion_features(lv, la, lr, qv, qa, qr, mask)
    if x.shape[1] != 24:
        raise ValueError(f"Fusion input dimension must be 24, got {x.shape[1]}")
    logits = np.stack([lv, la, lr], axis=1).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gate = _load_gate(args.gate_ckpt, device=device)
    gate_noq = _load_gate(args.gate_noq_ckpt, device=device)
    alpha = _load_static_alpha(args.static_alpha)

    p_avg, w_avg = _predict_average(logits, mask)
    p_static, w_static = _predict_static(logits, mask, alpha=alpha)
    p_gate, w_gate = _predict_gate(gate, x, logits, mask, device=device)
    x_noq = x.copy()
    x_noq[:, 9:21] = 0.0
    p_gate_noq, w_gate_noq = _predict_gate(gate_noq, x_noq, logits, mask, device=device)

    pred_cols = {
        "episode_id": episodes["episode_id"].astype(str).to_numpy(),
        "split": episodes["split"].astype(str).to_numpy(),
        "y_true": y.astype(np.int64),
        "logit_vis": lv,
        "logit_aud": la,
        "logit_rad": lr,
        "p_vis": safe_sigmoid(lv),
        "p_aud": safe_sigmoid(la),
        "p_rad": safe_sigmoid(lr),
        "p_avg": p_avg,
        "p_static": p_static,
        "p_gate": p_gate,
        "p_gate_noq": p_gate_noq,
        "w_vis": w_gate[:, 0],
        "w_aud": w_gate[:, 1],
        "w_rad": w_gate[:, 2],
        "w_avg_vis": w_avg[:, 0],
        "w_avg_aud": w_avg[:, 1],
        "w_avg_rad": w_avg[:, 2],
        "w_static_vis": w_static[:, 0],
        "w_static_aud": w_static[:, 1],
        "w_static_rad": w_static[:, 2],
        "w_gate_vis": w_gate[:, 0],
        "w_gate_aud": w_gate[:, 1],
        "w_gate_rad": w_gate[:, 2],
        "w_gate_noq_vis": w_gate_noq[:, 0],
        "w_gate_noq_aud": w_gate_noq[:, 1],
        "w_gate_noq_rad": w_gate_noq[:, 2],
        "miss_vis": mask[:, 0],
        "miss_aud": mask[:, 1],
        "miss_rad": mask[:, 2],
    }
    for c in ["vision_condition", "audio_snr_db", "radar_clutter_level", "src_video_id", "src_audio_id", "src_radar_id"]:
        if c in episodes.columns:
            pred_cols[c] = episodes[c].to_numpy()
    pred_df = pd.DataFrame(pred_cols)
    args.pred_out.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(args.pred_out, index=False)

    if args.split == "all":
        idx = np.arange(len(pred_df), dtype=np.int64)
        split_name = "all"
    else:
        idx = np.where(pred_df["split"].to_numpy() == args.split)[0]
        split_name = args.split
        if idx.size == 0:
            raise ValueError(f"No rows found for split='{args.split}'")
    y_eval = y[idx]

    rows = [
        metrics_row("F-AVG", split_name, y_eval, p_avg[idx]),
        metrics_row("F-STATIC", split_name, y_eval, p_static[idx]),
        metrics_row("F-GATE", split_name, y_eval, p_gate[idx]),
        metrics_row("F-GATE-NOQ", split_name, y_eval, p_gate_noq[idx]),
    ]
    metrics_df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(args.out, index=False)

    args.ablation_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(args.ablation_out, index=False)
    print(f"Wrote {args.out}, {args.ablation_out}, and {args.pred_out}")


if __name__ == "__main__":
    main()
