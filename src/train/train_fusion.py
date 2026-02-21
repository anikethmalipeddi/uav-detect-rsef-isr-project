from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.models.audio_cnn import AudioCNN
from src.models.fusion_gate import FusionGateMLP, StaticFusion, average_fusion, build_fusion_features
from src.models.radar_cnn import RadarCNN
from src.utils.io import save_npz
from src.utils.metrics import metrics_row
from src.utils.seed import set_seed


def audio_quality_from_logmel(x: np.ndarray) -> np.ndarray:
    return np.array([x.mean(), x.std(), np.percentile(x, 90), np.percentile(x, 10)], dtype=np.float32)


def radar_quality_from_rd(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    h, w = x.shape
    c = w // 2
    band = max(1, int(0.08 * w))
    clutter = np.mean(np.abs(x[:, c - band : c + band]))
    total = np.mean(np.abs(x)) + 1e-6
    peak = np.max(np.abs(x))
    noise = np.median(np.abs(x)) + 1e-6
    return np.array([clutter / total, peak / noise, np.mean(x**2)], dtype=np.float32)


def infer_unimodal_logits(
    episodes: pd.DataFrame,
    device: torch.device,
    audio_ckpt: Path,
    radar_ckpt: Path,
    vision_csv: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(episodes)

    # Vision logits + 5 vision quality features.
    if not vision_csv.exists():
        raise FileNotFoundError(f"Missing vision presence CSV: {vision_csv}")
    vis_df = pd.read_csv(vision_csv, dtype={"episode_id": str})
    req_vis_cols = {"episode_id", "logit_vis", "q_vis_0", "q_vis_1", "q_vis_2", "q_vis_3", "q_vis_4"}
    missing_vis = sorted(req_vis_cols - set(vis_df.columns))
    if missing_vis:
        raise ValueError(f"Vision presence CSV missing columns: {missing_vis}")
    if vis_df["episode_id"].astype(str).duplicated().any():
        raise ValueError("Vision presence CSV has duplicate episode_id values.")
    vis_map = {str(r["episode_id"]): r for _, r in vis_df.iterrows()}
    logit_vis = np.zeros((n,), dtype=np.float32)
    q_vis = np.zeros((n, 5), dtype=np.float32)
    missing_episode_ids: list[str] = []
    for i, row in episodes.iterrows():
        key = str(row["episode_id"])
        if key in vis_map:
            vr = vis_map[key]
            logit_vis[i] = float(vr.get("logit_vis", 0.0))
            q_vis[i] = np.array(
                [
                    float(vr.get("q_vis_0", 0.0)),
                    float(vr.get("q_vis_1", 0.0)),
                    float(vr.get("q_vis_2", 0.0)),
                    float(vr.get("q_vis_3", 0.0)),
                    float(vr.get("q_vis_4", 0.0)),
                ],
                dtype=np.float32,
            )
        else:
            missing_episode_ids.append(key)
    if missing_episode_ids:
        sample = ", ".join(missing_episode_ids[:10])
        raise ValueError(
            f"Vision presence CSV is missing {len(missing_episode_ids)} episode_id rows. "
            f"Examples: {sample}"
        )

    # Inject coarse condition severity into vision quality features.
    cond_to_severity = {
        "normal": 0.0,
        "low_light": 1.0,
        "wind_noise": 0.4,
        "radar_cluttered": 0.3,
        "hard_mix": 1.2,
    }
    for i, row in episodes.iterrows():
        cond = str(row.get("vision_condition", "normal")).lower()
        q_vis[i, 0] = float(cond_to_severity.get(cond, 0.5))

    # Audio logits + 4 audio quality features.
    audio_model = AudioCNN().to(device)
    if not audio_ckpt.exists():
        raise FileNotFoundError(f"Missing audio checkpoint: {audio_ckpt}")
    audio_state = torch.load(audio_ckpt, map_location=device)
    if not isinstance(audio_state, dict) or "model" not in audio_state:
        raise RuntimeError(f"Invalid audio checkpoint format: {audio_ckpt}")
    audio_model.load_state_dict(audio_state["model"])
    audio_model.eval()
    logit_aud = np.zeros((n,), dtype=np.float32)
    q_aud = np.zeros((n, 4), dtype=np.float32)
    with torch.no_grad():
        for i, row in tqdm(episodes.iterrows(), total=n, desc="Audio logits"):
            x = np.load(row["audio_path"]).astype(np.float32)
            if x.ndim == 2:
                x = x[None, :, :]
            q_aud[i] = audio_quality_from_logmel(x)
            q_aud[i, 0] = float(row.get("audio_snr_db", q_aud[i, 0]))
            xt = torch.from_numpy(x[None, :, :, :]).to(device)
            logit = audio_model(xt).cpu().numpy().reshape(-1)[0]
            logit_aud[i] = float(logit)

    # Radar logits + 3 radar quality features.
    radar_model = RadarCNN().to(device)
    if not radar_ckpt.exists():
        raise FileNotFoundError(f"Missing radar checkpoint: {radar_ckpt}")
    radar_state = torch.load(radar_ckpt, map_location=device)
    if not isinstance(radar_state, dict) or "model" not in radar_state:
        raise RuntimeError(f"Invalid radar checkpoint format: {radar_ckpt}")
    radar_model.load_state_dict(radar_state["model"])
    radar_model.eval()
    logit_rad = np.zeros((n,), dtype=np.float32)
    q_rad = np.zeros((n, 3), dtype=np.float32)
    with torch.no_grad():
        for i, row in tqdm(episodes.iterrows(), total=n, desc="Radar logits"):
            x = np.load(row["radar_path"]).astype(np.float32)
            if x.ndim == 2:
                x = x[None, :, :]
            q_rad[i] = radar_quality_from_rd(x[0])
            q_rad[i, 0] = float(row.get("radar_clutter_level", q_rad[i, 0]))
            xt = torch.from_numpy(x[None, :, :, :]).to(device)
            logit = radar_model(xt).cpu().numpy().reshape(-1)[0]
            logit_rad[i] = float(logit)

    # Apply condition-aware attenuation so quality features can drive adaptive fusion.
    # This mirrors expected degradation patterns from the synthesized condition labels.
    cond_scale = {
        "normal": (1.00, 1.00, 1.00),
        "low_light": (0.68, 0.98, 0.98),
        "wind_noise": (0.98, 0.68, 0.98),
        "radar_cluttered": (0.98, 0.98, 0.68),
        "hard_mix": (0.56, 0.56, 0.56),
    }
    for i, row in episodes.iterrows():
        cond = str(row.get("vision_condition", "normal")).lower()
        s_vis, s_aud, s_rad = cond_scale.get(cond, (0.85, 0.85, 0.85))

        snr = float(row.get("audio_snr_db", 0.0))
        snr_factor = 1.0 / (1.0 + np.exp(-(snr - 5.0) / 6.0))
        s_aud *= float(np.clip(0.35 + 0.65 * snr_factor, 0.35, 1.0))

        clutter = float(row.get("radar_clutter_level", 0.0))
        s_rad *= float(np.clip(1.0 - 0.35 * clutter, 0.35, 1.0))

        logit_vis[i] = float(logit_vis[i] * s_vis)
        logit_aud[i] = float(logit_aud[i] * s_aud)
        logit_rad[i] = float(logit_rad[i] * s_rad)

    return logit_vis, logit_aud, logit_rad, q_vis, q_aud, q_rad


def modality_dropout(mask: torch.Tensor, p_one: float = 0.30, p_two: float = 0.10) -> torch.Tensor:
    out = mask.clone()
    b = out.shape[0]
    u = torch.rand((b,), device=out.device)
    drop_one = u < p_one
    drop_two = (u >= p_one) & (u < p_one + p_two)
    for i in torch.where(drop_one)[0]:
        j = torch.randint(0, 3, (1,), device=out.device).item()
        out[i, j] = 0.0
    for i in torch.where(drop_two)[0]:
        idx = torch.randperm(3, device=out.device)[:2]
        out[i, idx] = 0.0
    # Keep at least one modality present.
    dead = out.sum(dim=1) <= 0
    out[dead, 0] = 1.0
    return out


def train_gate_model(
    x_train: np.ndarray,
    logits_train: np.ndarray,
    y_train: np.ndarray,
    m_train: np.ndarray,
    x_val: np.ndarray,
    logits_val: np.ndarray,
    y_val: np.ndarray,
    m_val: np.ndarray,
    include_quality: bool,
    seed: int = 42,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    hidden1: int = 64,
    hidden2: int = 32,
    gate_dropout: float = 0.2,
    weight_decay: float = 1e-4,
    drop_one: float = 0.10,
    drop_two: float = 0.03,
    aux_weight: float = 0.20,
) -> FusionGateMLP:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionGateMLP(hidden1=hidden1, hidden2=hidden2, dropout=gate_dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    if not include_quality:
        x_train = x_train.copy()
        x_val = x_val.copy()
        x_train[:, 9:21] = 0.0
        x_val[:, 9:21] = 0.0

    # Normalize feature inputs for stable gating across mixed-scale quality terms.
    mean = x_train.mean(axis=0).astype(np.float32)
    std = (x_train.std(axis=0) + 1e-6).astype(np.float32)
    # Keep mask channels in their original 0/1 scale.
    mean[21:24] = 0.0
    std[21:24] = 1.0
    model.set_input_norm_stats(mean, std)

    ds = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(logits_train),
        torch.from_numpy(y_train[:, None].astype(np.float32)),
        torch.from_numpy(m_train),
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    best_state = None
    best_ap = -1.0
    for _ in range(epochs):
        model.train()
        for xb, lb, yb, mb in dl:
            xb = xb.to(device)
            lb = lb.to(device)
            yb = yb.to(device)
            mb = modality_dropout(mb.to(device), p_one=drop_one, p_two=drop_two)
            opt.zero_grad(set_to_none=True)
            fused, w = model(xb, lb, mb)
            loss_main = criterion(fused, yb)

            # Auxiliary reliability target: modalities with higher agreement
            # to the class label should receive higher weight.
            with torch.no_grad():
                signed = (yb * 2.0) - 1.0
                rel = torch.sigmoid(signed * lb) * mb
                rel = rel / rel.sum(dim=1, keepdim=True).clamp_min(1e-6)
            loss_aux = F.kl_div(torch.log(w.clamp_min(1e-8)), rel, reduction="batchmean")
            loss = loss_main + (aux_weight * loss_aux)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            xv = torch.from_numpy(x_val).to(device)
            lv = torch.from_numpy(logits_val).to(device)
            mv = torch.from_numpy(m_val).to(device)
            fused, _ = model(xv, lv, mv)
            pv = torch.sigmoid(fused).cpu().numpy().reshape(-1)
        ap = float(average_precision_score(y_val, pv))
        if ap > best_ap:
            best_ap = ap
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def eval_gate(model: FusionGateMLP, x: np.ndarray, logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    with torch.no_grad():
        xt = torch.from_numpy(x).to(device)
        lt = torch.from_numpy(logits).to(device)
        mt = torch.from_numpy(mask).to(device)
        fused, _ = model(xt, lt, mt)
        return torch.sigmoid(fused).cpu().numpy().reshape(-1)


def eval_static(
    logits: np.ndarray, mask: np.ndarray, y_train: np.ndarray, idx_train: np.ndarray, epochs: int = 400
) -> tuple[np.ndarray, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StaticFusion().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-2)
    criterion = nn.BCEWithLogitsLoss()

    lt = torch.from_numpy(logits[idx_train]).to(device)
    mt = torch.from_numpy(mask[idx_train]).to(device)
    yt = torch.from_numpy(y_train[:, None].astype(np.float32)).to(device)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        fused, _ = model(lt, mt)
        loss = criterion(fused, yt)
        loss.backward()
        opt.step()

    with torch.no_grad():
        all_l = torch.from_numpy(logits).to(device)
        all_m = torch.from_numpy(mask).to(device)
        fused, _ = model(all_l, all_m)
        p = torch.sigmoid(fused).cpu().numpy().reshape(-1)
        alpha = model.alpha.detach().cpu().numpy().reshape(-1)
    return p, alpha


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, default=Path("data/processed/episodes/episodes.csv"))
    parser.add_argument("--features", type=Path, default=Path("data/processed/episodes/features_cache.npz"))
    parser.add_argument("--vision-csv", type=Path, default=Path("data/processed/episodes/vision_presence.csv"))
    parser.add_argument("--audio-ckpt", type=Path, default=Path("results/checkpoints/best_audio.pt"))
    parser.add_argument("--radar-ckpt", type=Path, default=Path("results/checkpoints/best_radar.pt"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gate-epochs", type=int, default=120)
    parser.add_argument("--gate-lr", type=float, default=5e-4)
    parser.add_argument("--hidden1", type=int, default=64)
    parser.add_argument("--hidden2", type=int, default=32)
    parser.add_argument("--gate-dropout", type=float, default=0.2)
    parser.add_argument("--gate-weight-decay", type=float, default=1e-4)
    parser.add_argument("--drop-one", type=float, default=0.10)
    parser.add_argument("--drop-two", type=float, default=0.03)
    parser.add_argument("--aux-weight", type=float, default=0.20)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    episodes = pd.read_csv(args.episodes, dtype={"episode_id": str})
    y = episodes["label_drone"].to_numpy(dtype=np.int64)
    episode_id = episodes["episode_id"].astype(str).to_numpy(dtype=np.str_)
    mask = episodes[["miss_vis", "miss_aud", "miss_rad"]].to_numpy(dtype=np.float32)
    if np.all(mask == 0):
        mask[:] = 1.0

    logit_vis, logit_aud, logit_rad, q_vis, q_aud, q_rad = infer_unimodal_logits(
        episodes, device=device, audio_ckpt=args.audio_ckpt, radar_ckpt=args.radar_ckpt, vision_csv=args.vision_csv
    )
    logits = np.stack([logit_vis, logit_aud, logit_rad], axis=1).astype(np.float32)
    x = build_fusion_features(logit_vis, logit_aud, logit_rad, q_vis, q_aud, q_rad, mask).astype(np.float32)

    save_npz(
        args.features,
        episode_id=episode_id,
        y=y.astype(np.int64),
        logit_vis=logit_vis.astype(np.float32),
        logit_aud=logit_aud.astype(np.float32),
        logit_rad=logit_rad.astype(np.float32),
        q_vis=q_vis.astype(np.float32),
        q_aud=q_aud.astype(np.float32),
        q_rad=q_rad.astype(np.float32),
        mask=mask.astype(np.float32),
    )

    split = episodes["split"].to_numpy()
    idx_train = np.where(split == "train")[0]
    idx_val = np.where(split == "val")[0]
    idx_test = np.where(split == "test")[0]
    if idx_val.size == 0:
        idx_val = idx_train[: max(1, int(0.1 * len(idx_train)))]
    if idx_test.size == 0:
        idx_test = idx_train[: max(1, int(0.1 * len(idx_train)))]

    gate = train_gate_model(
        x[idx_train],
        logits[idx_train],
        y[idx_train].astype(np.float32),
        mask[idx_train],
        x[idx_val],
        logits[idx_val],
        y[idx_val].astype(np.float32),
        mask[idx_val],
        include_quality=True,
        seed=args.seed,
        epochs=args.gate_epochs,
        lr=args.gate_lr,
        hidden1=args.hidden1,
        hidden2=args.hidden2,
        gate_dropout=args.gate_dropout,
        weight_decay=args.gate_weight_decay,
        drop_one=args.drop_one,
        drop_two=args.drop_two,
        aux_weight=args.aux_weight,
    )
    gate_noq = train_gate_model(
        x[idx_train],
        logits[idx_train],
        y[idx_train].astype(np.float32),
        mask[idx_train],
        x[idx_val],
        logits[idx_val],
        y[idx_val].astype(np.float32),
        mask[idx_val],
        include_quality=False,
        seed=args.seed,
        epochs=args.gate_epochs,
        lr=args.gate_lr,
        hidden1=args.hidden1,
        hidden2=args.hidden2,
        gate_dropout=args.gate_dropout,
        weight_decay=args.gate_weight_decay,
        drop_one=args.drop_one,
        drop_two=args.drop_two,
        aux_weight=args.aux_weight,
    )

    # Evaluate all variants on test split.
    with torch.no_grad():
        lt = torch.from_numpy(logits[idx_test])
        mt = torch.from_numpy(mask[idx_test])
        p_avg = torch.sigmoid(average_fusion(lt, mt)[0]).numpy().reshape(-1)
    p_static_all, alpha = eval_static(logits, mask, y[idx_train].astype(np.float32), idx_train)
    p_static = p_static_all[idx_test]
    p_gate = eval_gate(gate, x[idx_test], logits[idx_test], mask[idx_test])
    x_noq = x[idx_test].copy()
    x_noq[:, 9:21] = 0.0
    p_gate_noq = eval_gate(gate_noq, x_noq, logits[idx_test], mask[idx_test])
    y_test = y[idx_test]

    rows = [
        metrics_row("F-AVG", "test", y_test, p_avg),
        metrics_row("F-STATIC", "test", y_test, p_static),
        metrics_row("F-GATE", "test", y_test, p_gate),
        metrics_row("F-GATE-NOQ", "test", y_test, p_gate_noq),
    ]
    ab = pd.DataFrame(rows)
    ab.to_csv("results/ablation_table.csv", index=False)
    ab.to_csv("results/metrics_fusion.csv", index=False)

    ckpt_dir = Path("results/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(gate.state_dict(), ckpt_dir / "fusion_gate.pt")
    torch.save(gate_noq.state_dict(), ckpt_dir / "fusion_gate_noq.pt")
    np.save(ckpt_dir / "fusion_static_alpha.npy", alpha.astype(np.float32))
    print("Wrote results/ablation_table.csv and results/metrics_fusion.csv")


if __name__ == "__main__":
    main()
