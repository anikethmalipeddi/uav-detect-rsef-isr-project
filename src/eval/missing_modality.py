from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.eval.eval_fusion import (
    _align_episodes,
    _ensure_mask_valid,
    _load_features,
    _load_gate,
    _load_static_alpha,
    _predict_average,
    _predict_gate,
    _predict_static,
)
from src.models.fusion_gate import build_fusion_features
from src.utils.metrics import metrics_row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, default=Path("data/processed/episodes/episodes.csv"))
    parser.add_argument("--features", type=Path, default=Path("data/processed/episodes/features_cache.npz"))
    parser.add_argument("--gate-ckpt", type=Path, default=Path("results/checkpoints/fusion_gate.pt"))
    parser.add_argument("--gate-noq-ckpt", type=Path, default=Path("results/checkpoints/fusion_gate_noq.pt"))
    parser.add_argument("--static-alpha", type=Path, default=Path("results/checkpoints/fusion_static_alpha.npy"))
    parser.add_argument("--out", type=Path, default=Path("results/missing_modality.csv"))
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
    _ = _ensure_mask_valid(z["mask"][order])
    logits = np.stack([lv, la, lr], axis=1).astype(np.float32)

    idx = np.where(episodes["split"].to_numpy() == "test")[0]
    if idx.size == 0:
        idx = np.arange(len(episodes))
    y_test = y[idx]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gate = _load_gate(args.gate_ckpt, device=device)
    gate_noq = _load_gate(args.gate_noq_ckpt, device=device)
    alpha = _load_static_alpha(args.static_alpha)

    scenarios = {
        "all_present": np.array([1, 1, 1], dtype=np.float32),
        "vision_missing": np.array([0, 1, 1], dtype=np.float32),
        "audio_missing": np.array([1, 0, 1], dtype=np.float32),
        "radar_missing": np.array([1, 1, 0], dtype=np.float32),
        "vision_only": np.array([1, 0, 0], dtype=np.float32),
        "audio_only": np.array([0, 1, 0], dtype=np.float32),
        "radar_only": np.array([0, 0, 1], dtype=np.float32),
    }

    rows = []
    for name, m in scenarios.items():
        mask = np.broadcast_to(m[None, :], (len(idx), 3)).copy()
        lv_s = lv[idx]
        la_s = la[idx]
        lr_s = lr[idx]
        qv_s = qv[idx]
        qa_s = qa[idx]
        qr_s = qr[idx]
        logits_s = logits[idx]
        x = build_fusion_features(lv_s, la_s, lr_s, qv_s, qa_s, qr_s, mask)

        p_avg, _ = _predict_average(logits_s, mask)
        p_static, _ = _predict_static(logits_s, mask, alpha=alpha)
        p_gate, _ = _predict_gate(gate, x, logits_s, mask, device=device)
        x_noq = x.copy()
        x_noq[:, 9:21] = 0.0
        p_gate_noq, _ = _predict_gate(gate_noq, x_noq, logits_s, mask, device=device)

        rows.extend(
            [
                {**metrics_row("F-AVG", "test", y_test, p_avg), "scenario": name},
                {**metrics_row("F-STATIC", "test", y_test, p_static), "scenario": name},
                {**metrics_row("F-GATE", "test", y_test, p_gate), "scenario": name},
                {**metrics_row("F-GATE-NOQ", "test", y_test, p_gate_noq), "scenario": name},
            ]
        )

    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
