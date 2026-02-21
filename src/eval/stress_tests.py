from __future__ import annotations

import argparse
import json
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
from src.utils.latency import measure_cuda_ms, measure_wall_ms
from src.utils.metrics import metrics_row


def evaluate_methods(
    lv: np.ndarray,
    la: np.ndarray,
    lr: np.ndarray,
    qv: np.ndarray,
    qa: np.ndarray,
    qr: np.ndarray,
    mask: np.ndarray,
    gate: object,
    gate_noq: object,
    alpha: np.ndarray,
    device: torch.device,
) -> dict[str, np.ndarray]:
    logits = np.stack([lv, la, lr], axis=1).astype(np.float32)
    x = build_fusion_features(lv, la, lr, qv, qa, qr, mask)
    p_avg, _ = _predict_average(logits, mask)
    p_static, _ = _predict_static(logits, mask, alpha=alpha)
    p_gate, _ = _predict_gate(gate, x, logits, mask, device=device)
    x_noq = x.copy()
    x_noq[:, 9:21] = 0.0
    p_gate_noq, _ = _predict_gate(gate_noq, x_noq, logits, mask, device=device)
    return {
        "F-AVG": p_avg,
        "F-STATIC": p_static,
        "F-GATE": p_gate,
        "F-GATE-NOQ": p_gate_noq,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, default=Path("data/processed/episodes/episodes.csv"))
    parser.add_argument("--features", type=Path, default=Path("data/processed/episodes/features_cache.npz"))
    parser.add_argument("--gate-ckpt", type=Path, default=Path("results/checkpoints/fusion_gate.pt"))
    parser.add_argument("--gate-noq-ckpt", type=Path, default=Path("results/checkpoints/fusion_gate_noq.pt"))
    parser.add_argument("--static-alpha", type=Path, default=Path("results/checkpoints/fusion_static_alpha.npy"))
    parser.add_argument("--seed", type=int, default=42)
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

    idx = np.where(episodes["split"].to_numpy() == "test")[0]
    if idx.size == 0:
        idx = np.arange(len(episodes))

    y_test = y[idx]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gate = _load_gate(args.gate_ckpt, device=device)
    gate_noq = _load_gate(args.gate_noq_ckpt, device=device)
    alpha = _load_static_alpha(args.static_alpha)
    rng = np.random.default_rng(args.seed)

    # Vision stress.
    vision_rows = []
    vision_cases = [
        ("blur", 1, 0.95),
        ("blur", 2, 0.85),
        ("blur", 3, 0.72),
        ("lowlight", 1, 0.92),
        ("lowlight", 2, 0.80),
        ("lowlight", 3, 0.65),
        ("jpeg", 1, 0.95),
        ("jpeg", 2, 0.88),
        ("jpeg", 3, 0.74),
    ]
    for corr, sev, scale in vision_cases:
        lv2 = lv.copy()
        qv2 = qv.copy()
        lv2[idx] = lv2[idx] * scale
        if corr == "blur":
            qv2[idx, 2] = qv2[idx, 2] * (1.0 / (1.0 + 0.8 * sev))
        elif corr == "lowlight":
            qv2[idx, 0] = qv2[idx, 0] * (1.0 - 0.20 * sev)
            qv2[idx, 1] = qv2[idx, 1] * (1.0 - 0.15 * sev)
        elif corr == "jpeg":
            qv2[idx, 1] = qv2[idx, 1] * (1.0 - 0.12 * sev)
            qv2[idx, 2] = qv2[idx, 2] * (1.0 - 0.10 * sev)

        probs = evaluate_methods(
            lv=lv2[idx],
            la=la[idx],
            lr=lr[idx],
            qv=qv2[idx],
            qa=qa[idx],
            qr=qr[idx],
            mask=mask[idx],
            gate=gate,
            gate_noq=gate_noq,
            alpha=alpha,
            device=device,
        )
        for run_id, p in probs.items():
            row = metrics_row(run_id, "test", y_test, p)
            row["corruption"] = corr
            row["severity"] = sev
            row["logit_scale"] = scale
            vision_rows.append(row)
    pd.DataFrame(vision_rows).to_csv("results/stress_vision_corruptions.csv", index=False)

    # Audio stress.
    audio_rows = []
    snr_levels = [30, 20, 10, 0, -10]
    for snr in snr_levels:
        la2 = la.copy()
        qa2 = qa.copy()
        noise_scale = (30.0 - float(snr)) / 30.0
        la2[idx] += rng.normal(0.0, noise_scale, size=idx.shape[0]).astype(np.float32)
        qa2[idx, 0] = float(snr)
        probs = evaluate_methods(
            lv=lv[idx],
            la=la2[idx],
            lr=lr[idx],
            qv=qv[idx],
            qa=qa2[idx],
            qr=qr[idx],
            mask=mask[idx],
            gate=gate,
            gate_noq=gate_noq,
            alpha=alpha,
            device=device,
        )
        for run_id, p in probs.items():
            row = metrics_row(run_id, "test", y_test, p)
            row["snr_db"] = snr
            audio_rows.append(row)
    pd.DataFrame(audio_rows).to_csv("results/stress_audio_snr.csv", index=False)

    # Radar stress.
    radar_rows = []
    for clutter in [0.1, 0.3, 0.5, 0.7, 0.9]:
        lr2 = lr.copy()
        qr2 = qr.copy()
        severity = 1.0 + (1.8 * clutter)
        lr2[idx] = lr2[idx] / severity
        qr2[idx, 0] = clutter
        probs = evaluate_methods(
            lv=lv[idx],
            la=la[idx],
            lr=lr2[idx],
            qv=qv[idx],
            qa=qa[idx],
            qr=qr2[idx],
            mask=mask[idx],
            gate=gate,
            gate_noq=gate_noq,
            alpha=alpha,
            device=device,
        )
        for run_id, p in probs.items():
            row = metrics_row(run_id, "test", y_test, p)
            row["clutter_level"] = clutter
            row["severity"] = severity
            radar_rows.append(row)
    pd.DataFrame(radar_rows).to_csv("results/stress_radar_clutter.csv", index=False)

    # Latency summary.
    batch = min(256, len(idx))
    i0 = idx[:batch]
    sample_mask = mask[i0].astype(np.float32)
    sample_logits = np.stack([lv[i0], la[i0], lr[i0]], axis=1).astype(np.float32)
    sample_x = build_fusion_features(lv[i0], la[i0], lr[i0], qv[i0], qa[i0], qr[i0], sample_mask).astype(np.float32)
    sample_x_noq = sample_x.copy()
    sample_x_noq[:, 9:21] = 0.0
    xt = torch.from_numpy(sample_x).to(device)
    xt_noq = torch.from_numpy(sample_x_noq).to(device)
    lt = torch.from_numpy(sample_logits).to(device)
    mt = torch.from_numpy(sample_mask).to(device)

    def _lat_fn_avg() -> None:
        _predict_average(sample_logits, sample_mask)

    def _lat_fn_static() -> None:
        _predict_static(sample_logits, sample_mask, alpha=alpha)

    def _lat_fn_gate() -> None:
        gate(xt, lt, mt)

    def _lat_fn_gate_noq() -> None:
        gate_noq(xt_noq, lt, mt)

    lat = {
        "batch_size": batch,
        "device": str(device),
        "methods": {
            "F-AVG": {
                "wall_clock": measure_wall_ms(_lat_fn_avg, warmup=20, runs=200).__dict__,
                "cuda_event": measure_cuda_ms(_lat_fn_avg, warmup=20, runs=200).__dict__,
            },
            "F-STATIC": {
                "wall_clock": measure_wall_ms(_lat_fn_static, warmup=20, runs=200).__dict__,
                "cuda_event": measure_cuda_ms(_lat_fn_static, warmup=20, runs=200).__dict__,
            },
            "F-GATE": {
                "wall_clock": measure_wall_ms(_lat_fn_gate, warmup=20, runs=200).__dict__,
                "cuda_event": measure_cuda_ms(_lat_fn_gate, warmup=20, runs=200).__dict__,
            },
            "F-GATE-NOQ": {
                "wall_clock": measure_wall_ms(_lat_fn_gate_noq, warmup=20, runs=200).__dict__,
                "cuda_event": measure_cuda_ms(_lat_fn_gate_noq, warmup=20, runs=200).__dict__,
            },
        },
    }
    Path("results").mkdir(parents=True, exist_ok=True)
    Path("results/latency_summary.json").write_text(json.dumps(lat, indent=2), encoding="utf-8")
    print("Wrote stress CSVs and results/latency_summary.json")


if __name__ == "__main__":
    main()
