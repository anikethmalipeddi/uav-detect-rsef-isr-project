from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.models.audio_cnn import AudioCNN
from src.models.radar_cnn import RadarCNN
from src.utils.metrics import metrics_row, plot_pr_curve


def safe_sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, default=Path("data/processed/episodes/episodes.csv"))
    parser.add_argument("--audio-ckpt", type=Path, default=Path("results/checkpoints/best_audio.pt"))
    parser.add_argument("--radar-ckpt", type=Path, default=Path("results/checkpoints/best_radar.pt"))
    parser.add_argument("--vision-csv", type=Path, default=Path("data/processed/episodes/vision_presence.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/metrics_unimodal.csv"))
    args = parser.parse_args()

    if not args.episodes.exists():
        raise FileNotFoundError(f"Missing episodes CSV: {args.episodes}")
    if not args.vision_csv.exists():
        raise FileNotFoundError(f"Missing vision presence CSV: {args.vision_csv}")
    if not args.audio_ckpt.exists():
        raise FileNotFoundError(f"Missing audio checkpoint: {args.audio_ckpt}")
    if not args.radar_ckpt.exists():
        raise FileNotFoundError(f"Missing radar checkpoint: {args.radar_ckpt}")

    df = pd.read_csv(args.episodes, dtype={"episode_id": str})
    test = df[df["split"] == "test"].reset_index(drop=True)
    y = test["label_drone"].to_numpy(dtype=int)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Vision.
    vis_df = pd.read_csv(args.vision_csv, dtype={"episode_id": str})
    req_vis_cols = {"episode_id", "logit_vis"}
    missing_vis_cols = sorted(req_vis_cols - set(vis_df.columns))
    if missing_vis_cols:
        raise ValueError(f"Vision CSV missing required columns: {missing_vis_cols}")
    if vis_df["episode_id"].astype(str).duplicated().any():
        raise ValueError("Vision CSV has duplicate episode_id values.")
    vis_map = {str(r["episode_id"]): float(r["logit_vis"]) for _, r in vis_df.iterrows()}
    missing_ids = [str(eid) for eid in test["episode_id"].astype(str) if str(eid) not in vis_map]
    if missing_ids:
        sample = ", ".join(missing_ids[:10])
        raise ValueError(f"Vision CSV missing {len(missing_ids)} test episode IDs. Examples: {sample}")
    vis_logit = np.array([vis_map[str(eid)] for eid in test["episode_id"].astype(str)], dtype=np.float32)
    vis_prob = safe_sigmoid(vis_logit)

    # Audio.
    aud_model = AudioCNN().to(device)
    aud_state = torch.load(args.audio_ckpt, map_location=device)
    if not isinstance(aud_state, dict) or "model" not in aud_state:
        raise RuntimeError(f"Invalid audio checkpoint format: {args.audio_ckpt}")
    aud_model.load_state_dict(aud_state["model"])
    aud_model.eval()
    aud_logit = []
    with torch.no_grad():
        for p in tqdm(test["audio_path"], desc="Eval audio"):
            x = np.load(p).astype(np.float32)
            if x.ndim == 2:
                x = x[None, :, :]
            xt = torch.from_numpy(x[None, :, :, :]).to(device)
            aud_logit.append(float(aud_model(xt).cpu().numpy().reshape(-1)[0]))
    aud_prob = safe_sigmoid(np.array(aud_logit, dtype=np.float32))

    # Radar.
    rad_model = RadarCNN().to(device)
    rad_state = torch.load(args.radar_ckpt, map_location=device)
    if not isinstance(rad_state, dict) or "model" not in rad_state:
        raise RuntimeError(f"Invalid radar checkpoint format: {args.radar_ckpt}")
    rad_model.load_state_dict(rad_state["model"])
    rad_model.eval()
    rad_logit = []
    with torch.no_grad():
        for p in tqdm(test["radar_path"], desc="Eval radar"):
            x = np.load(p).astype(np.float32)
            if x.ndim == 2:
                x = x[None, :, :]
            xt = torch.from_numpy(x[None, :, :, :]).to(device)
            rad_logit.append(float(rad_model(xt).cpu().numpy().reshape(-1)[0]))
    rad_prob = safe_sigmoid(np.array(rad_logit, dtype=np.float32))

    rows = [
        metrics_row("UNI-V", "test", y, vis_prob),
        metrics_row("UNI-A", "test", y, aud_prob),
        metrics_row("UNI-R", "test", y, rad_prob),
    ]
    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    plot_pr_curve(y, vis_prob, "results/figures/pr_curves/vision_pr.png", "Vision PR Curve")
    plot_pr_curve(y, aud_prob, "results/figures/pr_curves/audio_pr.png", "Audio PR Curve")
    plot_pr_curve(y, rad_prob, "results/figures/pr_curves/radar_pr.png", "Radar PR Curve")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
