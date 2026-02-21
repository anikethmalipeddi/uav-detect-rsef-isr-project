from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
OUTPUT = ROOT / "output"


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pred_df = pd.read_csv(ASSETS / "sample_predictions.csv")
    frame_img = Image.open(ASSETS / "sample_frame.png")

    idx = min(30, len(pred_df) - 1)
    row = pred_df.iloc[idx]
    p_vis = float(row["p_vis"])
    p_aud = float(row["p_aud"])
    p_rad = float(row["p_rad"])
    p_fused = float(row["p_fused"])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(frame_img)
    axes[0].set_title("Vision Sample")
    axes[0].axis("off")

    axes[1].plot(pred_df["t_sec"], pred_df["w_vis"], label="w_vis")
    axes[1].plot(pred_df["t_sec"], pred_df["w_aud"], label="w_aud")
    axes[1].plot(pred_df["t_sec"], pred_df["w_rad"], label="w_rad")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Adaptive Gating Weights")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Weight")
    axes[1].legend()
    axes[1].grid(alpha=0.2)

    axes[2].bar(
        ["Vision", "Audio", "Radar", "Fusion"],
        [p_vis, p_aud, p_rad, p_fused],
        color=["#2563eb", "#16a34a", "#dc2626", "#0f766e"],
    )
    axes[2].set_ylim(0, 1)
    axes[2].set_title("Predicted: DRONE")
    axes[2].set_ylabel("Confidence")
    axes[2].grid(axis="y", alpha=0.2)

    fig.tight_layout()
    out = OUTPUT / "demo_screenshot.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
