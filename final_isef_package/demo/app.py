from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from PIL import Image


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
OUTPUT = ROOT / "output"


def infer_stub(row: pd.Series) -> dict:
    p_vis = float(row["p_vis"])
    p_aud = float(row["p_aud"])
    p_rad = float(row["p_rad"])
    w_vis = float(row["w_vis"])
    w_aud = float(row["w_aud"])
    w_rad = float(row["w_rad"])
    p_fused = float(row["p_fused"])
    return {
        "p_vis": p_vis,
        "p_aud": p_aud,
        "p_rad": p_rad,
        "w_vis": w_vis,
        "w_aud": w_aud,
        "w_rad": w_rad,
        "p_fused": p_fused,
        "predicted_class": "DRONE",
    }


def save_demo_screenshot(pred_df: pd.DataFrame, frame_img: Image.Image) -> Path:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT / "demo_screenshot.png"
    row = pred_df.iloc[min(30, len(pred_df) - 1)]
    scores = infer_stub(row)

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
        [scores["p_vis"], scores["p_aud"], scores["p_rad"], scores["p_fused"]],
        color=["#2563eb", "#16a34a", "#dc2626", "#0f766e"],
    )
    axes[2].set_ylim(0, 1)
    axes[2].set_title(f"Predicted: {scores['predicted_class']}")
    axes[2].set_ylabel("Confidence")
    axes[2].grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main() -> None:
    st.set_page_config(page_title="ISEF Demo (CPU)", layout="wide")
    st.title("ISEF Offline Demo")
    st.caption("Vision + Audio + Radar stub inference with adaptive fusion weights.")

    pred_df = pd.read_csv(ASSETS / "sample_predictions.csv")
    frame_img = Image.open(ASSETS / "sample_frame.png")
    idx = st.slider("Frame Index", 0, len(pred_df) - 1, min(20, len(pred_df) - 1))
    row = pred_df.iloc[idx]
    scores = infer_stub(row)

    c1, c2 = st.columns([1.8, 1.2])
    with c1:
        st.image(frame_img, caption="Packaged sample frame", use_container_width=True)
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(pred_df["t_sec"], pred_df["w_vis"], label="w_vis")
        ax.plot(pred_df["t_sec"], pred_df["w_aud"], label="w_aud")
        ax.plot(pred_df["t_sec"], pred_df["w_rad"], label="w_rad")
        ax.set_ylim(0, 1)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Weight")
        ax.set_title("Gating Weights")
        ax.grid(alpha=0.2)
        ax.legend()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with c2:
        st.metric("Predicted Class", scores["predicted_class"])
        st.metric("Fusion Confidence", f"{scores['p_fused']:.3f}")
        st.metric("Vision", f"{scores['p_vis']:.3f}")
        st.metric("Audio", f"{scores['p_aud']:.3f}")
        st.metric("Radar", f"{scores['p_rad']:.3f}")
        st.write(
            {
                "w_vis": round(scores["w_vis"], 3),
                "w_aud": round(scores["w_aud"], 3),
                "w_rad": round(scores["w_rad"], 3),
            }
        )
        if st.button("Save Demo Screenshot"):
            path = save_demo_screenshot(pred_df, frame_img)
            st.success(f"Saved: {path}")


if __name__ == "__main__":
    main()
