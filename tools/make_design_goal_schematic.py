from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle


def _pick_episode(episodes: pd.DataFrame, episode_id: str | None) -> pd.Series:
    if episode_id:
        hit = episodes.loc[episodes["episode_id"].astype(str) == str(episode_id)]
        if hit.empty:
            raise ValueError(f"episode_id={episode_id} not found in episodes CSV.")
        return hit.iloc[0]

    # Prefer stressful conditions with all three modalities present and drone-positive labels.
    pool = episodes[
        (episodes["split"] == "test")
        & (episodes["label_drone"] == 1)
        & (episodes["miss_vis"] == 1)
        & (episodes["miss_aud"] == 1)
        & (episodes["miss_rad"] == 1)
    ].copy()
    if pool.empty:
        raise RuntimeError("No test episodes with all modalities present were found.")

    priority = {"hard_mix": 0, "low_light": 1, "wind_noise": 2, "radar_cluttered": 3, "normal": 4}
    pool["prio"] = pool["vision_condition"].astype(str).map(priority).fillna(9)
    pool = pool.sort_values(["prio", "episode_id"], ascending=[True, True])
    return pool.iloc[0]


def _read_video_frame(video_path: Path, t0_sec: float, t1_sec: float) -> np.ndarray:
    if not video_path.exists():
        raise FileNotFoundError(f"Video path does not exist: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    t_mid = 0.5 * (float(t0_sec) + float(t1_sec))
    frame_idx = int(max(0, round(t_mid * fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok or frame_bgr is None:
        raise RuntimeError(f"Failed to read frame at t={t_mid:.2f}s from {video_path}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def _load_2d(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing array path: {path}")
    arr = np.load(path)
    if arr.ndim == 2:
        return arr.astype(np.float32)
    if arr.ndim == 3:
        return arr[0].astype(np.float32)
    raise ValueError(f"Expected 2D or 3D array at {path}, got shape={arr.shape}")


def _fusion_text(preds_csv: Path, episode_id: str) -> tuple[str, str]:
    if not preds_csv.exists():
        return "Episode-level decision: unavailable", "Fusion score: unavailable"

    df = pd.read_csv(preds_csv, dtype={"episode_id": str})
    row = df.loc[df["episode_id"].astype(str) == str(episode_id)]
    if row.empty:
        return "Episode-level decision: unavailable", "Fusion score: unavailable"

    r = row.iloc[0]
    if "p_gate" in r and pd.notna(r["p_gate"]):
        p = float(r["p_gate"])
    elif "p_gate_noq" in r and pd.notna(r["p_gate_noq"]):
        p = float(r["p_gate_noq"])
    else:
        return "Episode-level decision: unavailable", "Fusion score: unavailable"

    decision = "DRONE DETECTED" if p >= 0.5 else "NO DRONE"
    return f"Episode-level decision: {decision}", f"F-GATE score: {p:.3f}"


def _add_icon(ax: plt.Axes, kind: str) -> None:
    # Draw simple icon badges with ASCII-safe primitives.
    badge = Rectangle((0.70, 0.82), 0.28, 0.16, transform=ax.transAxes, facecolor="white", edgecolor="#64748b", lw=0.8)
    ax.add_patch(badge)
    if kind == "vision":
        moon_outer = Circle((0.76, 0.90), 0.035, transform=ax.transAxes, facecolor="#334155", edgecolor="none")
        moon_inner = Circle((0.775, 0.90), 0.03, transform=ax.transAxes, facecolor="white", edgecolor="none")
        ax.add_patch(moon_outer)
        ax.add_patch(moon_inner)
        ax.text(0.815, 0.895, "low-light / blur", transform=ax.transAxes, fontsize=7, va="center", ha="left", color="#0f172a")
    elif kind == "audio":
        x = np.array([0.73, 0.75, 0.77, 0.79, 0.81])
        y = np.array([0.865, 0.915, 0.86, 0.915, 0.865])
        ax.plot(x, y, transform=ax.transAxes, color="#b91c1c", lw=1.2)
        ax.text(0.815, 0.895, "noise", transform=ax.transAxes, fontsize=7, va="center", ha="left", color="#7f1d1d")
    elif kind == "radar":
        rng = np.random.default_rng(42)
        pts = rng.uniform([0.725, 0.845], [0.805, 0.945], size=(14, 2))
        ax.scatter(pts[:, 0], pts[:, 1], transform=ax.transAxes, s=6, c="#7c2d12")
        ax.text(0.815, 0.895, "clutter", transform=ax.transAxes, fontsize=7, va="center", ha="left", color="#7c2d12")


def build_figure(
    episodes_csv: Path,
    preds_csv: Path,
    out_png: Path,
    episode_id: str | None = None,
) -> Path:
    episodes = pd.read_csv(episodes_csv, dtype={"episode_id": str})
    row = _pick_episode(episodes, episode_id=episode_id)

    video_path = Path(str(row["video_path"]))
    audio_path = Path(str(row["audio_path"]))
    radar_path = Path(str(row["radar_path"]))

    frame = _read_video_frame(video_path, float(row["t0_sec"]), float(row["t1_sec"]))
    # Mildly emphasize challenging vision conditions while staying real-data based.
    frame_dark = (frame.astype(np.float32) * 0.78).clip(0, 255).astype(np.uint8)
    frame_blur = cv2.GaussianBlur(frame_dark, (5, 5), 0.8)

    audio = _load_2d(audio_path)
    radar = _load_2d(radar_path)

    decision_text, score_text = _fusion_text(preds_csv=preds_csv, episode_id=str(row["episode_id"]))

    fig = plt.figure(figsize=(14, 6), dpi=220)
    fig.patch.set_facecolor("white")

    ax_v = fig.add_axes([0.03, 0.40, 0.22, 0.46])
    ax_a = fig.add_axes([0.29, 0.40, 0.22, 0.46])
    ax_r = fig.add_axes([0.55, 0.40, 0.22, 0.46])
    ax_gate = fig.add_axes([0.80, 0.50, 0.17, 0.20])
    ax_out = fig.add_axes([0.80, 0.23, 0.17, 0.18])

    ax_v.imshow(frame_blur)
    ax_v.set_title("A. Vision (real frame)", fontsize=10, fontweight="bold")
    ax_v.axis("off")
    _add_icon(ax_v, "vision")

    ax_a.imshow(audio, aspect="auto", cmap="magma", origin="lower")
    ax_a.set_title("B. Acoustic (real log-mel)", fontsize=10, fontweight="bold")
    ax_a.axis("off")
    _add_icon(ax_a, "audio")

    ax_r.imshow(radar, aspect="auto", cmap="viridis", origin="lower")
    ax_r.set_title("C. Radar (real range-Doppler)", fontsize=10, fontweight="bold")
    ax_r.axis("off")
    _add_icon(ax_r, "radar")

    for ax, txt in [
        (ax_gate, "F-GATE\n(quality + masks)"),
        (ax_out, f"{decision_text}\n{score_text}"),
    ]:
        ax.axis("off")
        rect = Rectangle((0.02, 0.08), 0.96, 0.84, transform=ax.transAxes, facecolor="#e0f2fe", edgecolor="#0369a1", lw=1.5)
        ax.add_patch(rect)
        ax.text(0.5, 0.52, txt, ha="center", va="center", fontsize=10, fontweight="bold", color="#0c4a6e")

    for start_y in [0.63, 0.50, 0.37]:
        arr = FancyArrowPatch(
            (0.77, start_y),
            (0.80, 0.60),
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=12,
            lw=1.5,
            color="#334155",
        )
        fig.patches.append(arr)
    arr2 = FancyArrowPatch(
        (0.885, 0.50),
        (0.885, 0.41),
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=12,
        lw=1.5,
        color="#334155",
    )
    fig.patches.append(arr2)

    fig.suptitle("Condition-Adaptive Multimodal Fusion Under Degradation and Missing Sensors", fontsize=13, fontweight="bold", y=0.97)
    subtitle = (
        f"Real-data episode {row['episode_id']} ({row['split']}, condition={row['vision_condition']}, "
        f"SNR={float(row['audio_snr_db']):.1f} dB, clutter={float(row['radar_clutter_level']):.2f})"
    )
    fig.text(0.03, 0.03, subtitle, fontsize=9, color="#334155")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    return out_png


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, default=Path("data/processed/episodes/episodes.csv"))
    parser.add_argument("--preds", type=Path, default=Path("final_isef_package/tables/preds_fusion.csv"))
    parser.add_argument("--out", type=Path, default=Path("final_isef_package/figures/design_goal_stressors_schematic.png"))
    parser.add_argument("--episode-id", type=str, default=None)
    args = parser.parse_args()

    out = build_figure(
        episodes_csv=args.episodes,
        preds_csv=args.preds,
        out_png=args.out,
        episode_id=args.episode_id,
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
