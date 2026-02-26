from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


def pick_episode(episodes: pd.DataFrame, episode_id: str | None) -> pd.Series:
    if episode_id is not None:
        hit = episodes.loc[episodes["episode_id"].astype(str) == str(episode_id)]
        if hit.empty:
            raise ValueError(f"episode_id={episode_id} not found in {len(episodes)} episodes.")
        return hit.iloc[0]

    pool = episodes[
        (episodes["split"] == "test")
        & (episodes["label_drone"] == 1)
        & (episodes["miss_vis"] == 1)
        & (episodes["miss_aud"] == 1)
        & (episodes["miss_rad"] == 1)
    ].copy()
    if pool.empty:
        raise RuntimeError("No suitable all-present test episodes found.")

    priority = {"hard_mix": 0, "low_light": 1, "wind_noise": 2, "radar_cluttered": 3, "normal": 4}
    pool["prio"] = pool["vision_condition"].astype(str).map(priority).fillna(9)
    pool = pool.sort_values(["prio", "episode_id"])
    return pool.iloc[0]


def pick_diverse_rows(
    episodes: pd.DataFrame,
    vision_episode_id: str | None = None,
    audio_episode_id: str | None = None,
    radar_episode_id: str | None = None,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    pool = episodes[
        (episodes["split"] == "test")
        & (episodes["label_drone"] == 1)
        & (episodes["miss_vis"] == 1)
        & (episodes["miss_aud"] == 1)
        & (episodes["miss_rad"] == 1)
    ].copy()
    if pool.empty:
        raise RuntimeError("No suitable all-present test episodes found.")

    def pick_by_id(eid: str | None) -> pd.Series | None:
        if eid is None:
            return None
        hit = pool.loc[pool["episode_id"].astype(str) == str(eid)]
        if hit.empty:
            raise ValueError(f"episode_id={eid} not found in all-present test set.")
        return hit.iloc[0]

    used: set[str] = set()

    v_row = pick_by_id(vision_episode_id)
    if v_row is None:
        v_candidates = pool.loc[pool["vision_condition"].isin(["low_light", "hard_mix"])].copy()
        if v_candidates.empty:
            v_candidates = pool.copy()
        v_candidates = v_candidates.sort_values(["vision_condition", "episode_id"])
        v_row = v_candidates.iloc[0]
    used.add(str(v_row["episode_id"]))

    a_row = pick_by_id(audio_episode_id)
    if a_row is None:
        a_candidates = pool.loc[~pool["episode_id"].astype(str).isin(used)].copy()
        if a_candidates.empty:
            a_candidates = pool.copy()
        # Lower SNR means noisier audio; pick a strong noisy case.
        a_candidates = a_candidates.sort_values(["audio_snr_db", "episode_id"], ascending=[True, True])
        a_row = a_candidates.iloc[0]
    used.add(str(a_row["episode_id"]))

    r_row = pick_by_id(radar_episode_id)
    if r_row is None:
        r_candidates = pool.loc[~pool["episode_id"].astype(str).isin(used)].copy()
        if r_candidates.empty:
            r_candidates = pool.copy()
        # Higher clutter is visually more distinct for radar panel.
        r_candidates = r_candidates.sort_values(["radar_clutter_level", "episode_id"], ascending=[False, True])
        r_row = r_candidates.iloc[0]

    return v_row, a_row, r_row


def load_video_strip(video_path: Path, t0_sec: float, t1_sec: float, n_frames: int = 3) -> np.ndarray:
    if not video_path.exists():
        raise FileNotFoundError(f"Missing video: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    dt = max(0.15, (float(t1_sec) - float(t0_sec)) / float(max(1, n_frames + 1)))
    times = [float(t0_sec) + dt * (i + 1) for i in range(n_frames)]

    frames: list[np.ndarray] = []
    for t in times:
        idx = int(max(0, round(t * fps)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            if frames:
                frame = cv2.cvtColor(frames[-1], cv2.COLOR_RGB2BGR)
            else:
                frame = np.zeros((144, 256, 3), dtype=np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (280, 200), interpolation=cv2.INTER_AREA)
        cv2.putText(
            frame,
            f"t={t:.2f}s",
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"t={t:.2f}s",
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (30, 41, 59),
            1,
            cv2.LINE_AA,
        )
        frames.append(frame)

    cap.release()

    gap = 10
    spacer = 255 * np.ones((frames[0].shape[0], gap, 3), dtype=np.uint8)
    strip = frames[0]
    for fr in frames[1:]:
        strip = np.concatenate([strip, spacer, fr], axis=1)
    # Keep a thin margin only; large padding wastes poster space.
    top_bot = 12
    canvas = 245 * np.ones((strip.shape[0] + (2 * top_bot), strip.shape[1], 3), dtype=np.uint8)
    canvas[top_bot : top_bot + strip.shape[0], :, :] = strip
    return canvas


def load_2d_npy(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing array: {path}")
    arr = np.load(path)
    if arr.ndim == 2:
        return arr.astype(np.float32)
    if arr.ndim == 3:
        return arr[0].astype(np.float32)
    raise ValueError(f"Expected 2D/3D array in {path}, got shape={arr.shape}")


def draw_card(ax: plt.Axes, x: float, y: float, w: float, h: float, title: str, lines: list[str]) -> None:
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        facecolor="#f8fafc",
        edgecolor="#0f766e",
        linewidth=1.5,
        transform=ax.transAxes,
    )
    ax.add_patch(box)
    ax.text(
        x + 0.02,
        y + h - 0.07,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
        color="#0f172a",
    )
    yy = y + h - 0.28
    for line in lines:
        ax.text(x + 0.03, yy, f"- {line}", transform=ax.transAxes, fontsize=9.5, color="#334155", ha="left", va="top")
        yy -= 0.10


def draw_vision_icon(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    cx = x + (0.18 * w)
    cy = y + (0.50 * h)
    r_outer = 0.18 * h
    r_inner = 0.15 * h
    ax.add_patch(Circle((cx, cy), r_outer, transform=ax.transAxes, facecolor="#334155", edgecolor="none"))
    ax.add_patch(Circle((cx + (0.07 * w), cy), r_inner, transform=ax.transAxes, facecolor="#f8fafc", edgecolor="none"))
    ax.plot([x + (0.42 * w), x + (0.78 * w)], [y + (0.63 * h), y + (0.63 * h)], transform=ax.transAxes, color="#64748b", lw=2)
    ax.plot([x + (0.42 * w), x + (0.78 * w)], [y + (0.42 * h), y + (0.42 * h)], transform=ax.transAxes, color="#94a3b8", lw=2)
    for i in range(3):
        ax.add_patch(
            Rectangle(
                (x + (0.82 * w) + i * (0.085 * w), y + (0.35 * h) + i * (0.10 * h)),
                0.07 * w,
                0.07 * w,
                transform=ax.transAxes,
                facecolor="#64748b",
            )
        )


def draw_audio_icon(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    xs = np.linspace(x + (0.06 * w), x + (0.72 * w), 80)
    ys = y + (0.50 * h) + (0.12 * h) * np.sin(np.linspace(0, 4 * np.pi, 80))
    ax.plot(xs, ys, transform=ax.transAxes, color="#0f766e", lw=2)
    rng = np.random.default_rng(0)
    pts = rng.uniform([x + (0.76 * w), y + (0.34 * h)], [x + (0.98 * w), y + (0.70 * h)], size=(20, 2))
    ax.scatter(pts[:, 0], pts[:, 1], transform=ax.transAxes, s=8, c="#b91c1c")


def draw_radar_icon(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    ox = x + (0.34 * w)
    oy = y + (0.48 * h)
    ax.add_patch(Arc((ox, oy), 0.42 * w, 0.46 * h, theta1=0, theta2=180, transform=ax.transAxes, color="#0e7490", lw=2))
    ax.add_patch(Arc((ox, oy), 0.62 * w, 0.68 * h, theta1=0, theta2=180, transform=ax.transAxes, color="#0e7490", lw=2))
    ax.plot([ox, ox], [oy, oy + (0.28 * h)], transform=ax.transAxes, color="#0e7490", lw=2)
    rng = np.random.default_rng(1)
    pts = rng.uniform([x + (0.64 * w), y + (0.36 * h)], [x + (0.98 * w), y + (0.78 * h)], size=(18, 2))
    ax.scatter(pts[:, 0], pts[:, 1], transform=ax.transAxes, s=10, c="#7c2d12")


def draw_missing_icon(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    labels = [("V", "#2563eb"), ("A", "#0f766e"), ("R", "#0e7490")]
    for i, (lab, col) in enumerate(labels):
        cx = x + (0.18 * w) + i * (0.30 * w)
        cy = y + (0.53 * h)
        ax.add_patch(Circle((cx, cy), 0.018, transform=ax.transAxes, facecolor="white", edgecolor=col, lw=1.5))
        ax.text(cx, cy, lab, transform=ax.transAxes, ha="center", va="center", fontsize=8, color=col, fontweight="bold")
    # Cross out audio.
    cx = x + (0.48 * w)
    cy = y + (0.53 * h)
    ax.plot([cx - (0.06 * w), cx + (0.06 * w)], [cy - (0.16 * h), cy + (0.16 * h)], transform=ax.transAxes, color="#dc2626", lw=2)
    ax.plot([cx + (0.06 * w), cx - (0.06 * w)], [cy - (0.16 * h), cy + (0.16 * h)], transform=ax.transAxes, color="#dc2626", lw=2)


def fig_to_ax(ax: plt.Axes, fig: plt.Figure, x: float, y: float) -> tuple[float, float]:
    disp = fig.transFigure.transform((x, y))
    out = ax.transAxes.inverted().transform(disp)
    return float(out[0]), float(out[1])


def build_figure(
    episodes_csv: Path,
    out_png: Path,
    episode_id: str | None,
    diverse_modalities: bool = False,
    vision_episode_id: str | None = None,
    audio_episode_id: str | None = None,
    radar_episode_id: str | None = None,
) -> Path:
    episodes = pd.read_csv(episodes_csv, dtype={"episode_id": str})
    if diverse_modalities:
        v_row, a_row, r_row = pick_diverse_rows(
            episodes,
            vision_episode_id=vision_episode_id,
            audio_episode_id=audio_episode_id,
            radar_episode_id=radar_episode_id,
        )
    else:
        row = pick_episode(episodes, episode_id=episode_id)
        v_row, a_row, r_row = row, row, row

    strip = load_video_strip(Path(str(v_row["video_path"])), float(v_row["t0_sec"]), float(v_row["t1_sec"]))
    audio = load_2d_npy(Path(str(a_row["audio_path"])))
    radar = load_2d_npy(Path(str(r_row["radar_path"])))

    fig = plt.figure(figsize=(15.2, 7.8), dpi=240, facecolor="white")

    # Compact manual placement to reduce random whitespace.
    ax_v = fig.add_axes([0.03, 0.53, 0.36, 0.32])
    ax_a = fig.add_axes([0.41, 0.53, 0.27, 0.32])
    ax_r = fig.add_axes([0.70, 0.53, 0.27, 0.32])
    ax_s = fig.add_axes([0.03, 0.10, 0.94, 0.32])

    for ax in [ax_v, ax_a, ax_r]:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_edgecolor("#94a3b8")

    ax_v.imshow(strip)
    ax_v.set_title("Vision thumbnail strip (real video frames)", fontsize=11.5, fontweight="bold", pad=8)

    ax_a.imshow(audio, aspect="auto", cmap="magma", origin="lower")
    ax_a.set_title("Audio log-mel thumbnail (real)", fontsize=11.5, fontweight="bold", pad=8)

    ax_r.imshow(radar, aspect="auto", cmap="viridis", origin="lower")
    ax_r.set_title("Radar range-Doppler thumbnail (real)", fontsize=11.5, fontweight="bold", pad=8)

    fig.suptitle("Data & Conditions: Episode + Stressors", fontsize=20, fontweight="bold", y=0.965)
    if diverse_modalities:
        mid_text = "Episode definition: 2 s synchronized multimodal clip -> 1 binary label (UAV present/absent)"
        sub_text = "Top-row thumbnails are real examples from different test episodes for visual diversity."
        fig.text(
            0.5,
            0.49,
            mid_text,
            ha="center",
            va="center",
            fontsize=11.5,
            fontweight="bold",
            color="#0f172a",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 2.0, "alpha": 0.9},
        )
        fig.text(
            0.5,
            0.462,
            sub_text,
            ha="center",
            va="center",
            fontsize=9.8,
            color="#475569",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5, "alpha": 0.9},
        )
    else:
        fig.text(
            0.5,
            0.485,
            "Episode = 2 s synchronized multimodal clip -> 1 binary label (UAV present/absent)",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="#0f172a",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 2.0, "alpha": 0.9},
        )

    ax_s.set_xlim(0, 1)
    ax_s.set_ylim(0, 1)
    ax_s.axis("off")
    ax_s.text(0.01, 0.95, "Condition and availability stressors", fontsize=13.5, fontweight="bold", color="#0f172a")

    cards = [
        (0.02, 0.07, 0.22, 0.72, "Vision", ["Blur", "Low-light", "Compression"]),
        (0.265, 0.07, 0.22, 0.72, "Audio", ["SNR noise sweep", "Noisy backgrounds"]),
        (0.51, 0.07, 0.22, 0.72, "Radar", ["Clutter-level sweep", "Signal attenuation"]),
        (0.755, 0.07, 0.225, 0.72, "Missing modality", ["Vision missing", "Audio missing", "Radar missing", "Mask vectors [1,0,1]"]),
    ]
    for c in cards:
        draw_card(ax_s, *c)

    # Icons live inside a consistent icon-box in each card for perfect alignment.
    icon_w = 0.12
    icon_h = 0.12
    for idx, card in enumerate(cards):
        cx, cy, cw, ch, _, _ = card
        ix = cx + cw - icon_w - 0.02
        iy = cy + ch - icon_h - 0.10
        if idx == 0:
            draw_vision_icon(ax_s, ix, iy, icon_w, icon_h)
        elif idx == 1:
            draw_audio_icon(ax_s, ix, iy, icon_w, icon_h)
        elif idx == 2:
            draw_radar_icon(ax_s, ix, iy, icon_w, icon_h)
        else:
            draw_missing_icon(ax_s, ix, iy, icon_w, icon_h)

    # Arrows anchored to exact panel bottom-centers and card top-centers.
    panel_axes = [ax_v, ax_a, ax_r]
    arrow_rads = [0.09, 0.0, -0.09]
    for i, p_ax in enumerate(panel_axes):
        pb = p_ax.get_position()
        sx, sy = fig_to_ax(ax_s, fig, pb.x0 + (pb.width / 2), pb.y0 - 0.01)
        cx, cy, cw, ch, _, _ = cards[i]
        ex = cx + (cw / 2)
        ey = cy + ch + 0.005
        arr = FancyArrowPatch(
            (sx, sy),
            (ex, ey),
            transform=ax_s.transAxes,
            arrowstyle="-|>",
            mutation_scale=14,
            lw=1.7,
            color="#334155",
            connectionstyle=f"arc3,rad={arrow_rads[i]}",
            clip_on=False,
        )
        ax_s.add_patch(arr)

    # Connector to missing-modality card from the combined modality region.
    pb_r = ax_r.get_position()
    sx_m, sy_m = fig_to_ax(ax_s, fig, pb_r.x0 + (0.86 * pb_r.width), pb_r.y0 - 0.01)
    cx_m, cy_m, cw_m, ch_m, _, _ = cards[3]
    ex_m = cx_m + (cw_m / 2)
    ey_m = cy_m + ch_m + 0.005
    arr_m = FancyArrowPatch(
        (sx_m, sy_m),
        (ex_m, ey_m),
        transform=ax_s.transAxes,
        arrowstyle="-|>",
        mutation_scale=12,
        lw=1.5,
        color="#64748b",
        connectionstyle="arc3,rad=-0.08",
        clip_on=False,
    )
    ax_s.add_patch(arr_m)

    if diverse_modalities:
        footer = (
            "Real-data modality examples "
            f"(vision ep {v_row['episode_id']} cond={v_row['vision_condition']} | "
            f"audio ep {a_row['episode_id']} SNR={float(a_row['audio_snr_db']):.1f} dB | "
            f"radar ep {r_row['episode_id']} clutter={float(r_row['radar_clutter_level']):.2f})"
        )
    else:
        footer = (
            f"Real-data example episode {v_row['episode_id']} ({v_row['split']}, condition={v_row['vision_condition']}, "
            f"SNR={float(v_row['audio_snr_db']):.1f} dB, clutter={float(v_row['radar_clutter_level']):.2f})"
        )
    fig.text(0.01, 0.012, footer, fontsize=10, color="#475569")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return out_png


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, default=Path("data/processed/episodes/episodes.csv"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("final_isef_package/figures/data_conditions_episode_stressors.png"),
    )
    parser.add_argument("--episode-id", type=str, default=None)
    parser.add_argument("--diverse-modalities", action="store_true")
    parser.add_argument("--vision-episode-id", type=str, default=None)
    parser.add_argument("--audio-episode-id", type=str, default=None)
    parser.add_argument("--radar-episode-id", type=str, default=None)
    args = parser.parse_args()

    out = build_figure(
        episodes_csv=args.episodes,
        out_png=args.out,
        episode_id=args.episode_id,
        diverse_modalities=bool(args.diverse_modalities),
        vision_episode_id=args.vision_episode_id,
        audio_episode_id=args.audio_episode_id,
        radar_episode_id=args.radar_episode_id,
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
