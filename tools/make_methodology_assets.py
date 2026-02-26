from __future__ import annotations

import argparse
import csv
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


BG = "#F5F7FA"
WHITE = "#FFFFFF"
BORDER = "#C7D0DB"
HEADER = "#274C77"
TEXT_DARK = "#17232E"
TEXT_MUTED = "#5A6B7D"
ACCENT_A = "#1E88E5"
ACCENT_B = "#00A3A3"
ACCENT_C = "#6C5CE7"


def set_plot_style() -> None:
    plt.rcParams["figure.facecolor"] = BG
    plt.rcParams["axes.facecolor"] = BG
    plt.rcParams["savefig.facecolor"] = BG
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["font.size"] = 12


def wrap(s: str, width: int) -> str:
    return "\n".join(textwrap.wrap(s, width=width))


def intersection_area(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    x0 = max(ax0, bx0)
    y0 = max(ay0, by0)
    x1 = min(ax1, bx1)
    y1 = min(ay1, by1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float((x1 - x0) * (y1 - y0))


def qa_text_layout(fig: plt.Figure, fig_name: str) -> list[str]:
    issues: list[str] = []
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fb = fig.bbox
    texts = []
    for t in fig.findobj(match=plt.Text):
        if not t.get_visible():
            continue
        txt = t.get_text()
        if txt is None or str(txt).strip() == "":
            continue
        bb = t.get_window_extent(renderer=renderer).expanded(1.02, 1.08)
        texts.append((str(txt).strip(), bb))
        if bb.x0 < fb.x0 - 2 or bb.y0 < fb.y0 - 2 or bb.x1 > fb.x1 + 2 or bb.y1 > fb.y1 + 2:
            issues.append(f"{fig_name}: text out of bounds -> '{str(txt)[:80]}'")

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            ai, bi = texts[i], texts[j]
            a = (ai[1].x0, ai[1].y0, ai[1].x1, ai[1].y1)
            b = (bi[1].x0, bi[1].y0, bi[1].x1, bi[1].y1)
            inter = intersection_area(a, b)
            if inter > 48.0:
                issues.append(f"{fig_name}: overlapping text -> '{ai[0][:40]}' with '{bi[0][:40]}'")
    return issues


def save_checked(fig: plt.Figure, out: Path, fig_name: str, dpi: int = 280) -> list[str]:
    issues = qa_text_layout(fig, fig_name)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return issues


def draw_stage_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    lines: list[str],
    header_color: str = HEADER,
    body_color: str = WHITE,
) -> None:
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.008,rounding_size=0.02",
        facecolor=body_color,
        edgecolor=BORDER,
        linewidth=1.5,
        transform=ax.transAxes,
    )
    ax.add_patch(box)
    hh = 0.22 * h
    hr = Rectangle((x, y + h - hh), w, hh, transform=ax.transAxes, facecolor=header_color, edgecolor="none")
    hr.set_clip_path(box)
    ax.add_patch(hr)
    ax.text(
        x + 0.02 * w,
        y + h - 0.5 * hh,
        title,
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=13,
        fontweight="bold",
        color=WHITE,
    )
    y0 = y + h - hh - 0.10 * h
    step = 0.17 * h
    for i, ln in enumerate(lines):
        ax.text(
            x + 0.04 * w,
            y0 - i * step,
            ln,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=10.8,
            color=TEXT_DARK,
        )


def add_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], rad: float = 0.0) -> None:
    arr = FancyArrowPatch(
        start,
        end,
        transform=ax.transAxes,
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=1.7,
        color="#3C4A5A",
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arr)


def load_array_2d(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim == 2:
        return arr.astype(np.float32)
    if arr.ndim == 3:
        return arr[0].astype(np.float32)
    raise ValueError(f"Expected 2D/3D array for {path}, got shape {arr.shape}")


def extract_vision_strip(video_path: Path, t0: float, t1: float, n_frames: int = 2) -> np.ndarray:
    if not video_path.exists():
        raise FileNotFoundError(f"Missing video file: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    times = np.linspace(t0 + 0.35, max(t0 + 0.35, t1 - 0.35), n_frames)
    frames: list[np.ndarray] = []
    for t in times:
        idx = int(round(float(t) * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            frame = np.zeros((360, 640, 3), dtype=np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (520, 300), interpolation=cv2.INTER_AREA)
        cv2.putText(
            frame,
            f"t={float(t):.2f}s",
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"t={float(t):.2f}s",
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (20, 30, 40),
            1,
            cv2.LINE_AA,
        )
        frames.append(frame)
    cap.release()

    gap = 12
    spacer = np.full((frames[0].shape[0], gap, 3), 250, dtype=np.uint8)
    strip = frames[0]
    for fr in frames[1:]:
        strip = np.concatenate([strip, spacer, fr], axis=1)
    return strip


def likely_used_episode_ids(df: pd.DataFrame) -> set[str]:
    used: set[str] = set()
    pool = df[
        (df["split"] == "test")
        & (df["label_drone"] == 1)
        & (df["miss_vis"] == 1)
        & (df["miss_aud"] == 1)
        & (df["miss_rad"] == 1)
    ].copy()
    if pool.empty:
        return used

    priority = {"hard_mix": 0, "low_light": 1, "wind_noise": 2, "radar_cluttered": 3, "normal": 4}
    p1 = pool.copy()
    p1["prio"] = p1["vision_condition"].astype(str).map(priority).fillna(9)
    p1 = p1.sort_values(["prio", "episode_id"], ascending=[True, True])
    used.add(str(p1.iloc[0]["episode_id"]))

    p2 = pool.copy()
    vis_candidates = p2[p2["vision_condition"].isin(["low_light", "hard_mix"])].copy()
    if vis_candidates.empty:
        vis_candidates = p2.copy()
    vis_candidates = vis_candidates.sort_values(["vision_condition", "episode_id"])
    v_id = str(vis_candidates.iloc[0]["episode_id"])
    used.add(v_id)

    a_candidates = p2[~p2["episode_id"].astype(str).isin(used)].copy()
    if a_candidates.empty:
        a_candidates = p2.copy()
    a_candidates = a_candidates.sort_values(["audio_snr_db", "episode_id"], ascending=[True, True])
    a_id = str(a_candidates.iloc[0]["episode_id"])
    used.add(a_id)

    r_candidates = p2[~p2["episode_id"].astype(str).isin(used)].copy()
    if r_candidates.empty:
        r_candidates = p2.copy()
    r_candidates = r_candidates.sort_values(["radar_clutter_level", "episode_id"], ascending=[False, True])
    used.add(str(r_candidates.iloc[0]["episode_id"]))

    used.update({"003610", "003705", "003731", "003798"})
    return used


@dataclass
class M2Samples:
    vision_row: pd.Series
    audio_row: pd.Series
    radar_row: pd.Series
    excluded_ids: list[str]


def pick_m2_samples(episodes: pd.DataFrame) -> M2Samples:
    pool = episodes[
        (episodes["split"] == "test")
        & (episodes["label_drone"] == 1)
        & (episodes["miss_vis"] == 1)
        & (episodes["miss_aud"] == 1)
        & (episodes["miss_rad"] == 1)
    ].copy()
    if pool.empty:
        raise RuntimeError("No all-present positive test episodes available for Method M2.")

    for c in ["video_path", "audio_path", "radar_path"]:
        pool = pool[pool[c].astype(str).map(lambda p: Path(p).exists())]
    if pool.empty:
        raise RuntimeError("No eligible episodes have all modality files present on disk.")

    exclude = likely_used_episode_ids(episodes)
    filtered = pool[~pool["episode_id"].astype(str).isin(exclude)].copy()
    if filtered.empty:
        filtered = pool.copy()

    vis = filtered[filtered["video_path"].astype(str).str.contains(r"\\Video_V\\", case=False, regex=True)].copy()
    if vis.empty:
        vis = filtered.copy()
    vis = vis.sort_values(["episode_id"])
    v_row = vis.iloc[0]

    aud = filtered[filtered["episode_id"].astype(str) != str(v_row["episode_id"])].copy()
    if aud.empty:
        aud = filtered.copy()
    aud = aud.sort_values(["audio_snr_db", "episode_id"], ascending=[True, True])
    a_row = aud.iloc[0]

    rad = filtered[
        ~filtered["episode_id"].astype(str).isin({str(v_row["episode_id"]), str(a_row["episode_id"])})
    ].copy()
    if rad.empty:
        rad = filtered.copy()
    rad = rad.sort_values(["radar_clutter_level", "episode_id"], ascending=[False, True])
    r_row = rad.iloc[0]

    return M2Samples(v_row, a_row, r_row, sorted(exclude))


def make_m1_pipeline(out_path: Path) -> list[str]:
    fig = plt.figure(figsize=(20, 8.1), dpi=280, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1], facecolor=BG)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(
        0.5,
        0.96,
        "Methodology Pipeline: Raw Data to Episode-Level Fusion Decision",
        ha="center",
        va="center",
        fontsize=24,
        fontweight="bold",
        color=TEXT_DARK,
    )

    draw_stage_box(
        ax,
        0.02,
        0.68,
        0.22,
        0.22,
        "1) Dataset Verification",
        [
            "TSMS + DroneDetectionThesis",
            "download_tsms_full.ps1",
            "verify_tsms.py / verify_dronedetect.py",
        ],
    )
    draw_stage_box(
        ax,
        0.265,
        0.68,
        0.22,
        0.22,
        "2) Label + Preprocess",
        [
            "convert_mat_labels.py",
            "extract_frames.py / audio_windows.py",
            "radar_extract_tsms.py",
        ],
    )
    draw_stage_box(
        ax,
        0.51,
        0.68,
        0.22,
        0.22,
        "3) Episode Builder",
        [
            "episode_builder.py",
            "split-safe source IDs",
            "data/processed/episodes/episodes.csv",
        ],
    )
    draw_stage_box(
        ax,
        0.75,
        0.66,
        0.22,
        0.14,
        "Callouts",
        [
            "episode = 2.0 s (t0=0, t1=2)",
            "masks: miss_vis/miss_aud/miss_rad",
        ],
        header_color="#2E6A8E",
        body_color="#ECF6FF",
    )

    draw_stage_box(
        ax,
        0.03,
        0.39,
        0.20,
        0.20,
        "Vision (UNI-V)",
        [
            "YOLOv8n on episode frames",
            "logit_vis + q_vis(5)",
            "vision_presence.csv",
        ],
        header_color=ACCENT_A,
    )
    draw_stage_box(
        ax,
        0.275,
        0.39,
        0.20,
        0.20,
        "Audio (UNI-A)",
        [
            "AudioCNN on log-mel windows",
            "logit_aud + q_aud(4)",
            "BCEWithLogits",
        ],
        header_color=ACCENT_B,
    )
    draw_stage_box(
        ax,
        0.52,
        0.39,
        0.20,
        0.20,
        "Radar (UNI-R)",
        [
            "RadarCNN on 2D maps",
            "logit_rad + q_rad(3)",
            "BCEWithLogits",
        ],
        header_color=ACCENT_C,
    )

    draw_stage_box(
        ax,
        0.75,
        0.38,
        0.22,
        0.23,
        "Feature Cache",
        [
            "features_cache.npz",
            "episode_id, y, logits, quality, mask",
            "fusion input dimension: 24",
        ],
        header_color="#5E60CE",
        body_color="#F2F1FF",
    )
    draw_stage_box(
        ax,
        0.37,
        0.12,
        0.31,
        0.18,
        "Fusion Variants (Logit-Level)",
        [
            "F-AVG / F-STATIC",
            "F-GATE / F-GATE-NOQ",
            "hard mask before softmax",
        ],
        header_color="#126E82",
    )
    draw_stage_box(
        ax,
        0.70,
        0.12,
        0.27,
        0.18,
        "Final Output",
        [
            "results/preds_fusion.csv",
            "episode_id, p_* , w_*",
            "test predictions + gate weights",
        ],
        header_color="#2A9D8F",
    )

    add_arrow(ax, (0.24, 0.79), (0.265, 0.79))
    add_arrow(ax, (0.485, 0.79), (0.51, 0.79))

    # Aligned vertical fan-out from episode builder to unimodal branches.
    add_arrow(ax, (0.13, 0.68), (0.13, 0.59))
    add_arrow(ax, (0.375, 0.68), (0.375, 0.59))
    add_arrow(ax, (0.62, 0.68), (0.62, 0.59))

    # Three short inter-box arrows (no line crossing inside cards).
    add_arrow(ax, (0.23, 0.50), (0.275, 0.50))  # Vision -> Audio gap
    add_arrow(ax, (0.475, 0.50), (0.52, 0.50))  # Audio -> Radar gap
    add_arrow(ax, (0.72, 0.50), (0.75, 0.50))   # Radar -> Feature cache gap

    # Feature cache to fusion and final outputs.
    add_arrow(ax, (0.86, 0.38), (0.54, 0.30))
    add_arrow(ax, (0.68, 0.21), (0.70, 0.21))

    ax.text(
        0.17,
        0.33,
        wrap(
            "Fusion feature vector x = [(logit,p,entropy)x3, q_vis(5), q_aud(4), q_rad(3), mask(3)]",
            66,
        ),
        ha="left",
        va="center",
        fontsize=10.5,
        color=TEXT_DARK,
        bbox={"facecolor": "#EAF4FF", "edgecolor": "#9AB3C7", "boxstyle": "round,pad=0.35"},
    )

    ax.text(
        0.03,
        0.02,
        "Figure M1. End-to-end multimodal pipeline from verified raw datasets to final episode-level fused predictions.",
        ha="left",
        va="bottom",
        fontsize=11.5,
        color=TEXT_MUTED,
    )
    return save_checked(fig, out_path, "method_M1")


def make_m2_modality_strip(episodes_csv: Path, out_path: Path) -> tuple[list[str], M2Samples]:
    episodes = pd.read_csv(episodes_csv, dtype={"episode_id": str})
    samples = pick_m2_samples(episodes)

    vision_row = samples.vision_row
    audio_row = samples.audio_row
    radar_row = samples.radar_row

    vis_strip = extract_vision_strip(
        Path(str(vision_row["video_path"])),
        float(vision_row["t0_sec"]),
        float(vision_row["t1_sec"]),
        n_frames=2,
    )
    audio = load_array_2d(Path(str(audio_row["audio_path"])))
    radar = load_array_2d(Path(str(radar_row["radar_path"])))

    fig = plt.figure(figsize=(20, 8.5), dpi=280, facecolor=BG)
    fig.suptitle(
        "What Each Modality Sees in a 2.0 s Episode (Distinct Real Samples)",
        fontsize=23,
        fontweight="bold",
        y=0.96,
        color=TEXT_DARK,
    )

    ax_v = fig.add_axes([0.03, 0.24, 0.31, 0.62])
    ax_a = fig.add_axes([0.36, 0.24, 0.29, 0.62])
    ax_r = fig.add_axes([0.67, 0.24, 0.30, 0.62])
    for ax in [ax_v, ax_a, ax_r]:
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(1.6)
            sp.set_edgecolor(BORDER)

    ax_v.imshow(vis_strip)
    ax_v.set_title("Vision sample (new clip)", fontsize=14, fontweight="bold", color=TEXT_DARK, pad=10)

    ax_a.imshow(audio, origin="lower", aspect="auto", cmap="magma")
    ax_a.set_title("Audio sample (new log-mel)", fontsize=14, fontweight="bold", color=TEXT_DARK, pad=10)

    ax_r.imshow(radar, origin="lower", aspect="auto", cmap="viridis")
    ax_r.set_title("Radar sample (new 2D map)", fontsize=14, fontweight="bold", color=TEXT_DARK, pad=10)

    fig.text(0.185, 0.17, "Vision: 2.0 s window (30 FPS source)", ha="center", fontsize=12, color=TEXT_DARK)
    fig.text(0.505, 0.17, "Audio: Log-mel (16 kHz, 128x101)", ha="center", fontsize=12, color=TEXT_DARK)
    fig.text(0.82, 0.17, "Radar: Normalized 2D map (128x128)", ha="center", fontsize=12, color=TEXT_DARK)

    foot = (
        f"Samples for visual diversity: vision ep {vision_row['episode_id']} | "
        f"audio ep {audio_row['episode_id']} | radar ep {radar_row['episode_id']}. "
        "Selected to avoid sample reuse from earlier schematic figures."
    )
    fig.text(0.03, 0.08, foot, ha="left", fontsize=10.5, color=TEXT_MUTED)
    fig.text(
        0.03,
        0.03,
        "Figure M2. Per-modality episode representations shown as real inputs to each unimodal branch.",
        ha="left",
        fontsize=11.5,
        color=TEXT_MUTED,
    )
    return save_checked(fig, out_path, "method_M2"), samples


def draw_icon_vision(ax: plt.Axes, cx: float, cy: float, s: float) -> None:
    ax.add_patch(Rectangle((cx - 0.05 * s, cy - 0.025 * s), 0.10 * s, 0.05 * s, facecolor="none", edgecolor=ACCENT_A, lw=2))
    ax.add_patch(Circle((cx, cy), 0.016 * s, facecolor="none", edgecolor=ACCENT_A, lw=2))


def draw_icon_audio(ax: plt.Axes, cx: float, cy: float, s: float) -> None:
    xs = np.linspace(cx - 0.05 * s, cx + 0.05 * s, 60)
    ys = cy + 0.014 * s * np.sin(np.linspace(0, 4 * np.pi, 60))
    ax.plot(xs, ys, color=ACCENT_B, lw=2)


def draw_icon_radar(ax: plt.Axes, cx: float, cy: float, s: float) -> None:
    theta = np.linspace(0, np.pi, 80)
    for r in [0.02, 0.035, 0.05]:
        x = cx + r * s * np.cos(theta)
        y = cy + r * s * np.sin(theta)
        ax.plot(x, y, color=ACCENT_C, lw=2)
    ax.plot([cx, cx], [cy, cy + 0.05 * s], color=ACCENT_C, lw=2)


def make_m3_unimodal_cards(out_path: Path) -> list[str]:
    fig = plt.figure(figsize=(20, 6.5), dpi=280, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1], facecolor=BG)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.5, 0.94, "Unimodal Branches (UNI-V / UNI-A / UNI-R)", ha="center", va="center", fontsize=24, fontweight="bold", color=TEXT_DARK)

    cards = [
        (0.03, 0.13, 0.30, 0.74, "Vision Branch (UNI-V)", ACCENT_A, [
            "YOLOv8n detector over frames in [t0, t1).",
            "Episode score = max drone confidence across frames.",
            "Converted to episode logit (vision_presence.csv).",
            "Used as logit_vis + quality q_vis(5).",
        ]),
        (0.35, 0.13, 0.30, 0.74, "Audio Branch (UNI-A)", ACCENT_B, [
            "Input tensor shape: [1 x 128 x 101] log-mel.",
            "AudioCNN: conv blocks -> pooled -> FC head.",
            "Objective: BCEWithLogits (pos_weight).",
            "Output: episode logit_aud.",
        ]),
        (0.67, 0.13, 0.30, 0.74, "Radar Branch (UNI-R)", ACCENT_C, [
            "Input tensor shape: [1 x 128 x 128] 2D map.",
            "Residual CNN with two downsampling stages.",
            "Objective: BCEWithLogits (pos_weight).",
            "Output: episode logit_rad.",
        ]),
    ]

    for i, (x, y, w, h, title, col, lines) in enumerate(cards):
        draw_stage_box(ax, x, y, w, h, title, lines, header_color=col, body_color=WHITE)
        cx = x + 0.90 * w
        cy = y + 0.86 * h
        scale = w
        if i == 0:
            draw_icon_vision(ax, cx, cy, scale)
        elif i == 1:
            draw_icon_audio(ax, cx, cy, scale)
        else:
            draw_icon_radar(ax, cx, cy, scale)

    ax.text(
        0.03,
        0.025,
        "Figure M3. Three unimodal branches produce calibrated episode logits and modality-specific quality signals for fusion.",
        ha="left",
        fontsize=11.5,
        color=TEXT_MUTED,
    )
    return save_checked(fig, out_path, "method_M3")


def make_m4_fusion_gate(out_path: Path) -> list[str]:
    fig = plt.figure(figsize=(20, 7.7), dpi=280, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1], facecolor=BG)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.5, 0.95, "Adaptive Fusion Gate (F-GATE): Mechanism and Equations", ha="center", va="center", fontsize=24, fontweight="bold", color=TEXT_DARK)

    draw_stage_box(
        ax,
        0.03,
        0.50,
        0.33,
        0.35,
        "Input Feature Vector x (24-D)",
        [
            "[(logit,p,entropy) x3]",
            "q_vis(5), q_aud(4), q_rad(3)",
            "mask(3) -> [m_v, m_a, m_r]",
        ],
        header_color="#1D3557",
    )

    draw_stage_box(
        ax,
        0.40,
        0.58,
        0.21,
        0.24,
        "Gate Network",
        [
            "g = MLP(x)",
            "g in R^3 (one score per modality)",
            "hidden layers: 96 -> 48",
        ],
        header_color="#2A9D8F",
    )
    draw_stage_box(
        ax,
        0.40,
        0.34,
        0.21,
        0.21,
        "Masking Before Softmax",
        [
            "if m_k=0: g_k <- -1e9",
            "hard-exclude missing modality",
            "w = softmax(masked(g))",
        ],
        header_color="#E76F51",
    )
    draw_stage_box(
        ax,
        0.65,
        0.48,
        0.32,
        0.37,
        "Adaptive Weights + Fused Logit",
        [
            "weights: w_v, w_a, w_r",
            "z_fuse = sum_k w_k * l_k",
            "p_fuse = sigmoid(z_fuse)",
        ],
        header_color="#457B9D",
    )
    labels = ["Vision", "Audio", "Radar"]
    vals = [0.63, 0.22, 0.15]
    for i, (lab, val) in enumerate(zip(labels, vals)):
        y = 0.57 - i * 0.055
        bar_x = 0.77
        bar_w = 0.15
        ax.text(0.69, y + 0.012, lab, fontsize=10.5, color=TEXT_DARK, ha="left", va="center")
        ax.add_patch(Rectangle((bar_x, y), bar_w, 0.018, facecolor="#D9E2EC", edgecolor="none"))
        ax.add_patch(Rectangle((bar_x, y), bar_w * val, 0.018, facecolor="#2A9D8F", edgecolor="none"))
        ax.text(0.935, y + 0.009, f"{100.0 * val:.0f}%", fontsize=9.5, color=TEXT_DARK, ha="right", va="center")

    draw_stage_box(
        ax,
        0.65,
        0.25,
        0.32,
        0.18,
        "F-GATE-NOQ Inset",
        [
            "same gate architecture",
            "quality block zeroed: x[:,9:21] = 0",
            "tests value of quality signals",
        ],
        header_color="#6D597A",
        body_color="#F7F1FB",
    )

    eq_x, eq_y, eq_w, eq_h = 0.03, 0.06, 0.94, 0.20
    eq_box = FancyBboxPatch(
        (eq_x, eq_y),
        eq_w,
        eq_h,
        boxstyle="round,pad=0.008,rounding_size=0.02",
        facecolor="#EFF9F8",
        edgecolor=BORDER,
        linewidth=1.5,
        transform=ax.transAxes,
    )
    ax.add_patch(eq_box)
    hh = 0.24 * eq_h
    eq_header = Rectangle((eq_x, eq_y + eq_h - hh), eq_w, hh, transform=ax.transAxes, facecolor="#264653", edgecolor="none")
    eq_header.set_clip_path(eq_box)
    ax.add_patch(eq_header)
    ax.text(
        eq_x + 0.012,
        eq_y + eq_h - 0.5 * hh,
        "Core Equations (Logit-Level Fusion)",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=13,
        fontweight="bold",
        color=WHITE,
    )
    ax.text(eq_x + 0.035, eq_y + 0.100, r"$p_k = \sigma(l_k)$", transform=ax.transAxes, fontsize=12, color=TEXT_DARK, ha="left", va="center")
    ax.text(
        eq_x + 0.035,
        eq_y + 0.060,
        r"$w = \mathrm{softmax}(\mathrm{masked}(g)),\ \mathrm{masked}(g_k)=g_k\ \mathrm{if}\ m_k=1,\ -10^9\ \mathrm{if}\ m_k=0$",
        transform=ax.transAxes,
        fontsize=11.0,
        color=TEXT_DARK,
        ha="left",
        va="center",
    )
    ax.text(
        eq_x + 0.035,
        eq_y + 0.020,
        r"$z_{\mathrm{fuse}} = \sum_{k \in \{v,a,r\}} w_k l_k,\quad p_{\mathrm{fuse}}=\sigma(z_{\mathrm{fuse}})$",
        transform=ax.transAxes,
        fontsize=11.0,
        color=TEXT_DARK,
        ha="left",
        va="center",
    )

    add_arrow(ax, (0.36, 0.69), (0.40, 0.69))
    add_arrow(ax, (0.505, 0.58), (0.505, 0.55))
    add_arrow(ax, (0.61, 0.69), (0.65, 0.69))
    add_arrow(ax, (0.61, 0.47), (0.65, 0.62))
    add_arrow(ax, (0.505, 0.34), (0.505, 0.26))

    ax.text(
        0.02,
        0.03,
        "Figure M4. F-GATE computes mask-aware adaptive modality weights and performs hard-excluded logit-level fusion.",
        ha="left",
        fontsize=11.5,
        color=TEXT_MUTED,
    )
    return save_checked(fig, out_path, "method_M4")


def draw_modality_triplet(ax: plt.Axes, x: float, y: float, dropped: Iterable[int]) -> None:
    names = ["V", "A", "R"]
    colors = [ACCENT_A, ACCENT_B, ACCENT_C]
    dropped_set = set(dropped)
    for i in range(3):
        cx = x + i * 0.095
        ax.add_patch(Circle((cx, y), 0.027, facecolor=WHITE, edgecolor=colors[i], lw=2))
        ax.text(cx, y, names[i], ha="center", va="center", fontsize=12, color=colors[i], fontweight="bold")
        if i in dropped_set:
            ax.plot([cx - 0.020, cx + 0.020], [y - 0.020, y + 0.020], color="#D62828", lw=2.2)
            ax.plot([cx - 0.020, cx + 0.020], [y + 0.020, y - 0.020], color="#D62828", lw=2.2)


def make_m5_modality_dropout(out_path: Path) -> list[str]:
    fig = plt.figure(figsize=(18, 7.8), dpi=280, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1], facecolor=BG)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.5, 0.92, "Fusion Training Robustness via Modality Dropout", ha="center", va="center", fontsize=24, fontweight="bold", color=TEXT_DARK)

    cards = [
        (0.06, 0.24, 0.26, 0.52, "all_present", "prob = 0.87", []),
        (0.37, 0.24, 0.26, 0.52, "drop_one", "prob = 0.10", [1]),
        (0.68, 0.24, 0.26, 0.52, "drop_two", "prob = 0.03", [0, 2]),
    ]
    for x, y, w, h, title, prob, dropped in cards:
        draw_stage_box(
            ax,
            x,
            y,
            w,
            h,
            title,
            [
                f"{prob}",
                "mask channels modified in-batch",
                "logits kept, hard-mask before softmax",
            ],
            header_color="#1D3557",
            body_color=WHITE,
        )
        draw_modality_triplet(ax, x + 0.07, y + 0.11, dropped=dropped)

    add_arrow(ax, (0.32, 0.50), (0.37, 0.50))
    add_arrow(ax, (0.63, 0.50), (0.68, 0.50))

    ax.text(
        0.50,
        0.14,
        "Constraint: enforce at least one modality present in every training sample.",
        ha="center",
        va="center",
        fontsize=13,
        color=TEXT_DARK,
        bbox={"facecolor": "#E8F4FF", "edgecolor": "#8CB3D9", "boxstyle": "round,pad=0.35"},
    )
    ax.text(
        0.50,
        0.08,
        "Implementation note: mask value 0 excludes modality by setting gate logit to -1e9 before softmax.",
        ha="center",
        va="center",
        fontsize=11.5,
        color=TEXT_MUTED,
    )
    ax.text(
        0.03,
        0.02,
        "Figure M5. Modality-dropout training improves robustness to missing sensors without changing unimodal logits.",
        ha="left",
        fontsize=11.5,
        color=TEXT_MUTED,
    )
    return save_checked(fig, out_path, "method_M5")


def draw_table_figure(
    title: str,
    caption: str,
    columns: list[str],
    rows: list[list[str]],
    out_png: Path,
    out_csv: Path | None = None,
    col_widths: list[float] | None = None,
) -> list[str]:
    fig = plt.figure(figsize=(22, 5.8), dpi=280, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1], facecolor=BG)
    ax.axis("off")
    ax.text(0.5, 0.94, title, ha="center", va="center", fontsize=23, fontweight="bold", color=TEXT_DARK)

    bbox = [0.02, 0.14, 0.96, 0.74]
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="left",
        colLoc="left",
        colWidths=col_widths,
        loc="center",
        bbox=bbox,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(BORDER)
        cell.set_linewidth(1.0)
        if r == 0:
            cell.set_facecolor(HEADER)
            cell.get_text().set_color(WHITE)
            cell.get_text().set_fontweight("bold")
            cell.get_text().set_fontsize(12.0)
        else:
            cell.set_facecolor(WHITE if r % 2 == 1 else "#F9FBFE")
            cell.get_text().set_color(TEXT_DARK)
            cell.get_text().set_fontsize(11.2)

    ax.text(0.02, 0.035, caption, ha="left", va="bottom", fontsize=11.5, color=TEXT_MUTED)
    issues = save_checked(fig, out_png, out_png.stem)

    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            writer.writerows(rows)
    return issues


def make_table_m1(out_png: Path, out_csv: Path) -> list[str]:
    columns = ["Modality", "Raw Source", "Episode Window", "Representation", "Tensor Shape", "Normalization"]
    rows = [
        [
            "Vision",
            "DroneDetectionThesis videos",
            "2.0 s",
            "YOLOv8n frame scan",
            "frame stream",
            "YOLO internal norm/letterbox",
        ],
        [
            "Audio",
            "DroneDetectionThesis WAV",
            "2.0 s",
            "log-mel spectrogram",
            "128 x 101",
            "per-window z-score",
        ],
        [
            "Radar",
            "TSMS-Drone FMCW radar",
            "2.0 s sample",
            "normalized 2D radar map",
            "128 x 128",
            "per-sample z-score",
        ],
    ]
    return draw_table_figure(
        title="Table M1. Inputs and Preprocessing Summary",
        caption="Table M1. Canonical modality inputs and preprocessing used across all evaluation variants.",
        columns=columns,
        rows=rows,
        out_png=out_png,
        out_csv=out_csv,
        col_widths=[0.10, 0.22, 0.12, 0.18, 0.12, 0.20],
    )


def make_table_m2(out_png: Path, out_csv: Path) -> list[str]:
    columns = ["Variant", "Inputs used", "Weight rule", "Mask handling", "Fusion level"]
    rows = [
        [
            "F-AVG",
            "logits l_v,l_a,l_r + mask m",
            "w = m / sum(m)",
            "m_k=0 => w_k=0",
            "logit-level (z = sum w_k l_k)",
        ],
        [
            "F-STATIC",
            "logits + learned alpha",
            "w = softmax(mask(alpha))",
            "mask(alpha_k): alpha_k or -1e9",
            "logit-level",
        ],
        [
            "F-GATE",
            "x(24-D) + logits + mask",
            "w = softmax(mask(MLP(x)))",
            "hard mask before softmax",
            "logit-level",
        ],
        [
            "F-GATE-NOQ",
            "same as F-GATE",
            "same as F-GATE",
            "same hard mask",
            "quality dims zeroed: x[:,9:21]=0",
        ],
    ]
    return draw_table_figure(
        title="Table M2. Fusion Variant Definitions",
        caption="Table M2. Exact fusion rules used for averaging, static weighting, and adaptive gating.",
        columns=columns,
        rows=rows,
        out_png=out_png,
        out_csv=out_csv,
        col_widths=[0.12, 0.23, 0.20, 0.20, 0.21],
    )


def make_table_m3(out_png: Path, out_csv: Path) -> list[str]:
    columns = ["Stage", "Model", "Objective", "Optimizer", "LR", "Batch", "Epochs", "Selection criterion", "Output artifact"]
    rows = [
        [
            "Vision",
            "YOLOv8n",
            "YOLO detection loss",
            "MuSGD (auto)",
            "auto",
            "16",
            "100",
            "best val mAP50-95",
            "results/metrics_vision.json + best.pt",
        ],
        [
            "Audio",
            "AudioCNN",
            "BCEWithLogits",
            "AdamW",
            "1e-3",
            "64",
            "50",
            "best val AP",
            "results/checkpoints/best_audio.pt",
        ],
        [
            "Radar",
            "RadarCNN (residual)",
            "BCEWithLogits",
            "AdamW",
            "1e-3",
            "64",
            "50",
            "best val AP",
            "results/checkpoints/best_radar.pt",
        ],
        [
            "Fusion",
            "F-GATE / F-GATE-NOQ",
            "BCEWithLogits on fused logit",
            "AdamW",
            "5e-4",
            "256",
            "260",
            "best val PR-AUC (gate)",
            "fusion_gate.pt, fusion_gate_noq.pt",
        ],
    ]
    return draw_table_figure(
        title="Table M3. Training Protocol Summary (Seed=42)",
        caption="Table M3. Reproducible training settings from the final run configuration and logs.",
        columns=columns,
        rows=rows,
        out_png=out_png,
        out_csv=out_csv,
        col_widths=[0.07, 0.12, 0.13, 0.10, 0.07, 0.06, 0.06, 0.15, 0.20],
    )


def qa_image_files(paths: list[Path]) -> list[str]:
    issues: list[str] = []
    for p in paths:
        if not p.exists():
            issues.append(f"missing output: {p}")
            continue
        if p.stat().st_size <= 0:
            issues.append(f"empty file: {p}")
            continue
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im2:
                w, h = im2.size
                if w < 1400 or h < 500:
                    issues.append(f"low resolution: {p} -> {w}x{h}")
        except Exception as exc:  # noqa: BLE001
            issues.append(f"unreadable image: {p} ({exc})")
    return issues


def write_manifest(manifest_path: Path, samples: M2Samples, outputs: list[tuple[str, Path]]) -> None:
    lines = [
        "Methodology Assets Manifest",
        "==========================",
        "",
        "M2 real-sample IDs (non-reused intent):",
        f"- vision episode_id: {samples.vision_row['episode_id']}",
        f"- audio episode_id: {samples.audio_row['episode_id']}",
        f"- radar episode_id: {samples.radar_row['episode_id']}",
        "",
        "Likely excluded IDs from prior schematic picks:",
        ", ".join(samples.excluded_ids[:30]) + (" ..." if len(samples.excluded_ids) > 30 else ""),
        "",
        "Outputs:",
    ]
    for purpose, path in outputs:
        lines.append(f"- {path.as_posix()} :: {purpose}")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    set_plot_style()
    out_dir = Path(args.out_dir)
    episodes_csv = Path(args.episodes_csv)
    if not episodes_csv.exists():
        raise FileNotFoundError(f"episodes.csv not found: {episodes_csv}")

    qa_issues: list[str] = []
    outputs: list[tuple[str, Path]] = []

    f_m1 = out_dir / "method_M1.png"
    qa_issues += make_m1_pipeline(f_m1)
    outputs.append(("Figure M1 anchor pipeline diagram", f_m1))

    f_m2 = out_dir / "method_M2.png"
    q2, samples = make_m2_modality_strip(episodes_csv=episodes_csv, out_path=f_m2)
    qa_issues += q2
    outputs.append(("Figure M2 modality visual strip (new real samples)", f_m2))

    f_m3 = out_dir / "method_M3.png"
    qa_issues += make_m3_unimodal_cards(f_m3)
    outputs.append(("Figure M3 unimodal branch cards", f_m3))

    f_m4 = out_dir / "method_M4.png"
    qa_issues += make_m4_fusion_gate(f_m4)
    outputs.append(("Figure M4 adaptive fusion gate with formulas", f_m4))

    f_m5 = out_dir / "method_M5.png"
    qa_issues += make_m5_modality_dropout(f_m5)
    outputs.append(("Figure M5 modality-dropout robustness diagram", f_m5))

    t_m1_png = out_dir / "method_table_M1.png"
    t_m1_csv = out_dir / "method_table_M1.csv"
    qa_issues += make_table_m1(t_m1_png, t_m1_csv)
    outputs.append(("Table M1 inputs + preprocessing summary", t_m1_png))
    outputs.append(("Table M1 editable CSV", t_m1_csv))

    t_m2_png = out_dir / "method_table_M2.png"
    t_m2_csv = out_dir / "method_table_M2.csv"
    qa_issues += make_table_m2(t_m2_png, t_m2_csv)
    outputs.append(("Table M2 fusion variants definitions", t_m2_png))
    outputs.append(("Table M2 editable CSV", t_m2_csv))

    t_m3_png = out_dir / "method_table_M3.png"
    t_m3_csv = out_dir / "method_table_M3.csv"
    qa_issues += make_table_m3(t_m3_png, t_m3_csv)
    outputs.append(("Table M3 training protocol summary", t_m3_png))
    outputs.append(("Table M3 editable CSV", t_m3_csv))

    image_paths = [p for _, p in outputs if p.suffix.lower() == ".png"]
    qa_issues += qa_image_files(image_paths)

    manifest = out_dir / "method_manifest.txt"
    outputs.append(("Methodology outputs manifest", manifest))

    qa_report_path = out_dir / "method_qa_report.txt"
    qa_lines = [
        "Methodology Asset QA Report",
        "===========================",
        f"Output directory: {out_dir.as_posix()}",
        f"Total outputs: {len(outputs)}",
        "",
    ]
    if qa_issues:
        qa_lines.append("STATUS: FAIL")
        qa_lines.append("")
        qa_lines.append("Issues:")
        qa_lines.extend([f"- {x}" for x in qa_issues])
    else:
        qa_lines.append("STATUS: PASS")
        qa_lines.append("")
        qa_lines.append("Checks passed:")
        qa_lines.append("- text overlap/cutoff scan")
        qa_lines.append("- image readability/openability")
        qa_lines.append("- minimum resolution checks")
        qa_lines.append("- output existence checks")
    qa_report_path.write_text("\n".join(qa_lines) + "\n", encoding="utf-8")
    outputs.append(("Methodology QA report", qa_report_path))
    write_manifest(manifest, samples=samples, outputs=outputs)

    print(f"Wrote {len(outputs)} outputs.")
    for purpose, p in outputs:
        print(f"- {p.as_posix()} :: {purpose}")
    print(f"QA report: {qa_report_path.as_posix()}")

    if qa_issues:
        raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate methodology figures and tables for poster panel.")
    parser.add_argument("--episodes-csv", type=Path, default=Path("data/processed/episodes/episodes.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/figures"))
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
