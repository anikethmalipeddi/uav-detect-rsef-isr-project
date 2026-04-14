from __future__ import annotations

import json
import math
import re
import shutil
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image, ImageDraw, ImageOps


REPO_ROOT = Path(__file__).resolve().parents[3]
ASSET_ROOT = REPO_ROOT / "presentation_assets" / "final_virtual_video_images"
PNG_DIR = ASSET_ROOT / "final_png"
SVG_DIR = ASSET_ROOT / "final_svg"
SOURCE_WORK_DIR = ASSET_ROOT / "source_work"
SELECTED_DIR = ASSET_ROOT / "selected_project_inputs"
SUPPORT_DIR = ASSET_ROOT / "supporting_files"

FIGSIZE = (16, 9)
DPI = 150

TEXT = "#22303a"
TEXT_MUTED = "#556371"
BORDER = "#b7c4cf"
GRID = "#dce4ea"
BG_SOFT = "#f7fafc"
CARD_BG = "#ffffff"
SLATE = "#7f8c99"
SLATE_DARK = "#4a5b69"
VISION = "#2b78c4"
AUDIO = "#1b9e96"
RADAR = "#6a67c8"
FUSION = "#1f7f8c"
SUCCESS = "#2b8a63"
HILITE = "#0f766e"
WARNING = "#d97706"
RED = "#b94a48"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 14,
        "axes.titlesize": 26,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "svg.fonttype": "none",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


@dataclass(frozen=True)
class SelectedInput:
    key: str
    source_path: Path
    copy_name: str
    preview_name: str | None
    note: str


FINAL_ORDER = [
    "01_system_overview",
    "02_modality_challenges_combined",
    "02a_vision_challenge",
    "02b_audio_challenge",
    "02c_radar_challenge",
    "03_procedure_pipeline",
    "04_system_architecture",
    "05_prauc_results",
    "06_false_alarm_and_stress",
    "07_conclusion_summary_visual",
    "08_future_work_roadmap",
]


SELECTED_INPUTS: list[SelectedInput] = [
    SelectedInput(
        key="vision_drone",
        source_path=REPO_ROOT / "final_isef_package" / "verification" / "IR_DRONE_059_f000112_DRONE.png",
        copy_name="vision_drone_IR_DRONE_059_f000112_DRONE.png",
        preview_name=None,
        note="Verified IR drone frame used in the vision ambiguity panel.",
    ),
    SelectedInput(
        key="vision_bird",
        source_path=REPO_ROOT / "final_isef_package" / "verification" / "IR_BIRD_010_f000043_BIRD.png",
        copy_name="vision_bird_IR_BIRD_010_f000043_BIRD.png",
        preview_name=None,
        note="Verified IR bird frame used as the confusing negative in the vision ambiguity panel.",
    ),
    SelectedInput(
        key="audio_challenge",
        source_path=REPO_ROOT / "data" / "processed" / "audio_windows" / "DRONE_020_00000.npy",
        copy_name="audio_DRONE_020_00000.npy",
        preview_name="audio_DRONE_020_00000_preview.png",
        note="Processed 128x101 log-mel spectrogram window selected from test episode 003731 (source_audio_label=Drone, audio_snr_db=-25.20).",
    ),
    SelectedInput(
        key="radar_challenge",
        source_path=REPO_ROOT / "data" / "processed" / "radar_rd_real" / "rd_00047495.npy",
        copy_name="radar_rd_00047495.npy",
        preview_name="radar_rd_00047495_preview.png",
        note="Processed 128x128 range-Doppler map selected from test episode 003737 (source_radar_label=Drone, radar_clutter_level=4.05).",
    ),
]


def ensure_dirs() -> None:
    for path in [PNG_DIR, SVG_DIR, SOURCE_WORK_DIR, SELECTED_DIR, SUPPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def new_slide() -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(PNG_DIR / f"{stem}.png", dpi=DPI, facecolor="white")
    fig.savefig(SVG_DIR / f"{stem}.svg", facecolor="white")
    plt.close(fig)


def round_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    face: str = CARD_BG,
    edge: str = BORDER,
    lw: float = 1.8,
    radius: float = 0.025,
    alpha: float = 1.0,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
        alpha=alpha,
    )
    ax.add_patch(patch)
    return patch


def header_card(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    title: str,
    accent: str,
    body: list[str] | None = None,
    body_color: str = TEXT,
    title_size: float = 18,
    body_size: float = 13,
) -> None:
    round_box(ax, x, y, w, h, face=CARD_BG, edge=BORDER, lw=1.7)
    ax.add_patch(Rectangle((x, y + h - 0.065), w, 0.065, facecolor=accent, edgecolor="none"))
    ax.text(x + 0.015, y + h - 0.0325, title, ha="left", va="center", fontsize=title_size, color="white", weight="bold")
    if body:
        for i, line in enumerate(body):
            ax.text(
                x + 0.018,
                y + h - 0.09 - i * 0.053,
                line,
                ha="left",
                va="top",
                fontsize=body_size,
                color=body_color,
            )


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = SLATE_DARK,
    lw: float = 2.6,
    arrowstyle: str = "-|>",
    shrink_a: float = 0.0,
    shrink_b: float = 0.0,
    connectionstyle: str = "arc3,rad=0.0",
) -> None:
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=arrowstyle,
        linewidth=lw,
        color=color,
        mutation_scale=16,
        shrinkA=shrink_a,
        shrinkB=shrink_b,
        connectionstyle=connectionstyle,
    )
    ax.add_patch(patch)


def pill(ax: plt.Axes, x: float, y: float, text: str, face: str, edge: str | None = None, color: str = TEXT, size: float = 12) -> None:
    width = 0.012 * len(text) + 0.03
    round_box(ax, x, y, width, 0.04, face=face, edge=edge or face, lw=1.0, radius=0.02)
    ax.text(x + width / 2, y + 0.02, text, ha="center", va="center", fontsize=size, color=color, weight="bold")


def card_label(ax: plt.Axes, x: float, y: float, text: str, accent: str) -> None:
    pill(ax, x, y, text, face=accent, color="white", size=11)


def icon_camera(ax: plt.Axes, cx: float, cy: float, scale: float, color: str) -> None:
    w = 0.10 * scale
    h = 0.065 * scale
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, fill=False, lw=2.2, ec=color))
    ax.add_patch(Circle((cx, cy), 0.016 * scale, fill=False, lw=2.0, ec=color))
    ax.add_patch(Rectangle((cx - w / 2 + 0.012 * scale, cy + h / 2 - 0.012 * scale), 0.022 * scale, 0.012 * scale, fill=False, lw=2.0, ec=color))
    bx, by = cx + 0.02 * scale, cy - 0.002 * scale
    s = 0.026 * scale
    ax.plot([bx - s / 2, bx - s / 2, bx - s / 6], [by + s / 2, by + s / 6, by + s / 6], color=color, lw=1.8)
    ax.plot([bx + s / 2, bx + s / 2, bx + s / 6], [by + s / 2, by + s / 6, by + s / 6], color=color, lw=1.8)
    ax.plot([bx - s / 2, bx - s / 2, bx - s / 6], [by - s / 2, by - s / 6, by - s / 6], color=color, lw=1.8)
    ax.plot([bx + s / 2, bx + s / 2, bx + s / 6], [by - s / 2, by - s / 6, by - s / 6], color=color, lw=1.8)


def icon_waveform(ax: plt.Axes, cx: float, cy: float, scale: float, color: str) -> None:
    xs = np.linspace(-0.06, 0.06, 120) * scale + cx
    ys = cy + (0.010 * np.sin(38 * xs / max(scale, 1e-6)) + 0.020 * np.sin(14 * xs / max(scale, 1e-6))) * scale
    ax.plot(xs, ys, color=color, lw=2.2)
    ax.add_patch(Rectangle((cx - 0.065 * scale, cy - 0.045 * scale), 0.13 * scale, 0.09 * scale, fill=False, ec=color, lw=1.7))


def icon_radar(ax: plt.Axes, cx: float, cy: float, scale: float, color: str) -> None:
    for r in [0.025, 0.045, 0.065]:
        ax.add_patch(Arc((cx, cy), 2 * r * scale, 2 * r * scale, theta1=15, theta2=165, lw=1.8, ec=color))
    ax.plot([cx, cx + 0.055 * scale], [cy, cy + 0.035 * scale], color=color, lw=2.2)
    ax.add_patch(Circle((cx, cy), 0.0045 * scale, fc=color, ec="none"))


def icon_fusion(ax: plt.Axes, cx: float, cy: float, scale: float, color: str) -> None:
    pts = [(cx - 0.045 * scale, cy + 0.03 * scale), (cx - 0.045 * scale, cy), (cx - 0.045 * scale, cy - 0.03 * scale)]
    for px, py in pts:
        ax.add_patch(Circle((px, py), 0.009 * scale, fc="white", ec=color, lw=2.0))
        ax.plot([px + 0.009 * scale, cx + 0.015 * scale], [py, cy], color=color, lw=2.0)
    ax.add_patch(Circle((cx + 0.028 * scale, cy), 0.013 * scale, fc="white", ec=color, lw=2.0))


def icon_decision(ax: plt.Axes, cx: float, cy: float, scale: float, color: str) -> None:
    ax.add_patch(Circle((cx, cy), 0.038 * scale, fill=False, lw=2.0, ec=color))
    ax.plot([cx - 0.015 * scale, cx - 0.002 * scale, cx + 0.020 * scale], [cy - 0.002 * scale, cy - 0.018 * scale, cy + 0.020 * scale], color=color, lw=2.4)


def icon_folder(ax: plt.Axes, cx: float, cy: float, scale: float, color: str) -> None:
    ax.add_patch(Rectangle((cx - 0.05 * scale, cy - 0.03 * scale), 0.1 * scale, 0.06 * scale, fill=False, ec=color, lw=2.0))
    ax.add_patch(Rectangle((cx - 0.05 * scale, cy + 0.005 * scale), 0.035 * scale, 0.02 * scale, fill=False, ec=color, lw=2.0))


def icon_network(ax: plt.Axes, cx: float, cy: float, scale: float, color: str) -> None:
    layers = [[(cx - 0.04 * scale, cy + dy * scale) for dy in (-0.025, 0, 0.025)], [(cx, cy + dy * scale) for dy in (-0.035, -0.012, 0.012, 0.035)], [(cx + 0.04 * scale, cy)]]
    for a in layers[0]:
        for b in layers[1]:
            ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=1.2, alpha=0.8)
    for a in layers[1]:
        for b in layers[2]:
            ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=1.2, alpha=0.8)
    for layer in layers:
        for px, py in layer:
            ax.add_patch(Circle((px, py), 0.006 * scale, fc="white", ec=color, lw=1.6))


def icon_sliders(ax: plt.Axes, cx: float, cy: float, scale: float, color: str) -> None:
    for i, offset in enumerate([-0.022, 0.0, 0.022]):
        ax.plot([cx - 0.045 * scale, cx + 0.045 * scale], [cy + offset * scale, cy + offset * scale], color=color, lw=2.0)
        knob_x = cx + (-0.012 + 0.022 * i) * scale
        ax.add_patch(Circle((knob_x, cy + offset * scale), 0.008 * scale, fc="white", ec=color, lw=2.0))


def icon_compare(ax: plt.Axes, cx: float, cy: float, scale: float, color: str) -> None:
    heights = [0.025, 0.045, 0.065]
    offsets = [-0.025, 0.0, 0.025]
    for off, ht in zip(offsets, heights):
        ax.add_patch(Rectangle((cx + off * scale - 0.01 * scale, cy - 0.03 * scale), 0.02 * scale, ht * scale, fill=False, ec=color, lw=2.0))


def icon_checklist(ax: plt.Axes, cx: float, cy: float, scale: float, color: str) -> None:
    ax.add_patch(Rectangle((cx - 0.045 * scale, cy - 0.04 * scale), 0.09 * scale, 0.08 * scale, fill=False, ec=color, lw=2.0))
    for yoff in [0.018, -0.004, -0.026]:
        ax.plot([cx - 0.018 * scale, cx - 0.008 * scale, cx + 0.006 * scale], [cy + yoff * scale, cy + (yoff - 0.012) * scale, cy + (yoff + 0.010) * scale], color=color, lw=2.0)
        ax.plot([cx + 0.014 * scale, cx + 0.032 * scale], [cy + yoff * scale, cy + yoff * scale], color=color, lw=2.0)


def icon_target(ax: plt.Axes, cx: float, cy: float, scale: float, color: str) -> None:
    ax.add_patch(Circle((cx, cy), 0.04 * scale, fill=False, ec=color, lw=2.0))
    ax.add_patch(Circle((cx, cy), 0.022 * scale, fill=False, ec=color, lw=2.0))
    ax.plot([cx - 0.055 * scale, cx - 0.020 * scale], [cy, cy], color=color, lw=2.0)
    ax.plot([cx + 0.020 * scale, cx + 0.055 * scale], [cy, cy], color=color, lw=2.0)
    ax.plot([cx, cx], [cy - 0.055 * scale, cy - 0.020 * scale], color=color, lw=2.0)
    ax.plot([cx, cx], [cy + 0.020 * scale, cy + 0.055 * scale], color=color, lw=2.0)


def icon_shield(ax: plt.Axes, cx: float, cy: float, scale: float, color: str) -> None:
    verts_x = np.array([0.0, 0.045, 0.035, 0.0, -0.035, -0.045, 0.0]) * scale + cx
    verts_y = np.array([0.05, 0.02, -0.03, -0.06, -0.03, 0.02, 0.05]) * scale + cy
    ax.plot(verts_x, verts_y, color=color, lw=2.0)
    ax.plot([cx - 0.018 * scale, cx - 0.003 * scale, cx + 0.022 * scale], [cy - 0.005 * scale, cy - 0.02 * scale, cy + 0.02 * scale], color=color, lw=2.3)


def icon_filter(ax: plt.Axes, cx: float, cy: float, scale: float, color: str) -> None:
    ax.plot([cx - 0.05 * scale, cx + 0.05 * scale], [cy + 0.04 * scale, cy + 0.04 * scale], color=color, lw=2.0)
    ax.plot([cx - 0.05 * scale, cx - 0.012 * scale], [cy + 0.04 * scale, cy - 0.01 * scale], color=color, lw=2.0)
    ax.plot([cx + 0.05 * scale, cx + 0.012 * scale], [cy + 0.04 * scale, cy - 0.01 * scale], color=color, lw=2.0)
    ax.plot([cx - 0.012 * scale, cx - 0.012 * scale], [cy - 0.01 * scale, cy - 0.05 * scale], color=color, lw=2.0)
    ax.plot([cx + 0.012 * scale, cx + 0.012 * scale], [cy - 0.01 * scale, cy - 0.05 * scale], color=color, lw=2.0)


def annotate(ax: plt.Axes, text: str, xy: tuple[float, float], xytext: tuple[float, float], color: str, size: float = 12.5) -> None:
    ann = ax.annotate(
        text,
        xy=xy,
        xytext=xytext,
        textcoords="data",
        fontsize=size,
        color=TEXT,
        ha="left",
        va="center",
        bbox=dict(boxstyle="round,pad=0.28,rounding_size=0.02", fc="white", ec=color, lw=1.4),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4, shrinkA=6, shrinkB=3, mutation_scale=11),
    )
    ann.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])


def panel_shell(ax: plt.Axes, title: str, accent: str) -> None:
    round_box(ax, 0.02, 0.04, 0.96, 0.92, face=CARD_BG, edge=BORDER, lw=1.6, radius=0.03)
    ax.add_patch(Rectangle((0.02, 0.90), 0.96, 0.06, facecolor=accent, edgecolor="none"))
    ax.text(0.05, 0.93, title, ha="left", va="center", fontsize=18, color="white", weight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def load_inputs() -> dict[str, object]:
    r1 = pd.read_csv(REPO_ROOT / "final_isef_package" / "results_section" / "tables" / "R1_overall_test_performance.csv")
    stress_radar = pd.read_csv(REPO_ROOT / "final_isef_package" / "tables" / "stress_radar_clutter.csv")
    episodes = pd.read_csv(REPO_ROOT / "data" / "processed" / "episodes" / "episodes.csv", dtype={"episode_id": str})
    significance_text = (REPO_ROOT / "final_isef_package" / "metadata" / "significance_pr_auc.txt").read_text(encoding="utf-8")

    sig_match = re.search(
        r"Comparison: F-GATE - F-AVG\s+mean_diff=([-0-9.]+)\s+ci95=\(([-0-9.]+), ([-0-9.]+)\)",
        significance_text,
        flags=re.MULTILINE,
    )
    if not sig_match:
        raise RuntimeError("Could not parse F-GATE vs F-AVG bootstrap CI.")

    vision_drone = Image.open(SELECTED_INPUTS[0].source_path).convert("RGB")
    vision_bird = Image.open(SELECTED_INPUTS[1].source_path).convert("RGB")
    audio_array = np.load(SELECTED_INPUTS[2].source_path)
    radar_array = np.load(SELECTED_INPUTS[3].source_path)

    audio_ep = episodes.loc[episodes["episode_id"] == "003731"]
    radar_ep = episodes.loc[episodes["episode_id"] == "003737"]

    return {
        "performance": r1,
        "stress_radar": stress_radar,
        "sig_mean_diff": float(sig_match.group(1)),
        "sig_ci_low": float(sig_match.group(2)),
        "sig_ci_high": float(sig_match.group(3)),
        "vision_drone": vision_drone,
        "vision_bird": vision_bird,
        "audio_array": audio_array,
        "radar_array": radar_array,
        "audio_episode": audio_ep.iloc[0].to_dict(),
        "radar_episode": radar_ep.iloc[0].to_dict(),
    }


def draw_system_overview() -> None:
    fig, ax = new_slide()
    cards = [
        (0.07, 0.68, VISION, "Vision", icon_camera),
        (0.07, 0.42, AUDIO, "Acoustic", icon_waveform),
        (0.07, 0.16, RADAR, "Radar", icon_radar),
    ]
    for x, y, color, label, icon in cards:
        round_box(ax, x, y, 0.20, 0.17, face=BG_SOFT, edge=color, lw=2.0)
        ax.text(x + 0.10, y + 0.108, label, ha="center", va="center", fontsize=22, color=TEXT, weight="bold")
        icon(ax, x + 0.10, y + 0.055, 1.0, color)

    round_box(ax, 0.38, 0.28, 0.24, 0.34, face="#f4fbfb", edge=FUSION, lw=2.4)
    ax.add_patch(Rectangle((0.38, 0.54), 0.24, 0.08, facecolor=FUSION, edgecolor="none"))
    ax.text(0.50, 0.58, "Adaptive Fusion", ha="center", va="center", fontsize=26, color="white", weight="bold")
    ax.text(0.50, 0.42, "quality-aware decision layer", ha="center", va="center", fontsize=16, color=TEXT_MUTED)
    icon_fusion(ax, 0.50, 0.35, 1.3, FUSION)

    round_box(ax, 0.72, 0.35, 0.18, 0.20, face="#f8fbf8", edge=SUCCESS, lw=2.2)
    ax.text(0.81, 0.47, "UAV / Not UAV", ha="center", va="center", fontsize=24, color=TEXT, weight="bold")
    icon_decision(ax, 0.81, 0.40, 1.1, SUCCESS)

    for yc in [0.765, 0.505, 0.245]:
        arrow(ax, (0.27, yc), (0.38, 0.45), color=SLATE_DARK, lw=3.0)
    arrow(ax, (0.62, 0.45), (0.72, 0.45), color=SLATE_DARK, lw=3.2)
    ax.text(0.06, 0.92, "Condition-Adaptive Multimodal UAV Detection", fontsize=30, color=TEXT, weight="bold")
    save(fig, "01_system_overview")


def draw_vision_panel(ax: plt.Axes, drone_img: Image.Image, bird_img: Image.Image) -> None:
    panel_shell(ax, "Vision challenge", VISION)
    left = ax.inset_axes([0.07, 0.26, 0.39, 0.54])
    right = ax.inset_axes([0.54, 0.26, 0.39, 0.54])
    for iax, img in [(left, drone_img), (right, bird_img)]:
        iax.imshow(img)
        iax.axis("off")
    card_label(ax, 0.09, 0.81, "Drone", VISION)
    card_label(ax, 0.56, 0.81, "Bird", SLATE_DARK)
    annotate(ax, "distant object", (0.24, 0.52), (0.09, 0.16), VISION)
    annotate(ax, "low detail", (0.26, 0.54), (0.23, 0.11), VISION)
    annotate(ax, "similar distant silhouette", (0.70, 0.53), (0.54, 0.16), SLATE_DARK)
    annotate(ax, "false positive risk", (0.69, 0.53), (0.74, 0.11), RED)
    ax.text(
        0.05,
        0.075,
        "Distant silhouettes can blur the class boundary.",
        ha="left",
        va="center",
        fontsize=12.5,
        color=TEXT_MUTED,
    )


def draw_audio_panel(ax: plt.Axes, audio_array: np.ndarray, audio_meta: dict[str, object]) -> None:
    panel_shell(ax, "Audio challenge", AUDIO)
    spec_ax = ax.inset_axes([0.08, 0.24, 0.74, 0.60])
    im = spec_ax.imshow(audio_array, origin="lower", aspect="auto", cmap="magma")
    spec_ax.set_xlabel("Time", labelpad=0)
    spec_ax.set_ylabel("Frequency")
    spec_ax.set_xticks([0, 25, 50, 75, 100], labels=["0.0", "0.5", "1.0", "1.5", "2.0"])
    spec_ax.set_yticks([0, 32, 64, 96, 127], labels=["low", "", "", "", "high"])
    cax = ax.inset_axes([0.85, 0.28, 0.03, 0.48])
    cb = plt.colorbar(im, cax=cax)
    cb.set_label("Intensity", fontsize=12)
    annotate(ax, "target energy", (0.44, 0.68), (0.08, 0.11), AUDIO, size=11.0)
    annotate(ax, "background noise", (0.21, 0.32), (0.30, 0.11), WARNING, size=11.0)
    annotate(ax, "signal quality drops\nwith distance/noise", (0.55, 0.46), (0.56, 0.11), RED, size=10.6)
    note = f"ep 003731 | Drone | SNR {float(audio_meta['audio_snr_db']):.1f} dB"
    ax.text(0.08, 0.055, note, ha="left", va="center", fontsize=11.5, color=TEXT_MUTED)


def draw_radar_panel(ax: plt.Axes, radar_array: np.ndarray, radar_meta: dict[str, object]) -> None:
    panel_shell(ax, "Radar challenge", RADAR)
    rd_ax = ax.inset_axes([0.08, 0.24, 0.74, 0.60])
    vmin = float(np.percentile(radar_array, 2))
    vmax = float(np.percentile(radar_array, 99))
    im = rd_ax.imshow(radar_array, origin="lower", aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    rd_ax.set_xlabel("Doppler", labelpad=0)
    rd_ax.set_ylabel("Range")
    rd_ax.set_xticks([0, 32, 64, 96, 127], labels=["neg.", "", "0", "", "pos."])
    rd_ax.set_yticks([0, 32, 64, 96, 127], labels=["near", "", "", "", "far"])
    cax = ax.inset_axes([0.85, 0.28, 0.03, 0.48])
    cb = plt.colorbar(im, cax=cax)
    cb.set_label("Intensity", fontsize=12)
    annotate(ax, "target return", (0.45, 0.46), (0.11, 0.11), RADAR, size=11.0)
    annotate(ax, "clutter / interference", (0.35, 0.34), (0.58, 0.11), RED, size=10.6)
    ax.text(
        0.08,
        0.055,
        f"ep 003737 | Drone | clutter {float(radar_meta['radar_clutter_level']):.2f}",
        ha="left",
        va="center",
        fontsize=11.5,
        color=TEXT_MUTED,
    )


def draw_modality_challenges(data: dict[str, object]) -> None:
    fig, ax = new_slide()
    draw_vision_panel(ax, data["vision_drone"], data["vision_bird"])
    save(fig, "02a_vision_challenge")

    fig, ax = new_slide()
    draw_audio_panel(ax, data["audio_array"], data["audio_episode"])
    save(fig, "02b_audio_challenge")

    fig, ax = new_slide()
    draw_radar_panel(ax, data["radar_array"], data["radar_episode"])
    save(fig, "02c_radar_challenge")

    fig = plt.figure(figsize=FIGSIZE, dpi=DPI, facecolor="white")
    axes = [
        fig.add_axes([0.02, 0.14, 0.30, 0.74]),
        fig.add_axes([0.35, 0.14, 0.30, 0.74]),
        fig.add_axes([0.68, 0.14, 0.30, 0.74]),
    ]
    for axis in axes:
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.axis("off")
    draw_vision_panel(axes[0], data["vision_drone"], data["vision_bird"])
    draw_audio_panel(axes[1], data["audio_array"], data["audio_episode"])
    draw_radar_panel(axes[2], data["radar_array"], data["radar_episode"])
    fig.savefig(PNG_DIR / "02_modality_challenges_combined.png", dpi=DPI, facecolor="white")
    fig.savefig(SVG_DIR / "02_modality_challenges_combined.svg", facecolor="white")
    plt.close(fig)


def draw_procedure_pipeline() -> None:
    fig, ax = new_slide()
    ax.text(0.06, 0.90, "Procedure pipeline", fontsize=30, color=TEXT, weight="bold")
    ax.text(
        0.06,
        0.84,
        "General workflow used to develop and evaluate the multimodal detector",
        fontsize=16,
        color=TEXT_MUTED,
    )
    xs = np.linspace(0.06, 0.80, 5)
    labels = [
        "Prepare data",
        "Train modality models",
        "Extract confidence\nand quality signals",
        "Compare fusion\nmethods",
        "Evaluate under normal\nand stressed\nconditions",
    ]
    icons = [icon_folder, icon_network, icon_sliders, icon_compare, icon_checklist]
    accents = [SLATE_DARK, VISION, AUDIO, FUSION, SUCCESS]
    for i, (x, label, icon, accent) in enumerate(zip(xs, labels, icons, accents), start=1):
        round_box(ax, x, 0.34, 0.16, 0.32, face=BG_SOFT, edge=accent, lw=2.0)
        ax.add_patch(Circle((x + 0.03, 0.61), 0.022, fc=accent, ec="none"))
        ax.text(x + 0.03, 0.61, str(i), ha="center", va="center", fontsize=13, color="white", weight="bold")
        icon(ax, x + 0.08, 0.53, 1.0, accent)
        ax.text(x + 0.08, 0.41, label, ha="center", va="center", fontsize=15, color=TEXT, weight="bold")
    for i in range(4):
        arrow(ax, (xs[i] + 0.16, 0.50), (xs[i + 1], 0.50), color=SLATE_DARK, lw=2.8)
    save(fig, "03_procedure_pipeline")


def image_thumbnail(ax: plt.Axes, bbox: list[float], image: Image.Image | np.ndarray, cmap: str | None = None, vmin: float | None = None, vmax: float | None = None) -> None:
    iax = ax.inset_axes(bbox)
    if isinstance(image, np.ndarray):
        iax.imshow(image, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        iax.imshow(image)
    iax.axis("off")


def draw_system_architecture(data: dict[str, object]) -> None:
    fig, ax = new_slide()
    ax.text(0.06, 0.92, "System architecture", fontsize=30, color=TEXT, weight="bold")
    row_y = [0.72, 0.48, 0.24]
    input_cards = [
        ("Vision input", VISION, data["vision_drone"], icon_camera),
        ("Audio input", AUDIO, data["audio_array"], icon_waveform),
        ("Radar input", RADAR, data["radar_array"], icon_radar),
    ]
    model_titles = ["YOLO-based detector", "CNN on spectrogram\nfeatures", "CNN on range-Doppler\nfeatures"]
    quality_titles = ["image quality /\nconfidence", "acoustic quality", "clutter-related\nquality"]
    for y, (label, color, thumb, icon), model_title, quality_title in zip(row_y, input_cards, model_titles, quality_titles):
        header_card(ax, 0.04, y, 0.19, 0.14, title=label, accent=color)
        image_thumbnail(
            ax,
            [0.056, y + 0.020, 0.055, 0.08],
            thumb,
            cmap="magma" if isinstance(thumb, np.ndarray) and label.startswith("Audio") else ("viridis" if isinstance(thumb, np.ndarray) else None),
            vmin=float(np.percentile(thumb, 2)) if isinstance(thumb, np.ndarray) and label.startswith("Radar") else None,
            vmax=float(np.percentile(thumb, 99)) if isinstance(thumb, np.ndarray) and label.startswith("Radar") else None,
        )
        icon(ax, 0.145, y + 0.055, 0.85, color)
        header_card(ax, 0.28, y, 0.25, 0.14, title=model_title, accent=color, title_size=16)
        body_map = {
            "YOLO-based detector": ["Episode score from detector confidence", "Binary presence signal + features"],
            "CNN on spectrogram\nfeatures": ["Log-mel input [1 x 128 x 101]", "Episode logit + audio quality"],
            "CNN on range-Doppler\nfeatures": ["2D RD map input [1 x 128 x 128]", "Episode logit + radar quality"],
        }
        for idx, line in enumerate(body_map[model_title]):
            ax.text(0.295, y + 0.075 - idx * 0.040, line, fontsize=12.0, color=TEXT)
        round_box(ax, 0.56, y + 0.035, 0.12, 0.06, face=BG_SOFT, edge=color, lw=1.6, radius=0.02)
        ax.text(0.62, y + 0.065, quality_title, ha="center", va="center", fontsize=11.5, color=color, weight="bold")
        arrow(ax, (0.23, y + 0.07), (0.28, y + 0.07), color=color, lw=2.4)
        arrow(ax, (0.53, y + 0.07), (0.56, y + 0.07), color=color, lw=2.4)
        arrow(ax, (0.68, y + 0.07), (0.69, 0.50), color=SLATE_DARK, lw=2.2)

    header_card(ax, 0.68, 0.29, 0.18, 0.42, title="Fusion comparison", accent=FUSION)
    methods = [("Average", SLATE), ("Static weighting", SLATE_DARK), ("Adaptive gating", FUSION)]
    for i, (name, color) in enumerate(methods):
        yy = 0.58 - i * 0.105
        round_box(ax, 0.700, yy, 0.14, 0.072, face=BG_SOFT, edge=color, lw=1.6, radius=0.02)
        ax.text(0.770, yy + 0.036, name, ha="center", va="center", fontsize=15, color=TEXT, weight="bold")
    ax.text(0.700, 0.315, "Adaptive fusion reduces the\ninfluence of degraded inputs", fontsize=11.4, color=TEXT_MUTED, ha="left")
    arrow(ax, (0.86, 0.50), (0.87, 0.50), color=FUSION, lw=3.0)
    round_box(ax, 0.87, 0.39, 0.11, 0.18, face="#f8fbf8", edge=SUCCESS, lw=1.8, radius=0.02)
    ax.text(0.925, 0.515, "Final detection\nscore", ha="center", va="center", fontsize=16, color=SUCCESS, weight="bold")
    ax.text(0.925, 0.435, "UAV / Not UAV", ha="center", va="center", fontsize=11.0, color=TEXT, weight="bold")
    save(fig, "04_system_architecture")


def draw_prauc_results(data: dict[str, object]) -> None:
    perf = data["performance"].set_index("Model").loc[["UNI-A", "F-AVG", "F-STATIC", "F-GATE", "F-GATE-NOQ"]].reset_index()
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI, facecolor="white")
    ax = fig.add_axes([0.07, 0.14, 0.60, 0.72])
    note_ax = fig.add_axes([0.72, 0.16, 0.23, 0.68])
    note_ax.axis("off")
    ax.set_title("Episode-level TEST PR-AUC by method", loc="left", pad=16, color=TEXT, fontweight="bold")
    colors = ["#b5c0ca", "#9aa6b2", "#5c7285", FUSION, HILITE]
    bars = ax.bar(perf["Model"], perf["PR-AUC"], color=colors, edgecolor="#4a5b69", linewidth=1.0, width=0.66)
    ax.set_ylabel("PR-AUC")
    ax.set_ylim(0.68, 0.92)
    ax.grid(axis="y", color=GRID, lw=1.0)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, perf["PR-AUC"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.004, f"{value:.3f}", ha="center", va="bottom", fontsize=13, color=TEXT, weight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BORDER)
    ax.spines["bottom"].set_color(BORDER)

    round_box(note_ax, 0.02, 0.72, 0.96, 0.24, face="#f4fbfb", edge=HILITE, lw=1.8)
    note_ax.text(0.07, 0.87, "Top overall TEST PR-AUC", fontsize=16, color=TEXT, weight="bold")
    note_ax.text(0.07, 0.80, "F-GATE-NOQ", fontsize=21, color=HILITE, weight="bold")
    note_ax.text(0.07, 0.735, f"{float(perf.loc[perf['Model']=='F-GATE-NOQ', 'PR-AUC'].iloc[0]):.3f}", fontsize=28, color=TEXT, weight="bold")

    round_box(note_ax, 0.02, 0.42, 0.96, 0.24, face="#f9fbfd", edge=FUSION, lw=1.8)
    note_ax.text(0.07, 0.58, "Quality-aware F-GATE vs F-AVG", fontsize=15, color=TEXT, weight="bold")
    actual_delta = float(perf.loc[perf["Model"] == "F-GATE", "PR-AUC"].iloc[0] - perf.loc[perf["Model"] == "F-AVG", "PR-AUC"].iloc[0])
    note_ax.text(0.07, 0.51, f"Delta PR-AUC: +{actual_delta:.3f}", fontsize=18, color=FUSION, weight="bold")
    note_ax.text(0.07, 0.455, f"95% CI: [{data['sig_ci_low']:.3f}, {data['sig_ci_high']:.3f}]", fontsize=15, color=TEXT_MUTED)

    round_box(note_ax, 0.02, 0.14, 0.96, 0.20, face="#fbfbfc", edge=SLATE_DARK, lw=1.6)
    note_ax.text(0.07, 0.28, "Best unimodal baseline", fontsize=15, color=TEXT, weight="bold")
    note_ax.text(0.07, 0.21, f"UNI-A: {float(perf.loc[perf['Model']=='UNI-A', 'PR-AUC'].iloc[0]):.3f}", fontsize=18, color=SLATE_DARK, weight="bold")
    save(fig, "05_prauc_results")


def draw_false_alarm_and_stress(data: dict[str, object]) -> None:
    perf = data["performance"].set_index("Model").loc[["UNI-A", "F-AVG", "F-STATIC", "F-GATE", "F-GATE-NOQ"]].reset_index()
    stress = data["stress_radar"]
    radar_fgate = stress[stress["run_id"] == "F-GATE"].sort_values("clutter_level").reset_index(drop=True)
    before = radar_fgate.iloc[0]
    after = radar_fgate.iloc[-1]

    fig = plt.figure(figsize=FIGSIZE, dpi=DPI, facecolor="white")
    ax = fig.add_axes([0.07, 0.14, 0.60, 0.72])
    inset_ax = fig.add_axes([0.72, 0.16, 0.23, 0.68])
    inset_ax.axis("off")

    ax.set_title("Episode-level TEST false alarm rate at threshold 0.5", loc="left", pad=16, color=TEXT, fontweight="bold")
    colors = ["#b5c0ca", "#9aa6b2", SUCCESS, FUSION, HILITE]
    far_pct = perf["FAR"] * 100.0
    bars = ax.bar(perf["Model"], far_pct, color=colors, edgecolor="#4a5b69", linewidth=1.0, width=0.66)
    ax.set_ylabel("False alarm rate (%)")
    ax.set_ylim(0, max(float(far_pct.max()) + 8, 45))
    ax.grid(axis="y", color=GRID, lw=1.0)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, far_pct):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.2, f"{value:.1f}", ha="center", va="bottom", fontsize=13, color=TEXT, weight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BORDER)
    ax.spines["bottom"].set_color(BORDER)

    round_box(inset_ax, 0.02, 0.52, 0.96, 0.43, face="#f9fbfd", edge=RADAR, lw=1.8)
    inset_ax.text(0.07, 0.88, "Stress Test: Radar Clutter (F-GATE)", fontsize=15, color=TEXT, weight="bold")
    image_thumbnail(
        inset_ax,
        [0.08, 0.60, 0.34, 0.20],
        data["radar_array"],
        cmap="viridis",
        vmin=float(np.percentile(data["radar_array"], 2)),
        vmax=float(np.percentile(data["radar_array"], 99)),
    )
    bar_ax = inset_ax.inset_axes([0.48, 0.58, 0.42, 0.22])
    vals = [float(before["pr_auc"]), float(after["pr_auc"])]
    xpos = np.array([0.0, 1.0], dtype=float)
    bar_ax.bar(xpos, vals, color=[FUSION, "#8dbcc3"], edgecolor="#4a5b69", linewidth=0.8, width=0.55)
    bar_ax.set_ylim(0.86, 0.90)
    bar_ax.set_ylabel("PR-AUC", fontsize=10)
    bar_ax.set_xlabel("Clutter level", fontsize=10)
    bar_ax.set_xticks(xpos, labels=["0.1", "0.9"])
    bar_ax.tick_params(labelsize=10)
    bar_ax.grid(axis="y", color=GRID, lw=0.8)
    bar_ax.set_axisbelow(True)
    for x, v in zip(xpos, vals):
        bar_ax.text(x, v + 0.001, f"{v:.3f}", ha="center", va="bottom", fontsize=10, weight="bold", color=TEXT)
    inset_ax.text(0.08, 0.54, f"Before: {float(before['pr_auc']):.3f}  at clutter {float(before['clutter_level']):.1f}", fontsize=12.5, color=TEXT)
    inset_ax.text(0.08, 0.49, f"After:  {float(after['pr_auc']):.3f}  at clutter {float(after['clutter_level']):.1f}", fontsize=12.5, color=TEXT)
    inset_ax.text(0.08, 0.43, f"Delta: {float(after['pr_auc'] - before['pr_auc']):+.3f} PR-AUC", fontsize=13.5, color=RADAR, weight="bold")
    inset_ax.text(0.08, 0.37, "Modest degradation under clutter\nwhile the radar return remains visible.", fontsize=11.2, color=TEXT_MUTED)

    round_box(inset_ax, 0.02, 0.12, 0.96, 0.18, face="#f8fbf8", edge=SUCCESS, lw=1.4)
    inset_ax.text(0.07, 0.22, "Lowest FAR in the compared set", fontsize=14, color=TEXT, weight="bold")
    inset_ax.text(0.07, 0.155, f"F-STATIC: {float(perf.loc[perf['Model']=='F-STATIC', 'FAR'].iloc[0] * 100):.1f}%", fontsize=19, color=SUCCESS, weight="bold")
    save(fig, "06_false_alarm_and_stress")


def summary_card(ax: plt.Axes, x: float, title: str, subtitle: str, accent: str, icon_fn) -> None:
    round_box(ax, x, 0.24, 0.26, 0.52, face=BG_SOFT, edge=accent, lw=2.0)
    ax.add_patch(Circle((x + 0.13, 0.63), 0.055, fc="white", ec=accent, lw=2.0))
    icon_fn(ax, x + 0.13, 0.63, 0.9, accent)
    ax.text(x + 0.13, 0.47, title, ha="center", va="center", fontsize=24, color=TEXT, weight="bold")
    ax.text(x + 0.13, 0.36, textwrap.fill(subtitle, width=28), ha="center", va="center", fontsize=15, color=TEXT_MUTED)


def draw_conclusion_summary() -> None:
    fig, ax = new_slide()
    ax.text(0.06, 0.90, "Conclusion summary", fontsize=30, color=TEXT, weight="bold")
    summary_card(ax, 0.06, "Stronger detection", "Fusion outperformed single-modality methods", VISION, icon_target)
    summary_card(ax, 0.37, "Better robustness", "Adaptive fusion handled degraded inputs more effectively", FUSION, icon_shield)
    summary_card(ax, 0.68, "Greater practicality", "Lower false alarms improved real-world usefulness", SUCCESS, icon_filter)
    save(fig, "07_conclusion_summary_visual")


def milestone(ax: plt.Axes, x: float, y: float, num: int, label: str, accent: str) -> None:
    ax.add_patch(Circle((x, y), 0.035, fc=accent, ec="white", lw=2.0))
    ax.text(x, y, str(num), ha="center", va="center", fontsize=14, color="white", weight="bold")
    round_box(ax, x - 0.11, y + 0.05, 0.22, 0.14, face=BG_SOFT, edge=accent, lw=1.8, radius=0.02)
    ax.text(x, y + 0.12, textwrap.fill(label, width=24), ha="center", va="center", fontsize=15, color=TEXT, weight="bold")


def draw_future_work() -> None:
    fig, ax = new_slide()
    ax.text(0.06, 0.90, "Future work roadmap", fontsize=30, color=TEXT, weight="bold")
    ax.text(0.06, 0.84, "Key next steps for moving from experimental results toward field-ready deployment", fontsize=16, color=TEXT_MUTED)
    pts = [(0.14, 0.34), (0.36, 0.46), (0.60, 0.34), (0.84, 0.46)]
    labels = [
        "Larger real-world synchronized datasets",
        "Faster real-time deployment",
        "More stress testing and missing-modality scenarios",
        "Live sensor integration",
    ]
    colors = [VISION, AUDIO, RADAR, FUSION]
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=SLATE_DARK, lw=3.0)
    for i, ((x, y), label, color) in enumerate(zip(pts, labels, colors), start=1):
        milestone(ax, x, y, i, label, color)
    save(fig, "08_future_work_roadmap")


def render_preview_png(array: np.ndarray, out_path: Path, *, cmap: str, title: str) -> None:
    fig = plt.figure(figsize=(8, 5), dpi=200, facecolor="white")
    ax = fig.add_axes([0.12, 0.14, 0.73, 0.75])
    if cmap == "viridis":
        vmin = float(np.percentile(array, 2))
        vmax = float(np.percentile(array, 99))
        im = ax.imshow(array, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        im = ax.imshow(array, origin="lower", aspect="auto", cmap=cmap)
    ax.set_title(title, fontsize=16, color=TEXT)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)


def copy_selected_inputs(data: dict[str, object]) -> None:
    for item in SELECTED_INPUTS:
        dest = SELECTED_DIR / item.copy_name
        shutil.copy2(item.source_path, dest)
        if item.preview_name:
            if item.key == "audio_challenge":
                render_preview_png(data["audio_array"], SELECTED_DIR / item.preview_name, cmap="magma", title="Audio preview")
            elif item.key == "radar_challenge":
                render_preview_png(data["radar_array"], SELECTED_DIR / item.preview_name, cmap="viridis", title="Radar preview")


def build_contact_sheet() -> None:
    thumbs: list[tuple[str, Image.Image]] = []
    for stem in FINAL_ORDER:
        png_path = PNG_DIR / f"{stem}.png"
        img = Image.open(png_path).convert("RGB")
        thumbs.append((stem, img))

    cols = 3
    rows = math.ceil(len(thumbs) / cols)
    tile_w, tile_h = 720, 430
    margin = 40
    label_h = 46
    canvas = Image.new("RGB", (cols * tile_w + (cols + 1) * margin, rows * (tile_h + label_h) + (rows + 1) * margin), "white")

    for idx, (name, img) in enumerate(thumbs):
        r = idx // cols
        c = idx % cols
        x = margin + c * tile_w
        y = margin + r * (tile_h + label_h)
        thumb = ImageOps.contain(img, (tile_w - 30, tile_h - 30))
        thumb_bg = Image.new("RGB", (tile_w - 24, tile_h - 24), "white")
        tx = (thumb_bg.width - thumb.width) // 2
        ty = (thumb_bg.height - thumb.height) // 2
        thumb_bg.paste(thumb, (tx, ty))
        canvas.paste(thumb_bg, (x + 10, y + 10))
        draw = ImageDraw.Draw(canvas)
        draw.rounded_rectangle([x + 6, y + 6, x + tile_w - 6, y + tile_h - 6], radius=18, outline="#c5d1da", width=2)
        draw.text((x + 16, y + tile_h + 8), name.replace("_", " "), fill=TEXT, font=None)
    canvas.save(ASSET_ROOT / "contact_sheet.png")


def manifest_data(data: dict[str, object]) -> dict[str, object]:
    return {
        "asset_root": str(ASSET_ROOT.relative_to(REPO_ROOT)).replace("\\", "/"),
        "selected_inputs": {
            item.copy_name: {
                "source_path": str(item.source_path.relative_to(REPO_ROOT)).replace("\\", "/"),
                "note": item.note,
            }
            for item in SELECTED_INPUTS
        },
        "figures": {
            "01_system_overview": {
                "type": "author_created_diagram",
                "sources": ["final_isef_package/README.md", "final_isef_package/methodology_materials_updated.md"],
                "scientific_point": "High-level multimodal system overview from vision, acoustic, and radar inputs to adaptive fusion and final UAV / Not UAV decision.",
            },
            "02a_vision_challenge": {
                "type": "project_derived_panel",
                "sources": [
                    "final_isef_package/verification/IR_DRONE_059_f000112_DRONE.png",
                    "final_isef_package/verification/IR_BIRD_010_f000043_BIRD.png",
                ],
                "scientific_point": "Distant IR drone and bird silhouettes can create a difficult class boundary for vision-only detection.",
            },
            "02b_audio_challenge": {
                "type": "project_derived_panel",
                "sources": ["data/processed/audio_windows/DRONE_020_00000.npy", "data/processed/episodes/episodes.csv (episode_id=003731)"],
                "scientific_point": "Real low-SNR drone log-mel window showing target harmonics embedded in noise.",
            },
            "02c_radar_challenge": {
                "type": "project_derived_panel",
                "sources": ["data/processed/radar_rd_real/rd_00047495.npy", "data/processed/episodes/episodes.csv (episode_id=003737)"],
                "scientific_point": "Real cluttered range-Doppler map showing a visible target return plus interference structure.",
            },
            "02_modality_challenges_combined": {
                "type": "project_derived_composite",
                "sources": ["02a_vision_challenge", "02b_audio_challenge", "02c_radar_challenge"],
                "scientific_point": "Why fusion matters: each single modality has credible failure modes under degraded sensing conditions.",
            },
            "03_procedure_pipeline": {
                "type": "author_created_diagram",
                "sources": ["final_isef_package/methodology_materials_updated.md"],
                "scientific_point": "Compact high-level workflow used to build and evaluate the multimodal detector.",
            },
            "04_system_architecture": {
                "type": "mixed_diagram_with_project_thumbnails",
                "sources": [
                    "final_isef_package/methodology_materials_updated.md",
                    "final_isef_package/methodology_assets/tables/method_table_M2.csv",
                    "final_isef_package/methodology_assets/tables/method_table_M3.csv",
                    "selected_project_inputs/*",
                ],
                "scientific_point": "Branch-level architecture, quality signals, and fusion comparison methods used in the project.",
            },
            "05_prauc_results": {
                "type": "project_derived_chart",
                "sources": ["final_isef_package/results_section/tables/R1_overall_test_performance.csv", "final_isef_package/metadata/significance_pr_auc.txt"],
                "scientific_point": "Exact episode-level TEST PR-AUC comparison across the best unimodal and fusion methods.",
            },
            "06_false_alarm_and_stress": {
                "type": "project_derived_chart_with_inset",
                "sources": [
                    "final_isef_package/results_section/tables/R1_overall_test_performance.csv",
                    "final_isef_package/tables/stress_radar_clutter.csv",
                    "data/processed/radar_rd_real/rd_00047495.npy",
                ],
                "scientific_point": "Operational false alarm comparison plus exact F-GATE radar-clutter stress behavior.",
            },
            "07_conclusion_summary_visual": {
                "type": "author_created_diagram",
                "sources": ["final_isef_package/results_section/tables/R1_overall_test_performance.csv", "final_isef_package/tables/stress_radar_clutter.csv"],
                "scientific_point": "Conclusion synthesis: stronger detection, better robustness, and greater practicality.",
            },
            "08_future_work_roadmap": {
                "type": "author_created_diagram",
                "sources": ["final_isef_package/methodology_materials_updated.md", "final_isef_package/README.md"],
                "scientific_point": "Four concrete next-step milestones for extending the project toward deployment.",
            },
        },
    }


def main() -> None:
    ensure_dirs()
    data = load_inputs()
    copy_selected_inputs(data)
    draw_system_overview()
    draw_modality_challenges(data)
    draw_procedure_pipeline()
    draw_system_architecture(data)
    draw_prauc_results(data)
    draw_false_alarm_and_stress(data)
    draw_conclusion_summary()
    draw_future_work()
    build_contact_sheet()
    manifest = manifest_data(data)
    (SUPPORT_DIR / "asset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
