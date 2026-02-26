from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


BG = "#F5F7FA"
WHITE = "#FFFFFF"
BORDER = "#C7D0DB"
TEXT_DARK = "#17232E"
TEXT_MUTED = "#5A6B7D"
HEADER = "#274C77"
ARROW = "#3C4A5A"


def choose_font() -> str:
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Roboto", "Montserrat", "DejaVu Sans"):
        if name in available:
            return name
    return "DejaVu Sans"


def draw_database_icon(ax: plt.Axes, x: float, y: float, s: float) -> None:
    r = 0.20 * s
    cx = x + 0.5 * s
    for i in range(3):
        yy = y + s - (0.25 * s) - i * (0.24 * s)
        ax.add_patch(Circle((cx, yy), r, edgecolor=HEADER, facecolor="none", linewidth=1.2))
        ax.plot([cx - r, cx + r], [yy, yy], color=HEADER, linewidth=1.2)


def draw_slider_icon(ax: plt.Axes, x: float, y: float, s: float) -> None:
    ys = [y + 0.75 * s, y + 0.52 * s, y + 0.29 * s]
    knobs = [0.30, 0.64, 0.46]
    for yy, k in zip(ys, knobs):
        ax.plot([x + 0.12 * s, x + 0.88 * s], [yy, yy], color=HEADER, linewidth=1.2)
        ax.add_patch(Circle((x + k * s, yy), 0.07 * s, edgecolor=HEADER, facecolor=WHITE, linewidth=1.2))


def draw_stack_icon(ax: plt.Axes, x: float, y: float, s: float) -> None:
    w = 0.62 * s
    h = 0.18 * s
    x0 = x + 0.19 * s
    y0 = y + 0.18 * s
    for i in range(3):
        ax.add_patch(
            Rectangle(
                (x0 + 0.04 * i * s, y0 + 0.21 * i * s),
                w,
                h,
                edgecolor=HEADER,
                facecolor="none",
                linewidth=1.2,
            )
        )


def draw_chart_ci_icon(ax: plt.Axes, x: float, y: float, s: float) -> None:
    ax.plot([x + 0.14 * s, x + 0.14 * s], [y + 0.20 * s, y + 0.82 * s], color=HEADER, linewidth=1.2)
    ax.plot([x + 0.14 * s, x + 0.88 * s], [y + 0.20 * s, y + 0.20 * s], color=HEADER, linewidth=1.2)
    xs = [x + 0.20 * s, x + 0.44 * s, x + 0.62 * s, x + 0.82 * s]
    ys = [y + 0.32 * s, y + 0.46 * s, y + 0.40 * s, y + 0.68 * s]
    ax.plot(xs, ys, color=HEADER, linewidth=1.5)
    ax.plot([x + 0.70 * s, x + 0.70 * s], [y + 0.52 * s, y + 0.78 * s], color=TEXT_MUTED, linewidth=1.0)
    ax.plot([x + 0.74 * s, x + 0.74 * s], [y + 0.52 * s, y + 0.78 * s], color=TEXT_MUTED, linewidth=1.0)
    ax.plot([x + 0.70 * s, x + 0.74 * s], [y + 0.78 * s, y + 0.78 * s], color=TEXT_MUTED, linewidth=1.0)
    ax.plot([x + 0.70 * s, x + 0.74 * s], [y + 0.52 * s, y + 0.52 * s], color=TEXT_MUTED, linewidth=1.0)


def draw_stage_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    header_text: str,
    body_lines: list[str],
    icon: str | None,
) -> None:
    # Main rounded box.
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.006,rounding_size=0.02",
        facecolor=WHITE,
        edgecolor=BORDER,
        linewidth=1.25,
        transform=ax.transAxes,
    )
    ax.add_patch(box)

    # Header bar clipped to rounded box boundary.
    hh = 0.20 * h
    header_rect = Rectangle(
        (x, y + h - hh),
        w,
        hh,
        transform=ax.transAxes,
        facecolor=HEADER,
        edgecolor="none",
    )
    header_rect.set_clip_path(box)
    ax.add_patch(header_rect)

    t_header = ax.text(
        x + (0.03 * w),
        y + h - (0.50 * hh),
        header_text,
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=13,
        fontweight="bold",
        color=WHITE,
    )
    t_header.set_clip_path(box)

    text_x = x + (0.055 * w)
    text_y0 = y + h - hh - (0.11 * h)
    line_step = 0.155 * h
    for i, line in enumerate(body_lines):
        fs = 10.9
        if len(line) > 34:
            fs = 10.3
        if len(line) > 40:
            fs = 9.8
        t = ax.text(
            text_x,
            text_y0 - (i * line_step),
            line,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=fs,
            color=TEXT_DARK,
            clip_on=True,
        )
        t.set_clip_path(box)

    if icon:
        icon_size = 0.23 * h
        # Reserve icon in lower-right body area to avoid text overlap.
        ix = x + w - icon_size - (0.04 * w)
        iy = y + (0.08 * h)
        if icon == "database":
            draw_database_icon(ax, ix, iy, icon_size)
        elif icon == "slider":
            draw_slider_icon(ax, ix, iy, icon_size)
        elif icon == "stack":
            draw_stack_icon(ax, ix, iy, icon_size)
        elif icon == "chart_ci":
            draw_chart_ci_icon(ax, ix, iy, icon_size)


def build_figure(out_png: Path, transparent: bool = False) -> Path:
    font = choose_font()
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [font, "DejaVu Sans"]

    fig = plt.figure(figsize=(12, 4), dpi=200, facecolor=("none" if transparent else BG))
    ax = fig.add_axes([0, 0, 1, 1], facecolor=("none" if transparent else BG))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Title
    ax.text(
        0.5,
        0.925,
        "Evaluation Setup: Protocol + Comparisons",
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
        color=TEXT_DARK,
        transform=ax.transAxes,
    )

    margin = 0.045
    gap = 0.018
    y = 0.26
    h = 0.54
    w = (1.0 - (2 * margin) - (3 * gap)) / 4.0

    boxes = [
        {
            "header": "Held-Out Test Set",
            "lines": [
                "N = 400 episodes (80/10/10 split)",
                "Episode-level (2.0 s)",
                "Leakage checks: PASS",
            ],
            "icon": "database",
        },
        {
            "header": "Evaluation Scenarios",
            "lines": [
                "All-present (no masking)",
                "Stress sweeps (single-factor)",
                "Missing-modality masks",
            ],
            "icon": "slider",
        },
        {
            "header": "Models Compared",
            "lines": [
                "Unimodal: UNI-V, UNI-A, UNI-R",
                "Non-adaptive: F-AVG, F-STATIC",
                "Adaptive: F-GATE, F-GATE-NOQ",
            ],
            "icon": "stack",
        },
        {
            "header": "Metrics + Reliability",
            "lines": [
                "Primary: PR-AUC",
                "Secondary: ROC-AUC, confusion, FAR",
                "Bootstrap CI for dPR-AUC",
            ],
            "icon": "chart_ci",
        },
    ]

    xs: list[float] = []
    for i, cfg in enumerate(boxes):
        x = margin + i * (w + gap)
        xs.append(x)
        draw_stage_box(
            ax=ax,
            x=x,
            y=y,
            w=w,
            h=h,
            header_text=cfg["header"],
            body_lines=cfg["lines"],
            icon=cfg["icon"],
        )

    # Arrows between boxes.
    ym = y + (0.34 * h)
    for i in range(3):
        x0 = xs[i] + w + 0.004
        x1 = xs[i + 1] - 0.004
        arr = FancyArrowPatch(
            (x0, ym),
            (x1, ym),
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=1.5,
            color=ARROW,
        )
        ax.add_patch(arr)

    # Footer note.
    footer = "Same held-out test episodes; perturbations/masks applied at evaluation time. Fixed decision threshold = 0.5 (no threshold tuning)."
    ax.text(
        0.5,
        0.17,
        footer,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10.2,
        color=TEXT_DARK,
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_png,
        dpi=200,
        transparent=transparent,
        bbox_inches="tight",
        pad_inches=0.08,
    )
    plt.close(fig)
    return out_png


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("final_isef_package/figures/evaluation_setup_protocol_comparisons.png"),
    )
    parser.add_argument("--transparent", action="store_true")
    args = parser.parse_args()

    out = build_figure(out_png=args.out, transparent=bool(args.transparent))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
