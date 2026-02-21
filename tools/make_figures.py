from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import auc, precision_recall_curve, roc_curve


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PKG = ROOT / "final_isef_package"
FIG_DIR = PKG / "figures"
PROVENANCE_DIR = PKG / "provenance"


FIGURE_NAMES = [
    "pr_curve_fusion_vs_baselines.png",
    "roc_curve_fusion_vs_baselines.png",
    "confusion_matrix_final_raw.png",
    "confusion_matrix_final_norm.png",
    "ablation_bars.png",
    "missing_modality_bars.png",
    "stress_audio_snr_curve.png",
    "stress_vision_corruption_curve.png",
    "stress_radar_clutter_curve.png",
    "gating_weights_timeseries.png",
    "quality_vs_weight_audio.png",
    "quality_vs_weight_vision.png",
    "quality_vs_weight_radar.png",
    "latency_breakdown.png",
    "qualitative_overlays_grid.png",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_pr_and_roc(df: pd.DataFrame) -> None:
    model_cols = [
        ("F-AVG", "p_avg"),
        ("F-STATIC", "p_static"),
        ("F-GATE", "p_gate"),
    ]
    if "p_gate_noq" in df.columns:
        model_cols.append(("F-GATE-NOQ", "p_gate_noq"))

    y = df["y_true"].astype(int).to_numpy()

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for name, col in model_cols:
        scores = df[col].astype(float).to_numpy()
        precision, recall, _ = precision_recall_curve(y, scores)
        pr_auc = auc(recall, precision)
        ax.plot(recall, precision, linewidth=2.0, label=f"{name} (AUC={pr_auc:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.set_title("PR Curve: Fusion vs Baselines")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pr_curve_fusion_vs_baselines.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for name, col in model_cols:
        scores = df[col].astype(float).to_numpy()
        fpr, tpr, _ = roc_curve(y, scores)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2.0, label=f"{name} (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.2)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.set_title("ROC Curve: Fusion vs Baselines")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "roc_curve_fusion_vs_baselines.png", dpi=220)
    plt.close(fig)


def copy_confusion_matrices(placeholders: List[Dict[str, str]]) -> None:
    raw_target = FIG_DIR / "confusion_matrix_final_raw.png"
    norm_target = FIG_DIR / "confusion_matrix_final_norm.png"

    candidates_raw: List[Path] = []
    candidates_norm: List[Path] = []

    metrics_vision = RESULTS / "metrics_vision.json"
    if metrics_vision.exists():
        payload = json.loads(metrics_vision.read_text(encoding="utf-8"))
        save_dir = payload.get("save_dir")
        if isinstance(save_dir, str):
            d = Path(save_dir)
            candidates_raw.append(d / "confusion_matrix.png")
            candidates_norm.append(d / "confusion_matrix_normalized.png")

    candidates_raw.extend(sorted((ROOT / "runs").rglob("confusion_matrix.png")))
    candidates_norm.extend(sorted((ROOT / "runs").rglob("confusion_matrix_normalized.png")))

    raw_src = next((p for p in candidates_raw if p.exists()), None)
    norm_src = next((p for p in candidates_norm if p.exists()), None)

    if raw_src is not None:
        shutil.copy2(raw_src, raw_target)
    else:
        _make_placeholder(raw_target, "Missing confusion_matrix.png from run artifacts")
        placeholders.append(
            {"figure": "confusion_matrix_final_raw.png", "reason": "source confusion matrix not found"}
        )

    if norm_src is not None:
        shutil.copy2(norm_src, norm_target)
    else:
        _make_placeholder(norm_target, "Missing confusion_matrix_normalized.png from run artifacts")
        placeholders.append(
            {"figure": "confusion_matrix_final_norm.png", "reason": "source normalized confusion matrix not found"}
        )


def _make_placeholder(path: Path, message: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12, wrap=True)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_ablation_and_missing() -> None:
    ablation = pd.read_csv(RESULTS / "ablation_table.csv")
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    order = ["F-AVG", "F-STATIC", "F-GATE", "F-GATE-NOQ"]
    ablation = ablation.set_index("run_id").reindex(order).reset_index()
    ax.bar(ablation["run_id"], ablation["pr_auc"], color=["#4b5563", "#1f2937", "#0f766e", "#334155"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("PR-AUC")
    ax.set_title("Ablation Comparison (Test PR-AUC)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ablation_bars.png", dpi=220)
    plt.close(fig)

    missing = pd.read_csv(RESULTS / "missing_modality.csv")
    keep_scenarios = ["all_present", "vision_missing", "audio_missing", "radar_missing"]
    missing = missing[missing["scenario"].isin(keep_scenarios)].copy()
    methods = ["F-AVG", "F-STATIC", "F-GATE"]
    scen_order = keep_scenarios
    x = np.arange(len(scen_order))
    width = 0.23
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for i, method in enumerate(methods):
        vals = []
        for scen in scen_order:
            rows = missing[(missing["run_id"] == method) & (missing["scenario"] == scen)]
            vals.append(float(rows["pr_auc"].iloc[0]) if not rows.empty else np.nan)
        ax.bar(x + (i - 1) * width, vals, width=width, label=method)
    ax.set_xticks(x)
    ax.set_xticklabels(scen_order, rotation=15)
    ax.set_ylim(0, 1)
    ax.set_ylabel("PR-AUC")
    ax.set_title("Missing-Modality Robustness")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "missing_modality_bars.png", dpi=220)
    plt.close(fig)


def plot_stress_curves() -> None:
    audio = pd.read_csv(RESULTS / "stress_audio_snr.csv")
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for run_id, g in audio.groupby("run_id"):
        g = g.sort_values("snr_db")
        ax.plot(g["snr_db"], g["pr_auc"], marker="o", linewidth=2.0, label=run_id)
    ax.set_xlabel("Audio SNR (dB)")
    ax.set_ylabel("PR-AUC")
    ax.set_title("Stress Test: Audio SNR")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "stress_audio_snr_curve.png", dpi=220)
    plt.close(fig)

    vision = pd.read_csv(RESULTS / "stress_vision_corruptions.csv")
    vision_agg = (
        vision.groupby(["run_id", "severity"], as_index=False)["pr_auc"]
        .mean()
        .sort_values(["run_id", "severity"])
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for run_id, g in vision_agg.groupby("run_id"):
        ax.plot(g["severity"], g["pr_auc"], marker="o", linewidth=2.0, label=run_id)
    ax.set_xlabel("Vision Corruption Severity")
    ax.set_ylabel("PR-AUC")
    ax.set_title("Stress Test: Vision Corruptions")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "stress_vision_corruption_curve.png", dpi=220)
    plt.close(fig)

    radar = pd.read_csv(RESULTS / "stress_radar_clutter.csv")
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for run_id, g in radar.groupby("run_id"):
        g = g.sort_values("clutter_level")
        ax.plot(g["clutter_level"], g["pr_auc"], marker="o", linewidth=2.0, label=run_id)
    ax.set_xlabel("Radar Clutter Level")
    ax.set_ylabel("PR-AUC")
    ax.set_title("Stress Test: Radar Clutter")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "stress_radar_clutter_curve.png", dpi=220)
    plt.close(fig)


def plot_gate_interpretability(df: pd.DataFrame) -> None:
    plot_df = df.copy()
    plot_df = plot_df.sort_values("episode_id").reset_index(drop=True)
    plot_df["t_idx"] = np.arange(len(plot_df))

    series = plot_df.head(160)
    fig, ax = plt.subplots(figsize=(10.2, 4.2))
    ax.plot(series["t_idx"], series["w_vis"], label="w_vis", linewidth=2.0)
    ax.plot(series["t_idx"], series["w_aud"], label="w_aud", linewidth=2.0)
    ax.plot(series["t_idx"], series["w_rad"], label="w_rad", linewidth=2.0)
    ax.set_xlabel("Sample Index (proxy time)")
    ax.set_ylabel("Gate Weight")
    ax.set_ylim(0, 1)
    ax.set_title("Gating Weights Timeseries (Test Sample)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "gating_weights_timeseries.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.scatter(plot_df["audio_snr_db"], plot_df["w_aud"], s=12, alpha=0.35, color="#16a34a")
    bins = np.linspace(plot_df["audio_snr_db"].min(), plot_df["audio_snr_db"].max(), 8)
    plot_df["_snr_bin"] = pd.cut(plot_df["audio_snr_db"], bins=bins, include_lowest=True)
    means = plot_df.groupby("_snr_bin", observed=False)["w_aud"].mean().dropna()
    mids = np.array([interval.mid for interval in means.index])
    ax.plot(mids, means.values, color="black", linewidth=2.0, marker="o", label="Binned mean")
    ax.set_xlabel("Audio SNR (dB)")
    ax.set_ylabel("w_audio")
    ax.set_ylim(0, 1)
    ax.set_title("Audio Quality vs Gate Weight")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "quality_vs_weight_audio.png", dpi=220)
    plt.close(fig)

    vision_map = {
        "normal": 0.0,
        "low_light": 1.0,
        "wind_noise": 1.0,
        "hard_mix": 2.0,
        "radar_cluttered": 2.0,
    }
    plot_df["vision_quality_proxy"] = plot_df["vision_condition"].map(vision_map).fillna(1.5)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    jitter = np.random.default_rng(42).normal(0.0, 0.05, size=len(plot_df))
    ax.scatter(
        plot_df["vision_quality_proxy"] + jitter,
        plot_df["w_vis"],
        s=12,
        alpha=0.35,
        color="#2563eb",
    )
    means = plot_df.groupby("vision_quality_proxy", as_index=False)["w_vis"].mean()
    ax.plot(means["vision_quality_proxy"], means["w_vis"], color="black", marker="o", linewidth=2.0)
    ax.set_xticks([0.0, 1.0, 2.0], ["normal", "mild", "hard"])
    ax.set_xlabel("Vision Condition Proxy")
    ax.set_ylabel("w_vision")
    ax.set_ylim(0, 1)
    ax.set_title("Vision Quality Proxy vs Gate Weight")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "quality_vs_weight_vision.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.scatter(
        plot_df["radar_clutter_level"],
        plot_df["w_rad"],
        s=12,
        alpha=0.35,
        color="#dc2626",
    )
    bins = np.linspace(plot_df["radar_clutter_level"].min(), plot_df["radar_clutter_level"].max(), 8)
    plot_df["_cl_bin"] = pd.cut(plot_df["radar_clutter_level"], bins=bins, include_lowest=True)
    means = plot_df.groupby("_cl_bin", observed=False)["w_rad"].mean().dropna()
    mids = np.array([interval.mid for interval in means.index])
    ax.plot(mids, means.values, color="black", marker="o", linewidth=2.0, label="Binned mean")
    ax.set_xlabel("Radar Clutter Level")
    ax.set_ylabel("w_radar")
    ax.set_ylim(0, 1)
    ax.set_title("Radar Clutter vs Gate Weight")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "quality_vs_weight_radar.png", dpi=220)
    plt.close(fig)


def plot_latency() -> None:
    payload = json.loads((RESULTS / "latency_summary.json").read_text(encoding="utf-8"))
    rows: List[Tuple[str, float]] = []
    methods = payload.get("methods", {})
    for method, values in methods.items():
        wall = values.get("wall_clock", {})
        rows.append((method, float(wall.get("mean_ms", np.nan))))
    rows = sorted(rows, key=lambda x: x[0])
    df = pd.DataFrame(rows, columns=["method", "mean_ms"])
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.bar(df["method"], df["mean_ms"], color="#0f766e")
    ax.set_ylabel("Mean Latency (ms)")
    ax.set_title("Latency Breakdown by Fusion Method")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "latency_breakdown.png", dpi=220)
    plt.close(fig)


def plot_qualitative_grid(placeholders: List[Dict[str, str]]) -> None:
    label_viz = RESULTS / "label_viz"
    if not label_viz.exists():
        _make_placeholder(FIG_DIR / "qualitative_overlays_grid.png", "results/label_viz missing")
        placeholders.append(
            {"figure": "qualitative_overlays_grid.png", "reason": "results/label_viz missing"}
        )
        return

    picks: List[Path] = []
    for cls in ["AIRPLANE", "BIRD", "DRONE", "HELICOPTER"]:
        cls_imgs = sorted(label_viz.glob(f"*_{cls}.png"))
        picks.extend(cls_imgs[:3])
    picks = picks[:12]
    if not picks:
        _make_placeholder(FIG_DIR / "qualitative_overlays_grid.png", "No overlay samples found")
        placeholders.append(
            {"figure": "qualitative_overlays_grid.png", "reason": "no overlay png found"}
        )
        return

    fig, axes = plt.subplots(3, 4, figsize=(14, 9))
    for i in range(12):
        ax = axes[i // 4][i % 4]
        if i < len(picks):
            img = Image.open(picks[i]).convert("RGB")
            ax.imshow(img)
            ax.set_title(picks[i].stem, fontsize=8)
            ax.axis("off")
        else:
            ax.axis("off")
    fig.suptitle("Qualitative Overlay Examples", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "qualitative_overlays_grid.png", dpi=220)
    plt.close(fig)


def main() -> int:
    ensure_dir(FIG_DIR)
    ensure_dir(PROVENANCE_DIR)

    placeholders: List[Dict[str, str]] = []
    preds = pd.read_csv(RESULTS / "preds_fusion.csv")
    test_df = preds[preds["split"] == "test"].copy()
    if test_df.empty:
        raise RuntimeError("results/preds_fusion.csv has no test split rows")

    plot_pr_and_roc(test_df)
    copy_confusion_matrices(placeholders)
    plot_ablation_and_missing()
    plot_stress_curves()
    plot_gate_interpretability(test_df)
    plot_latency()
    plot_qualitative_grid(placeholders)

    for name in FIGURE_NAMES:
        p = FIG_DIR / name
        if not p.exists():
            _make_placeholder(p, f"Auto-generated placeholder: missing source for {name}")
            placeholders.append({"figure": name, "reason": "auto placeholder fallback"})

    write_payload = {
        "generated_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "figure_dir": str(FIG_DIR.relative_to(ROOT)).replace("\\", "/"),
        "figures": FIGURE_NAMES,
        "placeholders": placeholders,
    }
    (PROVENANCE_DIR / "figure_generation_manifest.json").write_text(
        json.dumps(write_payload, indent=2), encoding="utf-8"
    )
    print("Generated figures:")
    for name in FIGURE_NAMES:
        print(f" - {name}")
    if placeholders:
        print("Placeholder figures were required:")
        for item in placeholders:
            print(f" - {item['figure']}: {item['reason']}")
    else:
        print("No placeholders used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
