from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required input missing: {path}")
    return path


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_ablation(ablation_csv: Path, out_png: Path) -> None:
    df = pd.read_csv(_require(ablation_csv))
    req = {"run_id", "pr_auc"}
    missing = req - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns in {ablation_csv}: {sorted(missing)}")
    order = ["F-AVG", "F-STATIC", "F-GATE", "F-GATE-NOQ"]
    d = df[df["run_id"].isin(order)].copy()
    d["run_id"] = pd.Categorical(d["run_id"], categories=order, ordered=True)
    d = d.sort_values("run_id")

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(d["run_id"], d["pr_auc"], color=["#4c78a8", "#f58518", "#54a24b", "#e45756"])
    ax.set_ylabel("PR-AUC")
    ax.set_title("Fusion Ablation (TEST)")
    ax.grid(axis="y", alpha=0.25)
    _save(fig, out_png)


def plot_stress_curves(
    audio_csv: Path,
    vision_csv: Path,
    radar_csv: Path,
    out_png: Path,
) -> None:
    a = pd.read_csv(_require(audio_csv))
    v = pd.read_csv(_require(vision_csv))
    r = pd.read_csv(_require(radar_csv))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for run_id, g in a.groupby("run_id"):
        gs = g.sort_values("snr_db")
        axes[0].plot(gs["snr_db"], gs["pr_auc"], marker="o", label=run_id)
    axes[0].set_title("Audio SNR Stress")
    axes[0].set_xlabel("SNR (dB)")
    axes[0].set_ylabel("PR-AUC")
    axes[0].grid(alpha=0.25)

    if {"corruption", "severity", "run_id", "pr_auc"}.issubset(v.columns):
        agg = (
            v.groupby(["run_id", "severity"], as_index=False)["pr_auc"]
            .mean()
            .sort_values("severity")
        )
        for run_id, g in agg.groupby("run_id"):
            axes[1].plot(g["severity"], g["pr_auc"], marker="o", label=run_id)
    axes[1].set_title("Vision Corruption Stress")
    axes[1].set_xlabel("Severity")
    axes[1].set_ylabel("PR-AUC")
    axes[1].grid(alpha=0.25)

    for run_id, g in r.groupby("run_id"):
        gs = g.sort_values("clutter_level")
        axes[2].plot(gs["clutter_level"], gs["pr_auc"], marker="o", label=run_id)
    axes[2].set_title("Radar Clutter Stress")
    axes[2].set_xlabel("Clutter Level")
    axes[2].set_ylabel("PR-AUC")
    axes[2].grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    _save(fig, out_png)


def plot_missing_modality(missing_csv: Path, out_png: Path) -> None:
    df = pd.read_csv(_require(missing_csv))
    req = {"scenario", "run_id", "pr_auc"}
    missing = req - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns in {missing_csv}: {sorted(missing)}")

    scenarios = [
        "all_present",
        "vision_missing",
        "audio_missing",
        "radar_missing",
        "vision_only",
        "audio_only",
        "radar_only",
    ]
    methods = ["F-AVG", "F-STATIC", "F-GATE", "F-GATE-NOQ"]
    pivot = (
        df.pivot_table(index="scenario", columns="run_id", values="pr_auc", aggfunc="mean")
        .reindex(index=scenarios, columns=methods)
        .fillna(np.nan)
    )

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    x = np.arange(len(pivot.index))
    width = 0.2
    for i, method in enumerate(methods):
        ax.bar(x + (i - 1.5) * width, pivot[method].to_numpy(), width=width, label=method)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=25, ha="right")
    ax.set_ylabel("PR-AUC")
    ax.set_title("Missing-Modality Comparison (TEST)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    _save(fig, out_png)


def _mean_sem(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    g = df.groupby(x_col)[y_col]
    out = g.agg(["mean", "std", "count"]).reset_index()
    out["sem"] = out["std"] / np.sqrt(out["count"].clip(lower=1))
    return out


def plot_gate_behavior(pred_csv: Path, out_png: Path) -> None:
    df = pd.read_csv(_require(pred_csv))
    req = {"split", "vision_condition", "audio_snr_db", "w_gate_vis", "w_gate_aud", "w_gate_rad"}
    missing = req - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns in {pred_csv}: {sorted(missing)}")
    d = df[df["split"].astype(str) == "test"].copy()
    if d.empty:
        d = df.copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    cond_order = sorted(d["vision_condition"].dropna().astype(str).unique().tolist())
    for w_col, label in [("w_gate_vis", "Vision"), ("w_gate_aud", "Audio"), ("w_gate_rad", "Radar")]:
        s = _mean_sem(d, "vision_condition", w_col)
        s["vision_condition"] = pd.Categorical(s["vision_condition"], categories=cond_order, ordered=True)
        s = s.sort_values("vision_condition")
        x = np.arange(len(s))
        axes[0].errorbar(x, s["mean"], yerr=s["sem"], marker="o", capsize=3, label=label)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(s["vision_condition"].astype(str), rotation=20, ha="right")
    axes[0].set_title("Gate Weights vs Vision Condition")
    axes[0].set_ylabel("Mean Weight")
    axes[0].grid(alpha=0.25)

    bins = np.array([-80, -30, -20, -10, 0, 10, 20, 40], dtype=np.float64)
    d["snr_bin"] = pd.cut(d["audio_snr_db"], bins=bins, include_lowest=True)
    for w_col, label in [("w_gate_vis", "Vision"), ("w_gate_aud", "Audio"), ("w_gate_rad", "Radar")]:
        s = _mean_sem(d.dropna(subset=["snr_bin"]), "snr_bin", w_col)
        x = np.arange(len(s))
        axes[1].errorbar(x, s["mean"], yerr=s["sem"], marker="o", capsize=3, label=label)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([str(v) for v in s["snr_bin"]], rotation=25, ha="right")
    axes[1].set_title("Gate Weights vs Audio SNR Bin")
    axes[1].set_ylabel("Mean Weight")
    axes[1].grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    _save(fig, out_png)


def plot_latency(latency_json: Path, out_png: Path) -> None:
    data = json.loads(_require(latency_json).read_text(encoding="utf-8"))
    if "methods" not in data:
        raise KeyError(f"Missing 'methods' in {latency_json}")

    methods = []
    means = []
    p95 = []
    for method, stats in data["methods"].items():
        wall = stats.get("wall_clock", {})
        methods.append(method)
        means.append(float(wall.get("mean_ms", np.nan)))
        p95.append(float(wall.get("p95_ms", np.nan)))

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    x = np.arange(len(methods))
    ax.bar(x, means, color="#4c78a8", alpha=0.85, label="Mean")
    ax.plot(x, p95, color="#e45756", marker="o", linewidth=2, label="P95")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Fusion Latency Summary")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    _save(fig, out_png)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fig-dir", type=Path, default=Path("results/figures"))
    args = parser.parse_args()

    fig_dir = args.fig_dir
    plot_ablation(Path("results/ablation_table.csv"), fig_dir / "ablation_comparison.png")
    plot_stress_curves(
        Path("results/stress_audio_snr.csv"),
        Path("results/stress_vision_corruptions.csv"),
        Path("results/stress_radar_clutter.csv"),
        fig_dir / "stress_curves.png",
    )
    plot_missing_modality(Path("results/missing_modality.csv"), fig_dir / "missing_modality_comparison.png")
    plot_gate_behavior(Path("results/preds_fusion.csv"), fig_dir / "gate_weight_behavior.png")
    plot_latency(Path("results/latency_summary.json"), fig_dir / "latency_plot.png")
    print(f"Wrote figures to {fig_dir}")


if __name__ == "__main__":
    main()
