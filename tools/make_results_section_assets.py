from __future__ import annotations

import json
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve


ROOT = Path(__file__).resolve().parents[1]
FINAL_PKG = ROOT / "final_isef_package"
OUT = FINAL_PKG / "results_section"
FIG_DIR = OUT / "figures"
TAB_DIR = OUT / "tables"
DATA_DIR = OUT / "data_exports"

RESULTS = ROOT / "results"
EPISODES = ROOT / "data" / "processed" / "episodes" / "episodes.csv"


@dataclass(frozen=True)
class CaptionItem:
    asset_id: str
    filename: str
    what: str
    sources: List[str]
    caption: str


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linestyle": "-",
            "axes.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def ensure_dirs() -> None:
    for p in [OUT, FIG_DIR, TAB_DIR, DATA_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def req(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file missing: {path}")
    return path


def save_fig(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            fig.tight_layout()
        except Exception:
            pass
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def metrics_with_derivatives(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    d = df.copy()
    for c in ["tn", "fp", "fn", "tp", "pr_auc", "roc_auc"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["precision"] = d["tp"] / (d["tp"] + d["fp"]).replace(0, np.nan)
    d["recall"] = d["tp"] / (d["tp"] + d["fn"]).replace(0, np.nan)
    d["f1"] = (
        2.0
        * d["precision"].fillna(0.0)
        * d["recall"].fillna(0.0)
        / (d["precision"].fillna(0.0) + d["recall"].fillna(0.0)).replace(0, np.nan)
    ).fillna(0.0)
    d["accuracy"] = (d["tp"] + d["tn"]) / (d["tp"] + d["tn"] + d["fp"] + d["fn"]).replace(0, np.nan)
    d["far"] = d["fp"] / (d["fp"] + d["tn"]).replace(0, np.nan)
    d["threshold"] = threshold
    return d


def write_table_png(df: pd.DataFrame, out_png: Path, col_widths: List[float] | None = None) -> None:
    n_rows, n_cols = df.shape
    fig_w = max(10.0, 1.55 * n_cols)
    fig_h = max(2.2, 0.5 + 0.48 * (n_rows + 1))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    cell_text = []
    for _, row in df.iterrows():
        vals = []
        for v in row:
            if isinstance(v, (float, np.floating)):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        cell_text.append(vals)

    table = ax.table(
        cellText=cell_text,
        colLabels=df.columns.tolist(),
        cellLoc="center",
        colLoc="center",
        bbox=[0.0, 0.0, 1.0, 1.0],
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1.0, 1.30)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#AAB7C8")
        if r == 0:
            cell.set_facecolor("#1F4E79")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#FFFFFF" if r % 2 else "#F6F8FB")

    save_fig(fig, out_png)


def load_base() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mu = pd.read_csv(req(RESULTS / "metrics_unimodal.csv"))
    mf = pd.read_csv(req(RESULTS / "metrics_fusion.csv"))
    preds = pd.read_csv(req(RESULTS / "preds_fusion.csv"))
    return mu, mf, preds


def build_table_r1(mu: pd.DataFrame, mf: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    d = pd.concat([mu, mf], ignore_index=True)
    d = d[d["split"].astype(str) == "test"].copy()
    d = metrics_with_derivatives(d, threshold=0.5)

    order = ["UNI-V", "UNI-A", "UNI-R", "F-AVG", "F-STATIC", "F-GATE", "F-GATE-NOQ"]
    d["run_id"] = pd.Categorical(d["run_id"], categories=order, ordered=True)
    d = d.sort_values("run_id")

    out = d[
        [
            "run_id",
            "pr_auc",
            "roc_auc",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "far",
            "threshold",
        ]
    ].rename(
        columns={
            "run_id": "Model",
            "pr_auc": "PR-AUC",
            "roc_auc": "AUROC",
            "accuracy": "Accuracy",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1",
            "far": "FAR",
            "threshold": "Threshold",
        }
    )
    out.to_csv(TAB_DIR / "R1_overall_test_performance.csv", index=False)
    write_table_png(out, TAB_DIR / "R1_overall_test_performance.png")

    uni = out[out["Model"].str.startswith("UNI-")].copy()
    best_unimodal = str(uni.sort_values("PR-AUC", ascending=False).iloc[0]["Model"])
    return out, best_unimodal


def score_col_for_model(model: str) -> str:
    mapping = {
        "UNI-V": "p_vis",
        "UNI-A": "p_aud",
        "UNI-R": "p_rad",
        "F-AVG": "p_avg",
        "F-STATIC": "p_static",
        "F-GATE": "p_gate",
        "F-GATE-NOQ": "p_gate_noq",
    }
    if model not in mapping:
        raise KeyError(f"Unknown model: {model}")
    return mapping[model]


def build_r1_r2_curves(preds: pd.DataFrame, best_unimodal: str) -> tuple[Path, Path]:
    d = preds[preds["split"].astype(str) == "test"].copy()
    y = d["y_true"].astype(int).to_numpy()

    selected = [best_unimodal, "F-AVG", "F-STATIC", "F-GATE", "F-GATE-NOQ"]
    palette = {
        best_unimodal: "#4C78A8",
        "F-AVG": "#9AA5B1",
        "F-STATIC": "#F58518",
        "F-GATE": "#2E8B57",
        "F-GATE-NOQ": "#D62728",
    }

    pr_rows: List[Dict[str, float | str]] = []
    roc_rows: List[Dict[str, float | str]] = []

    fig_pr, ax_pr = plt.subplots(figsize=(9.0, 6.2))
    for m in selected:
        col = score_col_for_model(m)
        s = d[col].astype(float).to_numpy()
        p, r, _ = precision_recall_curve(y, s)
        ap = auc(r, p)
        ax_pr.plot(r, p, lw=2.4, color=palette[m], label=f"{m} (AP={ap:.3f})")
        pr_rows.extend({"model": m, "recall": float(rr), "precision": float(pp)} for rr, pp in zip(r, p))

    ax_pr.set_xlim(0, 1)
    ax_pr.set_ylim(0, 1)
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("TEST Precision-Recall Curves")
    ax_pr.legend(loc="lower left", frameon=True)
    save_fig(fig_pr, FIG_DIR / "R1_pr_curves_test.png")
    pd.DataFrame(pr_rows).to_csv(DATA_DIR / "R1_pr_curves_test_points.csv", index=False)

    fig_roc, ax_roc = plt.subplots(figsize=(9.0, 6.2))
    for m in selected:
        col = score_col_for_model(m)
        s = d[col].astype(float).to_numpy()
        fpr, tpr, _ = roc_curve(y, s)
        roc_auc = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, lw=2.4, color=palette[m], label=f"{m} (AUROC={roc_auc:.3f})")
        roc_rows.extend({"model": m, "fpr": float(ff), "tpr": float(tt)} for ff, tt in zip(fpr, tpr))
    ax_roc.plot([0, 1], [0, 1], "--", color="#8A8A8A", lw=1.5)
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("TEST ROC Curves")
    ax_roc.legend(loc="lower right", frameon=True)
    save_fig(fig_roc, FIG_DIR / "R2_roc_curves_test.png")
    pd.DataFrame(roc_rows).to_csv(DATA_DIR / "R2_roc_curves_test_points.csv", index=False)
    return FIG_DIR / "R1_pr_curves_test.png", FIG_DIR / "R2_roc_curves_test.png"


def build_r3_confusions(preds: pd.DataFrame) -> Path:
    d = preds[preds["split"].astype(str) == "test"].copy()
    y = d["y_true"].astype(int).to_numpy()
    models = ["UNI-V", "UNI-A", "UNI-R", "F-AVG", "F-STATIC", "F-GATE", "F-GATE-NOQ"]

    cms: Dict[str, np.ndarray] = {}
    vmax = 1
    for m in models:
        yp = (d[score_col_for_model(m)].to_numpy(dtype=float) >= 0.5).astype(int)
        cm = confusion_matrix(y, yp, labels=[0, 1])
        cms[m] = cm
        vmax = max(vmax, int(cm.max()))

    rows: List[dict] = []
    for m, cm in cms.items():
        rows.append({"Model": m, "TN": int(cm[0, 0]), "FP": int(cm[0, 1]), "FN": int(cm[1, 0]), "TP": int(cm[1, 1])})
    pd.DataFrame(rows).to_csv(DATA_DIR / "R3_confusion_counts_test.csv", index=False)

    fig, axes = plt.subplots(2, 4, figsize=(16.0, 8.6))
    axes = axes.flatten()
    for i, m in enumerate(models):
        ax = axes[i]
        cm = cms[m]
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=vmax)
        for rr in range(2):
            for cc in range(2):
                ax.text(cc, rr, f"{cm[rr, cc]}", ha="center", va="center", fontsize=11, fontweight="bold")
        ax.set_title(m, fontsize=12.5, pad=8)
        ax.set_xticks([0, 1], ["Pred 0", "Pred 1"])
        ax.set_yticks([0, 1], ["True 0", "True 1"])

    axes[-1].axis("off")
    cbar = fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.02)
    cbar.ax.set_ylabel("Episode count", rotation=90)
    save_fig(fig, FIG_DIR / "R3_confusion_matrices_binary_test.png")
    return FIG_DIR / "R3_confusion_matrices_binary_test.png"


def choose_best_unimodal_available(
    scenario: str,
    uni_rows: pd.DataFrame,
) -> Tuple[str, float, float]:
    # Returns (model_name, pr_auc, far)
    uni = {r["Model"]: r for _, r in uni_rows.iterrows()}
    far = lambda m: float(uni[m]["FAR"])  # noqa: E731
    pra = lambda m: float(uni[m]["PR-AUC"])  # noqa: E731

    if scenario == "vision_only":
        m = "UNI-V"
    elif scenario == "audio_only":
        m = "UNI-A"
    elif scenario == "radar_only":
        m = "UNI-R"
    elif scenario == "vision_missing":
        m = max(["UNI-A", "UNI-R"], key=pra)
    elif scenario == "audio_missing":
        m = max(["UNI-V", "UNI-R"], key=pra)
    elif scenario == "radar_missing":
        m = max(["UNI-V", "UNI-A"], key=pra)
    else:  # all_present
        m = max(["UNI-V", "UNI-A", "UNI-R"], key=pra)
    return m, pra(m), far(m)


def build_r4_r2(missing_df: pd.DataFrame, r1: pd.DataFrame) -> tuple[Path, Path]:
    scenarios = ["all_present", "vision_missing", "audio_missing", "radar_missing", "vision_only", "audio_only", "radar_only"]
    mm = missing_df.copy()
    mm = mm[mm["scenario"].isin(scenarios)]

    uni_rows = r1[r1["Model"].str.startswith("UNI-")].copy()
    records: List[dict] = []
    for scen in scenarios:
        uni_model, uni_pr, uni_far = choose_best_unimodal_available(scen, uni_rows)

        row_avg = mm[(mm["scenario"] == scen) & (mm["run_id"] == "F-AVG")].iloc[0]
        row_gate = mm[(mm["scenario"] == scen) & (mm["run_id"] == "F-GATE")].iloc[0]

        far_avg = float(row_avg["fp"]) / max(1.0, float(row_avg["fp"] + row_avg["tn"]))
        far_gate = float(row_gate["fp"]) / max(1.0, float(row_gate["fp"] + row_gate["tn"]))

        records.append(
            {
                "Scenario": scen,
                "UNI_best_available": uni_model,
                "UNI_best_available_PR-AUC": uni_pr,
                "UNI_best_available_FAR": uni_far,
                "F-AVG_PR-AUC": float(row_avg["pr_auc"]),
                "F-AVG_FAR": far_avg,
                "F-GATE_PR-AUC": float(row_gate["pr_auc"]),
                "F-GATE_FAR": far_gate,
            }
        )

    r2 = pd.DataFrame(records)
    r2.to_csv(TAB_DIR / "R2_missing_modality_per_scenario.csv", index=False)
    write_table_png(r2, TAB_DIR / "R2_missing_modality_per_scenario.png")

    plot = r2.copy()
    x = np.arange(len(plot))
    w = 0.25
    fig, ax = plt.subplots(figsize=(11.4, 6.0))
    ax.bar(x - w, plot["UNI_best_available_PR-AUC"], width=w, label="UNI-best-available", color="#4C78A8")
    ax.bar(x, plot["F-AVG_PR-AUC"], width=w, label="F-AVG", color="#9AA5B1")
    ax.bar(x + w, plot["F-GATE_PR-AUC"], width=w, label="F-GATE", color="#2E8B57")
    ax.set_xticks(x)
    ax.set_xticklabels(plot["Scenario"], rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("PR-AUC")
    ax.set_title("Missing-Modality Robustness (TEST)")
    ax.legend(loc="upper right")
    save_fig(fig, FIG_DIR / "R4_missing_modality_robustness.png")

    return FIG_DIR / "R4_missing_modality_robustness.png", TAB_DIR / "R2_missing_modality_per_scenario.png"


def build_r5_r3(mf: pd.DataFrame, missing_df: pd.DataFrame) -> tuple[Path, Path]:
    mf2 = metrics_with_derivatives(mf[mf["split"].astype(str) == "test"].copy(), threshold=0.5)
    gate = mf2[mf2["run_id"] == "F-GATE"].iloc[0]
    noq = mf2[mf2["run_id"] == "F-GATE-NOQ"].iloc[0]

    r3 = pd.DataFrame(
        [
            {
                "Delta": "F-GATE - F-GATE-NOQ",
                "PR-AUC_delta": float(gate["pr_auc"] - noq["pr_auc"]),
                "AUROC_delta": float(gate["roc_auc"] - noq["roc_auc"]),
                "Accuracy_delta": float(gate["accuracy"] - noq["accuracy"]),
                "FAR_delta": float(gate["far"] - noq["far"]),
            }
        ]
    )
    r3.to_csv(TAB_DIR / "R3_ablation_deltas_gate_vs_noq.csv", index=False)
    write_table_png(r3, TAB_DIR / "R3_ablation_deltas_gate_vs_noq.png")

    mm = missing_df.copy()
    rows = []
    overall = {
        "Scenario": "overall_test",
        "F-GATE_PR-AUC": float(gate["pr_auc"]),
        "F-GATE-NOQ_PR-AUC": float(noq["pr_auc"]),
    }
    rows.append(overall)
    for scen in ["all_present", "vision_missing", "audio_missing", "radar_missing", "vision_only", "audio_only", "radar_only"]:
        g = mm[(mm["scenario"] == scen) & (mm["run_id"] == "F-GATE")].iloc[0]
        n = mm[(mm["scenario"] == scen) & (mm["run_id"] == "F-GATE-NOQ")].iloc[0]
        rows.append(
            {
                "Scenario": scen,
                "F-GATE_PR-AUC": float(g["pr_auc"]),
                "F-GATE-NOQ_PR-AUC": float(n["pr_auc"]),
            }
        )
    d = pd.DataFrame(rows)
    d["Delta_PR-AUC_(GATE-NOQ)"] = d["F-GATE_PR-AUC"] - d["F-GATE-NOQ_PR-AUC"]
    d.to_csv(DATA_DIR / "R5_gate_vs_noq_by_scenario.csv", index=False)

    fig, ax = plt.subplots(figsize=(11.0, 5.8))
    colors = ["#2E8B57" if v >= 0 else "#D62728" for v in d["Delta_PR-AUC_(GATE-NOQ)"]]
    x = np.arange(len(d))
    ax.bar(x, d["Delta_PR-AUC_(GATE-NOQ)"], color=colors)
    ax.axhline(0.0, color="#333333", lw=1.2)
    ax.set_ylabel("PR-AUC Delta")
    ax.set_title("F-GATE vs F-GATE-NOQ by Scenario (TEST)")
    ax.set_xticks(x)
    ax.set_xticklabels(d["Scenario"], rotation=20, ha="right")
    save_fig(fig, FIG_DIR / "R5_gate_vs_noq_delta.png")

    return FIG_DIR / "R5_gate_vs_noq_delta.png", TAB_DIR / "R3_ablation_deltas_gate_vs_noq.png"


def build_r6_stress(mu: pd.DataFrame, stress_audio: pd.DataFrame, stress_vision: pd.DataFrame, stress_radar: pd.DataFrame) -> tuple[Path, Path, Path]:
    um = mu[mu["split"].astype(str) == "test"].set_index("run_id")
    uni_lines = {
        "UNI-V": float(um.loc["UNI-V", "pr_auc"]),
        "UNI-A": float(um.loc["UNI-A", "pr_auc"]),
        "UNI-R": float(um.loc["UNI-R", "pr_auc"]),
    }

    # R6a Audio SNR
    a = stress_audio[stress_audio["run_id"] == "F-GATE"].sort_values("snr_db")
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ax.plot(a["snr_db"], a["pr_auc"], marker="o", lw=2.5, color="#2E8B57", label="F-GATE")
    for k, v in uni_lines.items():
        ax.axhline(v, ls="--", lw=1.8, label=f"{k} (test baseline)")
    ax.set_xlabel("Audio SNR (dB)")
    ax.set_ylabel("PR-AUC")
    ax.set_ylim(0, 1)
    ax.set_title("Stress Test: Audio SNR")
    ax.legend(loc="lower right", ncol=2)
    save_fig(fig, FIG_DIR / "R6a_stress_audio_snr.png")
    a.to_csv(DATA_DIR / "R6a_stress_audio_fgate.csv", index=False)

    # R6b Vision severity (mean over corruption type)
    v = stress_vision[stress_vision["run_id"] == "F-GATE"].groupby("severity", as_index=False)["pr_auc"].mean().sort_values("severity")
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ax.plot(v["severity"], v["pr_auc"], marker="o", lw=2.5, color="#2E8B57", label="F-GATE")
    for k, val in uni_lines.items():
        ax.axhline(val, ls="--", lw=1.8, label=f"{k} (test baseline)")
    ax.set_xlabel("Vision Corruption Severity")
    ax.set_ylabel("PR-AUC")
    ax.set_ylim(0, 1)
    ax.set_title("Stress Test: Vision Corruption")
    ax.legend(loc="lower right", ncol=2)
    save_fig(fig, FIG_DIR / "R6b_stress_vision_corruption.png")
    v.to_csv(DATA_DIR / "R6b_stress_vision_fgate.csv", index=False)

    # R6c Radar clutter
    r = stress_radar[stress_radar["run_id"] == "F-GATE"].sort_values("clutter_level")
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ax.plot(r["clutter_level"], r["pr_auc"], marker="o", lw=2.5, color="#2E8B57", label="F-GATE")
    for k, val in uni_lines.items():
        ax.axhline(val, ls="--", lw=1.8, label=f"{k} (test baseline)")
    ax.set_xlabel("Radar Clutter Level")
    ax.set_ylabel("PR-AUC")
    ax.set_ylim(0, 1)
    ax.set_title("Stress Test: Radar Clutter")
    ax.legend(loc="lower right", ncol=2)
    save_fig(fig, FIG_DIR / "R6c_stress_radar_clutter.png")
    r.to_csv(DATA_DIR / "R6c_stress_radar_fgate.csv", index=False)

    return (
        FIG_DIR / "R6a_stress_audio_snr.png",
        FIG_DIR / "R6b_stress_vision_corruption.png",
        FIG_DIR / "R6c_stress_radar_clutter.png",
    )


def build_r4_split_table() -> Path:
    ep = pd.read_csv(req(EPISODES))
    rows = []
    for split, g in ep.groupby("split"):
        pos = int((g["label_drone"] == 1).sum())
        neg = int((g["label_drone"] == 0).sum())
        total = int(len(g))
        rows.append(
            {
                "Split": split,
                "Episodes": total,
                "Positives": pos,
                "Negatives": neg,
                "PositiveRate": pos / max(1, total),
            }
        )
    d = pd.DataFrame(rows).sort_values("Split")
    d.to_csv(TAB_DIR / "R4_dataset_split_counts.csv", index=False)
    write_table_png(d, TAB_DIR / "R4_dataset_split_counts.png")
    return TAB_DIR / "R4_dataset_split_counts.png"


def build_or_reuse_r7_latency() -> Path:
    curated = FINAL_PKG / "figures" / "latency_breakdown.png"
    out = FIG_DIR / "R7_latency_summary.png"
    if curated.exists():
        shutil.copy2(curated, out)
        return out

    lat = json.loads(req(RESULTS / "latency_summary.json").read_text(encoding="utf-8"))
    rows = []
    for m, stats in lat.get("methods", {}).items():
        wall = stats.get("wall_clock", {})
        rows.append({"Method": m, "Mean_ms": float(wall.get("mean_ms", np.nan)), "P95_ms": float(wall.get("p95_ms", np.nan))})
    d = pd.DataFrame(rows).sort_values("Method")
    d.to_csv(DATA_DIR / "R7_latency_summary_values.csv", index=False)
    fig, ax = plt.subplots(figsize=(8.4, 5.1))
    x = np.arange(len(d))
    ax.bar(x, d["Mean_ms"], color="#4C78A8", label="Mean")
    ax.plot(x, d["P95_ms"], color="#D62728", marker="o", lw=2.0, label="P95")
    ax.set_xticks(x)
    ax.set_xticklabels(d["Method"])
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Fusion Latency Summary")
    ax.legend(loc="upper left")
    save_fig(fig, out)
    return out


def build_readme(captions: Iterable[CaptionItem]) -> None:
    lines = [
        "# Results Section Asset Manifest",
        "",
        "This folder contains poster-ready RESULTS assets for ISEF science fair layout.",
        "All metrics are TEST split unless explicitly stated.",
        "",
        "## Asset Index",
    ]
    for c in captions:
        lines += [
            f"### {c.asset_id}: `{c.filename}`",
            f"- **What it shows:** {c.what}",
            f"- **Data sources:** {', '.join(c.sources)}",
            f"- **Poster caption (paste as-is):** {c.caption}",
            "",
        ]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    setup_style()
    ensure_dirs()

    mu, mf, preds = load_base()
    missing = pd.read_csv(req(RESULTS / "missing_modality.csv"))
    stress_audio = pd.read_csv(req(RESULTS / "stress_audio_snr.csv"))
    stress_vision = pd.read_csv(req(RESULTS / "stress_vision_corruptions.csv"))
    stress_radar = pd.read_csv(req(RESULTS / "stress_radar_clutter.csv"))

    r1, best_uni = build_table_r1(mu, mf)
    build_r1_r2_curves(preds, best_uni)
    build_r3_confusions(preds)
    build_r4_r2(missing, r1)
    build_r5_r3(mf, missing)
    build_r6_stress(mu, stress_audio, stress_vision, stress_radar)
    build_r4_split_table()
    build_or_reuse_r7_latency()

    captions = [
        CaptionItem(
            "Table R1",
            "tables/R1_overall_test_performance.png (+ .csv)",
            "Overall TEST comparison of all unimodal and fusion variants with thresholded operating metrics.",
            ["results/metrics_unimodal.csv", "results/metrics_fusion.csv", "results/preds_fusion.csv"],
            "Table R1. Episode-level TEST performance across unimodal and fusion variants at a fixed threshold of 0.5 (FAR = FP/(FP+TN)).",
        ),
        CaptionItem(
            "Figure R1",
            "figures/R1_pr_curves_test.png",
            f"Precision-Recall overlay for {best_uni} plus fusion baselines and adaptive variants.",
            ["results/preds_fusion.csv"],
            f"Figure R1. TEST precision-recall curves for {best_uni}, F-AVG, F-STATIC, F-GATE, and F-GATE-NOQ.",
        ),
        CaptionItem(
            "Figure R2",
            "figures/R2_roc_curves_test.png",
            "ROC overlay for the same model set as Figure R1.",
            ["results/preds_fusion.csv"],
            "Figure R2. TEST ROC curves for the same models as Figure R1, using identical episode-level predictions.",
        ),
        CaptionItem(
            "Figure R3",
            "figures/R3_confusion_matrices_binary_test.png",
            "2x4 panel of binary episode-level confusion matrices for all 7 models at threshold 0.5.",
            ["results/preds_fusion.csv"],
            "Figure R3. Episode-level binary confusion matrices (threshold = 0.5) for UNI-V, UNI-A, UNI-R, F-AVG, F-STATIC, F-GATE, and F-GATE-NOQ.",
        ),
        CaptionItem(
            "Figure R4",
            "figures/R4_missing_modality_robustness.png",
            "PR-AUC vs missing-modality scenario comparing UNI-best-available, F-AVG, and F-GATE.",
            ["results/missing_modality.csv", "results/metrics_unimodal.csv"],
            "Figure R4. Missing-modality robustness on TEST across all_present, single-missing, and single-modality scenarios.",
        ),
        CaptionItem(
            "Table R2",
            "tables/R2_missing_modality_per_scenario.png (+ .csv)",
            "Per-scenario PR-AUC and FAR for UNI-best-available, F-AVG, and F-GATE.",
            ["results/missing_modality.csv", "results/metrics_unimodal.csv"],
            "Table R2. Per-scenario missing-modality performance; UNI-best-available selects the strongest available unimodal branch for each scenario.",
        ),
        CaptionItem(
            "Figure R5",
            "figures/R5_gate_vs_noq_delta.png",
            "Scenario-wise delta PR-AUC between F-GATE and F-GATE-NOQ (overall TEST + missingness scenarios).",
            ["results/metrics_fusion.csv", "results/missing_modality.csv"],
            "Figure R5. Effect of quality signals: PR-AUC difference (F-GATE minus F-GATE-NOQ) over TEST and missing-modality scenarios.",
        ),
        CaptionItem(
            "Table R3",
            "tables/R3_ablation_deltas_gate_vs_noq.png (+ .csv)",
            "Overall TEST deltas for F-GATE minus F-GATE-NOQ on PR-AUC, AUROC, Accuracy, and FAR.",
            ["results/metrics_fusion.csv"],
            "Table R3. Overall TEST deltas for F-GATE relative to F-GATE-NOQ at threshold 0.5.",
        ),
        CaptionItem(
            "Figure R6a",
            "figures/R6a_stress_audio_snr.png",
            "F-GATE PR-AUC trend over audio SNR with unimodal TEST baselines as references.",
            ["results/stress_audio_snr.csv", "results/metrics_unimodal.csv"],
            "Figure R6a. Audio SNR stress sweep (TEST): F-GATE trend with unimodal TEST baselines shown as dashed references.",
        ),
        CaptionItem(
            "Figure R6b",
            "figures/R6b_stress_vision_corruption.png",
            "F-GATE PR-AUC trend over vision corruption severity with unimodal TEST baseline references.",
            ["results/stress_vision_corruptions.csv", "results/metrics_unimodal.csv"],
            "Figure R6b. Vision corruption stress sweep (TEST): F-GATE trend with unimodal TEST baselines shown as dashed references.",
        ),
        CaptionItem(
            "Figure R6c",
            "figures/R6c_stress_radar_clutter.png",
            "F-GATE PR-AUC trend over radar clutter level with unimodal TEST baseline references.",
            ["results/stress_radar_clutter.csv", "results/metrics_unimodal.csv"],
            "Figure R6c. Radar clutter stress sweep (TEST): F-GATE trend with unimodal TEST baselines shown as dashed references.",
        ),
        CaptionItem(
            "Table R4",
            "tables/R4_dataset_split_counts.png (+ .csv)",
            "Train/val/test episode counts and class balance.",
            ["data/processed/episodes/episodes.csv"],
            "Table R4. Dataset split composition and class balance for train, validation, and held-out TEST episodes.",
        ),
        CaptionItem(
            "Figure R7",
            "figures/R7_latency_summary.png",
            "Fusion latency summary callout (reused curated artifact when available).",
            ["final_isef_package/figures/latency_breakdown.png", "results/latency_summary.json"],
            "Figure R7. Runtime summary of fusion methods (mean latency, with P95 context from the measured latency report).",
        ),
    ]
    build_readme(captions)

    # Save flat text captions for quick copy-paste.
    quick = []
    for c in captions:
        quick.append(f"{c.asset_id}: {c.caption}")
    (OUT / "CAPTIONS_QUICKCOPY.txt").write_text("\n".join(quick) + "\n", encoding="utf-8")

    print(f"Built results section assets at: {OUT}")


if __name__ == "__main__":
    main()
