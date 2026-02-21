from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)


@dataclass
class BinaryMetrics:
    pr_auc: float
    roc_auc: float
    confusion_matrix: list[list[int]]


def binary_metrics(y_true: np.ndarray, p_score: np.ndarray, thr: float = 0.5) -> BinaryMetrics:
    y_true = np.asarray(y_true).astype(int)
    p_score = np.asarray(p_score).astype(float)
    y_pred = (p_score >= thr).astype(int)
    return BinaryMetrics(
        pr_auc=float(average_precision_score(y_true, p_score)),
        roc_auc=float(roc_auc_score(y_true, p_score)),
        confusion_matrix=confusion_matrix(y_true, y_pred).tolist(),
    )


def metrics_row(run_id: str, split: str, y_true: np.ndarray, p_score: np.ndarray, thr: float = 0.5) -> dict:
    m = binary_metrics(y_true, p_score, thr=thr)
    return {
        "run_id": run_id,
        "split": split,
        "pr_auc": m.pr_auc,
        "roc_auc": m.roc_auc,
        "tn": m.confusion_matrix[0][0],
        "fp": m.confusion_matrix[0][1],
        "fn": m.confusion_matrix[1][0],
        "tp": m.confusion_matrix[1][1],
    }


def plot_pr_curve(y_true: np.ndarray, p_score: np.ndarray, out_path: str | Path, title: str) -> None:
    precision, recall, _ = precision_recall_curve(y_true, p_score)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def write_rows_csv(rows: Iterable[dict], out_path: str | Path) -> None:
    import pandas as pd

    df = pd.DataFrame(list(rows))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
