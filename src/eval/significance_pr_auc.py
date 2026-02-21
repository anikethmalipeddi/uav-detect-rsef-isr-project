from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def bootstrap_pr_auc_diff(
    y_true: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float, float]:
    rng = np.random.default_rng(seed)
    n = y_true.shape[0]
    diffs = np.zeros((n_boot,), dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ya = y_true[idx]
        diffs[i] = average_precision_score(ya, p_a[idx]) - average_precision_score(ya, p_b[idx])
    mean = float(diffs.mean())
    lo = float(np.percentile(diffs, 2.5))
    hi = float(np.percentile(diffs, 97.5))
    frac_le0 = float(np.mean(diffs <= 0.0))
    return mean, lo, hi, frac_le0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=Path, default=Path("results/preds_fusion.csv"))
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("results/significance_pr_auc.txt"))
    args = parser.parse_args()

    if not args.pred.exists():
        raise FileNotFoundError(f"Predictions file not found: {args.pred}")

    df = pd.read_csv(args.pred)
    required = {"split", "y_true", "p_avg", "p_static", "p_gate"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required prediction columns: {sorted(missing)}")

    sdf = df[df["split"].astype(str) == args.split].copy()
    if sdf.empty:
        raise ValueError(f"No rows found for split='{args.split}' in {args.pred}")

    y = sdf["y_true"].to_numpy(dtype=np.int64)
    p_gate = sdf["p_gate"].to_numpy(dtype=np.float64)
    p_avg = sdf["p_avg"].to_numpy(dtype=np.float64)
    p_static = sdf["p_static"].to_numpy(dtype=np.float64)

    g_vs_avg = bootstrap_pr_auc_diff(y, p_gate, p_avg, args.n_bootstrap, args.seed)
    g_vs_static = bootstrap_pr_auc_diff(y, p_gate, p_static, args.n_bootstrap, args.seed + 1)

    lines = [
        "Bootstrap PR-AUC Significance",
        f"predictions={args.pred}",
        f"split={args.split}",
        f"n_bootstrap={args.n_bootstrap}",
        "",
        "Comparison: F-GATE - F-AVG",
        f"mean_diff={g_vs_avg[0]:.6f}",
        f"ci95=({g_vs_avg[1]:.6f}, {g_vs_avg[2]:.6f})",
        f"fraction_le_zero={g_vs_avg[3]:.6f}",
        "",
        "Comparison: F-GATE - F-STATIC",
        f"mean_diff={g_vs_static[0]:.6f}",
        f"ci95=({g_vs_static[1]:.6f}, {g_vs_static[2]:.6f})",
        f"fraction_le_zero={g_vs_static[3]:.6f}",
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
