from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, default=Path("results/metrics_fusion.csv"))
    parser.add_argument("--out-txt", type=Path, default=Path("results/fusion_advantage_check.txt"))
    parser.add_argument("--metric", type=str, default="pr_auc")
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()

    if not args.metrics.exists():
        raise FileNotFoundError(f"Missing fusion metrics CSV: {args.metrics}")
    df = pd.read_csv(args.metrics)

    need = {"F-AVG", "F-STATIC", "F-GATE"}
    have = set(df["run_id"].astype(str))
    missing = sorted(need - have)
    if missing:
        raise ValueError(f"Missing required run_id rows in metrics CSV: {missing}")
    if args.metric not in df.columns:
        raise ValueError(f"Metric column not found: {args.metric}")

    sdf = df[df["split"].astype(str) == args.split]
    if sdf.empty:
        raise ValueError(f"No rows found for split='{args.split}'")

    avg = float(sdf.loc[sdf["run_id"] == "F-AVG", args.metric].iloc[0])
    static = float(sdf.loc[sdf["run_id"] == "F-STATIC", args.metric].iloc[0])
    gate = float(sdf.loc[sdf["run_id"] == "F-GATE", args.metric].iloc[0])
    ok = (gate > avg) and (gate > static)

    lines = [
        "Fusion Advantage Check",
        f"metrics={args.metrics}",
        f"split={args.split}",
        f"metric={args.metric}",
        f"F-AVG={avg:.9f}",
        f"F-STATIC={static:.9f}",
        f"F-GATE={gate:.9f}",
        f"GATE_MINUS_AVG={(gate - avg):.9f}",
        f"GATE_MINUS_STATIC={(gate - static):.9f}",
        f"FUSION_ADVANTAGE={'PASS' if ok else 'FAIL'}",
    ]
    args.out_txt.parent.mkdir(parents=True, exist_ok=True)
    args.out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.out_txt.read_text(encoding="utf-8"))

    if not ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
