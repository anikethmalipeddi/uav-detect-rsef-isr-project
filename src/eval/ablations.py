from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.eval.eval_fusion import main as eval_fusion_main


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, default=Path("data/processed/episodes/episodes.csv"))
    parser.add_argument("--features", type=Path, default=Path("data/processed/episodes/features_cache.npz"))
    args = parser.parse_args()

    # Reuse fusion evaluation to generate the four required ablation rows.
    import sys

    sys.argv = [
        "eval_fusion.py",
        "--episodes",
        str(args.episodes),
        "--features",
        str(args.features),
        "--out",
        "results/metrics_fusion.csv",
    ]
    eval_fusion_main()

    df = pd.read_csv("results/metrics_fusion.csv")
    df.to_csv("results/ablation_table.csv", index=False)
    print("Wrote results/ablation_table.csv")


if __name__ == "__main__":
    main()
