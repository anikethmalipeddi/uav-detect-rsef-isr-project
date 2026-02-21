from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import pandas as pd


def _normalized_id_set(df: pd.DataFrame, split_name: str, key_col: str) -> set[str]:
    vals = (
        df.loc[df["split"] == split_name, key_col]
        .dropna()
        .astype(str)
        .map(str.strip)
        .tolist()
    )
    bad = {"", "nan", "none", "null"}
    return {v for v in vals if v.lower() not in bad}


def overlap_ids(df: pd.DataFrame, key_col: str, split_a: str, split_b: str) -> list[str]:
    a = _normalized_id_set(df, split_a, key_col)
    b = _normalized_id_set(df, split_b, key_col)
    return sorted(a & b)


def check_split_integrity(df: pd.DataFrame) -> tuple[bool, dict]:
    required_cols = [
        "episode_id",
        "split",
        "src_video_id",
        "src_audio_id",
        "src_radar_id",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return False, {"error": f"Missing required columns: {missing}"}

    problems: list[str] = []
    n = len(df)
    n_unique_episode_id = int(df["episode_id"].astype(str).nunique())
    if n_unique_episode_id != n:
        problems.append(f"episode_id not unique: unique={n_unique_episode_id}, rows={n}")

    splits = sorted(df["split"].astype(str).unique().tolist())
    valid_splits = {"train", "val", "test"}
    bad_splits = [s for s in splits if s not in valid_splits]
    if bad_splits:
        problems.append(f"Invalid split labels found: {bad_splits}")

    split_pairs = [p for p in combinations(["train", "val", "test"], 2)]
    overlap = {}
    overlap_examples = {}
    for col in ["src_video_id", "src_audio_id", "src_radar_id"]:
        overlap[col] = {}
        overlap_examples[col] = {}
        for a, b in split_pairs:
            k = f"{a}_vs_{b}"
            ids = overlap_ids(df, key_col=col, split_a=a, split_b=b)
            c = int(len(ids))
            overlap[col][k] = c
            overlap_examples[col][k] = ids[:10]
            if c > 0:
                sample = ", ".join(ids[:5])
                problems.append(f"Leakage in {col}: {k} overlap={c}; examples=[{sample}]")

    result = {
        "rows": int(n),
        "unique_episode_id": int(n_unique_episode_id),
        "split_counts": df["split"].astype(str).value_counts().to_dict(),
        "overlap_counts": overlap,
        "overlap_examples": overlap_examples,
        "problems": problems,
        "ok": len(problems) == 0,
    }
    return result["ok"], result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, default=Path("data/processed/episodes/episodes.csv"))
    parser.add_argument("--out-txt", type=Path, default=Path("results/split_integrity_report.txt"))
    parser.add_argument("--out-json", type=Path, default=Path("results/split_integrity_report.json"))
    parser.add_argument("--allow-fail", action="store_true")
    args = parser.parse_args()

    if not args.episodes.exists():
        raise FileNotFoundError(f"Episodes file not found: {args.episodes}")
    df = pd.read_csv(args.episodes, dtype={"episode_id": str})
    ok, result = check_split_integrity(df)

    args.out_txt.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        "Split Integrity Report",
        f"episodes={args.episodes}",
        f"rows={result['rows']}",
        f"unique_episode_id={result['unique_episode_id']}",
        f"split_counts={result['split_counts']}",
    ]
    for col, d in result["overlap_counts"].items():
        lines.append(f"{col}_overlap={d}")
        lines.append(f"{col}_overlap_examples={result['overlap_examples'][col]}")
    if ok:
        lines.append("SPLIT_INTEGRITY=PASS")
    else:
        lines.append("SPLIT_INTEGRITY=FAIL")
        lines.append("Problems:")
        for p in result["problems"]:
            lines.append(f"- {p}")
    args.out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.out_txt.read_text(encoding="utf-8"))

    if not ok and not args.allow_fail:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
