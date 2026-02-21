from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.fusion_gate import build_fusion_features


PLAN_COLUMNS = [
    "episode_id",
    "split",
    "label_drone",
    "video_path",
    "t0_sec",
    "t1_sec",
    "audio_path",
    "radar_path",
    "vision_condition",
    "audio_snr_db",
    "radar_clutter_level",
    "miss_vis",
    "miss_aud",
    "miss_rad",
    "source_video_label",
    "source_audio_label",
    "source_radar_label",
]
SOURCE_ID_COLUMNS = ["src_video_id", "src_audio_id", "src_radar_id"]
FEATURE_KEYS = ["episode_id", "y", "logit_vis", "logit_aud", "logit_rad", "q_vis", "q_aud", "q_rad", "mask"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, default=Path("data/processed/episodes/episodes.csv"))
    parser.add_argument("--features", type=Path, default=Path("data/processed/episodes/features_cache.npz"))
    parser.add_argument("--out-json", type=Path, default=Path("results/sanity_checks.json"))
    parser.add_argument("--out-txt", type=Path, default=Path("results/sanity_checks.txt"))
    args = parser.parse_args()

    problems: list[str] = []
    notes: list[str] = []

    if not args.episodes.exists():
        raise FileNotFoundError(f"Missing episodes CSV: {args.episodes}")
    ep = pd.read_csv(args.episodes, dtype={"episode_id": str})

    cols = ep.columns.tolist()
    if cols[: len(PLAN_COLUMNS)] != PLAN_COLUMNS:
        problems.append(
            "episodes.csv first 17 columns do not match required order. "
            f"Found={cols[:len(PLAN_COLUMNS)]}"
        )
    missing_src = [c for c in SOURCE_ID_COLUMNS if c not in cols]
    if missing_src:
        problems.append(f"episodes.csv missing source-id columns: {missing_src}")

    if ep["episode_id"].nunique() != len(ep):
        problems.append("episode_id values are not unique.")

    bad_splits = sorted(set(ep["split"].astype(str)) - {"train", "val", "test"})
    if bad_splits:
        problems.append(f"Invalid split labels found: {bad_splits}")

    for c in ["video_path", "audio_path", "radar_path"]:
        if c not in ep.columns:
            problems.append(f"Missing path column: {c}")
            continue
        exists = ep[c].map(lambda x: Path(str(x)).exists())
        if not bool(exists.all()):
            problems.append(f"Path existence check failed for {c}: missing={(~exists).sum()}")

    if not args.features.exists():
        raise FileNotFoundError(f"Missing features cache: {args.features}")
    z = np.load(args.features, allow_pickle=False)
    missing_keys = [k for k in FEATURE_KEYS if k not in z.files]
    if missing_keys:
        problems.append(f"features_cache missing arrays: {missing_keys}")
    else:
        n = int(z["y"].shape[0])
        for k in FEATURE_KEYS:
            if z[k].shape[0] != n:
                problems.append(f"features_cache array {k} first dim mismatch: {z[k].shape[0]} vs {n}")
        if z["q_vis"].shape[1:] != (5,):
            problems.append(f"q_vis shape expected (N,5), got {z['q_vis'].shape}")
        if z["q_aud"].shape[1:] != (4,):
            problems.append(f"q_aud shape expected (N,4), got {z['q_aud'].shape}")
        if z["q_rad"].shape[1:] != (3,):
            problems.append(f"q_rad shape expected (N,3), got {z['q_rad'].shape}")
        if z["mask"].shape[1:] != (3,):
            problems.append(f"mask shape expected (N,3), got {z['mask'].shape}")

        try:
            x = build_fusion_features(
                z["logit_vis"].astype(np.float32),
                z["logit_aud"].astype(np.float32),
                z["logit_rad"].astype(np.float32),
                z["q_vis"].astype(np.float32),
                z["q_aud"].astype(np.float32),
                z["q_rad"].astype(np.float32),
                z["mask"].astype(np.float32),
            )
            if x.shape[1] != 24:
                problems.append(f"Fusion feature dimension mismatch: expected 24, got {x.shape[1]}")
        except Exception as e:  # noqa: BLE001
            problems.append(f"Failed to build fusion input: {e}")

        ep_ids = ep["episode_id"].astype(str).tolist()
        feat_ids = z["episode_id"].astype(str).tolist()
        if len(ep_ids) != len(feat_ids):
            problems.append(f"Episode count mismatch episodes.csv={len(ep_ids)} features_cache={len(feat_ids)}")
        elif ep_ids != feat_ids:
            notes.append("Episode order mismatch between episodes.csv and features_cache.npz (IDs differ by order).")

    ok = len(problems) == 0
    payload = {"ok": ok, "problems": problems, "notes": notes}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_txt.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = ["Sanity Checks", f"episodes={args.episodes}", f"features={args.features}"]
    if ok:
        lines.append("SANITY=PASS")
    else:
        lines.append("SANITY=FAIL")
        lines.append("Problems:")
        for p in problems:
            lines.append(f"- {p}")
    if notes:
        lines.append("Notes:")
        for n in notes:
            lines.append(f"- {n}")
    args.out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.out_txt.read_text(encoding="utf-8"))

    if not ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
