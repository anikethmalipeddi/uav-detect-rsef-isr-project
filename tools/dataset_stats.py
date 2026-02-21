from __future__ import annotations

import argparse
import json
from pathlib import Path


def folder_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def count_ext(path: Path, ext: str) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.rglob(f"*{ext}"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/raw")
    parser.add_argument("--out", default="results/dataset_stats.json")
    args = parser.parse_args()

    root = Path(args.root)
    thesis = root / "dronedetect_thesis"
    tsms = root / "tsms_drone"
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "dronedetect_thesis": {
            "exists": thesis.exists(),
            "size_gb": round(folder_size_bytes(thesis) / (1024**3), 3),
            "videos_mp4": count_ext(thesis, ".mp4"),
            "videos_avi": count_ext(thesis, ".avi"),
            "labels_mat": count_ext(thesis, ".mat"),
            "audio_wav": count_ext(thesis, ".wav"),
        },
        "tsms_drone": {
            "exists": tsms.exists(),
            "size_gb": round(folder_size_bytes(tsms) / (1024**3), 3),
            "npy": count_ext(tsms, ".npy"),
            "mat": count_ext(tsms, ".mat"),
            "png": count_ext(tsms, ".png"),
            "jpg": count_ext(tsms, ".jpg"),
        },
    }

    out_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
