from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".mpeg", ".mpg"}


def vision_quality_features(frame: np.ndarray) -> dict[str, float]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    contrast = float(gray.std())
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return {"brightness_mean": brightness, "contrast_std": contrast, "blur_var_laplacian": blur}


def extract_frames(video_path: Path, out_dir: Path, fps_out: int = 10, jpg_quality: int = 90) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0

    fps_in = cap.get(cv2.CAP_PROP_FPS)
    stride = max(1, int(round((fps_in if fps_in > 0 else fps_out) / float(fps_out))))

    idx = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            out_path = out_dir / f"frame_{idx:06d}.jpg"
            cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
            saved += 1
        idx += 1
    cap.release()
    return saved


def build_manifest(input_root: Path, output_root: Path, fps: int, max_videos: int = 0) -> list[dict]:
    rows = []
    videos = [p for p in input_root.rglob("*") if p.suffix.lower() in VIDEO_EXT]
    if max_videos > 0:
        videos = videos[:max_videos]
    for vid in tqdm(videos, desc="Extracting frames"):
        rel = vid.relative_to(input_root)
        out_dir = output_root / rel.with_suffix("")
        saved = extract_frames(vid, out_dir, fps_out=fps)
        rows.append({"video_path": str(vid), "frames_out_dir": str(out_dir), "frames_saved": saved})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/vision_frames/manifest.csv"))
    args = parser.parse_args()

    rows = build_manifest(args.input, args.output, fps=args.fps, max_videos=args.max_videos)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    pd.DataFrame(rows).to_csv(args.manifest, index=False)
    print(f"Wrote {len(rows)} rows to {args.manifest}")


if __name__ == "__main__":
    main()
