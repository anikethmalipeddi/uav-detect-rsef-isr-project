from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance


ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "final_isef_package"
APP = ROOT / "vercel_poster_demo"
DATA_DIR = APP / "data"
ASSETS_DIR = APP / "assets"
FRAMES_DIR = ASSETS_DIR / "frames"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: object) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def df_to_records(src: Path) -> list[dict]:
    return pd.read_csv(src).to_dict(orient="records")


def build_frame_sequence(base_frame: Path, frame_count: int) -> list[str]:
    if FRAMES_DIR.exists():
        shutil.rmtree(FRAMES_DIR)
    ensure_dir(FRAMES_DIR)

    src = Image.open(base_frame).convert("RGB")
    width, height = src.size
    out_files: list[str] = []
    for i in range(frame_count):
        t = i / max(1, frame_count - 1)
        dx = int(round(10.0 * math.sin(t * 2.0 * math.pi)))
        dy = int(round(6.0 * math.cos(t * 2.0 * math.pi)))
        shifted = src.transform(
            src.size,
            Image.AFFINE,
            (1, 0, dx, 0, 1, dy),
            resample=Image.BICUBIC,
            fillcolor=(18, 24, 33),
        )
        brightness = 0.95 + 0.08 * math.sin(t * 2.0 * math.pi + 0.5)
        shifted = ImageEnhance.Brightness(shifted).enhance(brightness)

        draw = ImageDraw.Draw(shifted)
        cx = int(width * (0.32 + 0.18 * math.sin(t * 2.0 * math.pi)))
        cy = int(height * (0.42 + 0.10 * math.cos(t * 2.0 * math.pi)))
        bw = max(26, width // 9)
        bh = max(16, height // 12)
        draw.rectangle((cx - bw, cy - bh, cx + bw, cy + bh), outline=(0, 255, 128), width=3)
        draw.text((cx - bw, max(4, cy - bh - 18)), "drone", fill=(255, 245, 120))

        name = f"frame_{i:03d}.jpg"
        shifted.save(FRAMES_DIR / name, format="JPEG", quality=68, optimize=True, progressive=True)
        out_files.append(name)
    return out_files


def main() -> int:
    ensure_dir(DATA_DIR)
    ensure_dir(ASSETS_DIR)

    # Scenario/frame-level predictions with modality + gate scores.
    pred = pd.read_csv(PKG / "demo" / "assets" / "sample_predictions.csv")
    pred = pred.copy()
    pred["scenario"] = "clean"

    demo_outputs = ROOT / "demo_outputs" / "predictions_per_frame.csv"
    if demo_outputs.exists():
        pred = pd.read_csv(demo_outputs)

    write_json(DATA_DIR / "predictions.json", pred.to_dict(orient="records"))

    # Core numeric artifacts.
    write_json(DATA_DIR / "metrics_fusion.json", df_to_records(PKG / "metrics" / "metrics_fusion.csv"))
    write_json(DATA_DIR / "metrics_unimodal.json", df_to_records(PKG / "metrics" / "metrics_unimodal.csv"))
    write_json(DATA_DIR / "missing_modality.json", df_to_records(PKG / "tables" / "missing_modality.csv"))
    write_json(DATA_DIR / "stress_audio_snr.json", df_to_records(PKG / "tables" / "stress_audio_snr.csv"))
    write_json(DATA_DIR / "stress_vision_corruptions.json", df_to_records(PKG / "tables" / "stress_vision_corruptions.csv"))
    write_json(DATA_DIR / "stress_radar_clutter.json", df_to_records(PKG / "tables" / "stress_radar_clutter.csv"))
    latency = json.loads((PKG / "metrics" / "latency_summary.json").read_text(encoding="utf-8"))
    write_json(DATA_DIR / "latency_summary.json", latency)
    audio_features = pd.read_csv(PKG / "demo" / "assets" / "sample_audio_features.csv")
    write_json(DATA_DIR / "audio_features.json", audio_features.to_dict(orient="records"))
    radar_features = pd.read_csv(PKG / "demo" / "assets" / "sample_radar_features.csv")
    write_json(DATA_DIR / "radar_features.json", radar_features.to_dict(orient="records"))

    # Small visual assets.
    sample_frame_dst = ASSETS_DIR / "sample_frame.png"
    for src, dst in [
        (PKG / "demo" / "assets" / "sample_frame.png", sample_frame_dst),
        (PKG / "figures" / "qualitative_overlays_grid.png", ASSETS_DIR / "qualitative_overlays_grid.png"),
        (PKG / "figures" / "ablation_bars.png", ASSETS_DIR / "ablation_bars.png"),
        (PKG / "figures" / "stress_audio_snr_curve.png", ASSETS_DIR / "stress_audio_snr_curve.png"),
    ]:
        if src.exists():
            shutil.copy2(src, dst)

    frame_count = int(pred["frame_idx"].max() + 1) if "frame_idx" in pred.columns and len(pred) else 120
    frame_count = max(30, min(180, frame_count))
    frame_files = build_frame_sequence(sample_frame_dst, frame_count)

    manifest = {
        "source": "final_isef_package",
        "data_files": sorted([p.name for p in DATA_DIR.glob("*.json")]),
        "asset_files": sorted([p.name for p in ASSETS_DIR.iterdir() if p.is_file()]),
        "frame_count": frame_count,
        "frame_files": frame_files,
        "frames_path": "assets/frames",
    }
    write_json(DATA_DIR / "manifest.json", manifest)
    print("Built vercel poster assets in vercel_poster_demo/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
