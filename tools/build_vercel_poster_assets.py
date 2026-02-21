from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "final_isef_package"
APP = ROOT / "vercel_poster_demo"
DATA_DIR = APP / "data"
ASSETS_DIR = APP / "assets"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: object) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def df_to_records(src: Path) -> list[dict]:
    return pd.read_csv(src).to_dict(orient="records")


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

    # Small visual assets.
    for src, dst in [
        (PKG / "demo" / "assets" / "sample_frame.png", ASSETS_DIR / "sample_frame.png"),
        (PKG / "figures" / "qualitative_overlays_grid.png", ASSETS_DIR / "qualitative_overlays_grid.png"),
        (PKG / "figures" / "ablation_bars.png", ASSETS_DIR / "ablation_bars.png"),
        (PKG / "figures" / "stress_audio_snr_curve.png", ASSETS_DIR / "stress_audio_snr_curve.png"),
    ]:
        if src.exists():
            shutil.copy2(src, dst)

    manifest = {
        "source": "final_isef_package",
        "data_files": sorted([p.name for p in DATA_DIR.glob("*.json")]),
        "asset_files": sorted([p.name for p in ASSETS_DIR.iterdir() if p.is_file()]),
    }
    write_json(DATA_DIR / "manifest.json", manifest)
    print("Built vercel poster assets in vercel_poster_demo/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

