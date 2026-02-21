from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple


REQUIRED_PATHS = [
    "demo_app/app.py",
    "demo_app/demo_utils.py",
    "demo_app/model_adapters/full_mode.py",
    "demo_app/model_adapters/poster_mode.py",
    "demo_assets/sample_video.mp4",
    "demo_assets/sample_audio.wav",
    "demo_assets/sample_radar.csv",
    "demo_assets/sample_metadata.json",
    "demo_outputs/predictions_per_frame.csv",
    "demo_outputs/gating_weights_per_frame.csv",
    "demo_outputs/bboxes_per_frame.csv",
    "scripts/run_demo.ps1",
    "scripts/run_demo.sh",
    "scripts/verify_demo.ps1",
    "scripts/verify_demo.sh",
    "docs/README_DEMO.md",
    "environment_demo.yml",
    "requirements_demo.txt",
]

PRED_REQUIRED_COLUMNS = {
    "scenario",
    "frame_idx",
    "t_sec",
    "p_vis",
    "p_aud",
    "p_rad",
    "p_gate",
    "w_vis",
    "w_aud",
    "w_rad",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def verify_paths(root: Path) -> List[str]:
    missing = []
    for rel in REQUIRED_PATHS:
        if not (root / rel).exists():
            missing.append(rel)
    return missing


def verify_predictions(root: Path) -> Tuple[bool, str]:
    pred_path = root / "demo_outputs" / "predictions_per_frame.csv"
    with pred_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        cols = set(reader.fieldnames or [])
        missing = sorted(PRED_REQUIRED_COLUMNS - cols)
        if missing:
            return False, f"predictions_per_frame.csv missing columns: {missing}"
        rows = list(reader)
    if len(rows) < 30:
        return False, "predictions_per_frame.csv has too few rows (<30)"
    scenarios = {row.get("scenario", "") for row in rows}
    needed = {"clean", "audio_degraded", "vision_degraded", "radar_clutter", "missing_audio"}
    if not needed.issubset(scenarios):
        return False, "predictions_per_frame.csv missing required scripted scenarios"
    return True, f"rows={len(rows)}, scenarios={sorted(scenarios)}"


def verify_metadata(root: Path) -> Tuple[bool, str]:
    meta_path = root / "demo_assets" / "sample_metadata.json"
    with meta_path.open("r", encoding="utf-8") as handle:
        payload: Dict[str, object] = json.load(handle)
    required = ["source_video", "source_audio", "fps", "duration_sec", "frame_step"]
    missing = [k for k in required if k not in payload]
    if missing:
        return False, f"sample_metadata.json missing keys: {missing}"
    return True, "sample_metadata.json contains required keys"


def print_commands() -> None:
    print("\nWindows:")
    print("  conda env create -f environment_demo.yml")
    print("  conda activate uav-demo")
    print("  powershell -ExecutionPolicy Bypass -File scripts/run_demo.ps1")
    print("\nmacOS M2 (conda):")
    print("  conda env create -f environment_demo.yml")
    print("  conda activate uav-demo")
    print("  bash scripts/run_demo.sh")
    print("\nmacOS M2 (pip):")
    print("  python3 -m venv .venv")
    print("  source .venv/bin/activate")
    print("  pip install -r requirements_demo.txt")
    print("  streamlit run demo_app/app.py")


def main() -> int:
    root = repo_root()
    failures: List[str] = []

    missing_paths = verify_paths(root)
    if missing_paths:
        failures.append("Missing required demo files: " + ", ".join(missing_paths))

    if not missing_paths:
        ok, detail = verify_predictions(root)
        if ok:
            print(f"PASS predictions: {detail}")
        else:
            failures.append(detail)

        ok, detail = verify_metadata(root)
        if ok:
            print(f"PASS metadata: {detail}")
        else:
            failures.append(detail)

    if (root / "models").exists():
        print("INFO full mode: models/ exists; app will attempt full mode if required files are present.")
    else:
        print("INFO full mode: models/ not present, Poster Mode fallback expected.")

    if failures:
        print("FAIL demo verification:")
        for msg in failures:
            print(f"  - {msg}")
        print_commands()
        return 1

    print("PASS demo verification.")
    print_commands()
    return 0


if __name__ == "__main__":
    sys.exit(main())

