from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _run_git(args: list[str]) -> str | None:
    try:
        out = subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL, text=True).strip()
        return out
    except Exception:  # noqa: BLE001
        return None


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def _safe_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    for enc in ("utf-8", "utf-8-sig"):
        try:
            return json.loads(path.read_text(encoding=enc))
        except Exception:  # noqa: BLE001
            continue
    return None


def _csv_row_count(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        import pandas as pd

        return int(len(pd.read_csv(path)))
    except Exception:  # noqa: BLE001
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/run_metadata.json"))
    parser.add_argument("--episodes", type=Path, default=Path("data/processed/episodes/episodes.csv"))
    parser.add_argument("--features", type=Path, default=Path("data/processed/episodes/features_cache.npz"))
    parser.add_argument("--tsms-state", type=Path, default=Path("results/tsms_state.json"))
    parser.add_argument("--dronedetect-state", type=Path, default=Path("results/dronedetect_state.json"))
    parser.add_argument("--label-stats", type=Path, default=Path("results/label_stats.json"))
    parser.add_argument("--fusion-best", type=Path, default=Path("results/fusion_best_config.json"))
    parser.add_argument("--fusion-tuning", type=Path, default=Path("results/fusion_tuning.csv"))
    parser.add_argument("--fallback-warnings", type=Path, default=Path("results/fallback_warnings.txt"))
    parser.add_argument("--config-snapshot", type=Path, default=Path("results/run_config_snapshot.json"))
    parser.add_argument("--config-used", type=Path, default=Path("results/config_used.yaml"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch_info: dict[str, object] = {"available": False}
    try:
        import torch

        torch_info = {
            "available": True,
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": getattr(torch.version, "cuda", None),
            "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            if torch.cuda.is_available()
            else [],
        }
    except Exception as e:  # noqa: BLE001
        torch_info = {"available": False, "error": str(e)}

    warnings: list[str] = []
    if args.fallback_warnings.exists():
        warnings.extend([line.strip() for line in args.fallback_warnings.read_text(encoding="utf-8").splitlines() if line.strip()])

    tsms_state = _safe_json(args.tsms_state)
    dronedetect_state = _safe_json(args.dronedetect_state)
    tsms_full = bool(isinstance(tsms_state, dict) and tsms_state.get("complete", False))
    dronedetect_full = bool(isinstance(dronedetect_state, dict) and dronedetect_state.get("complete", False))

    if not tsms_full:
        warnings.append("TSMS verification indicates incomplete dataset.")
    if not dronedetect_full:
        warnings.append("DroneDetectionThesis verification indicates incomplete dataset.")
    if args.config_used.exists() is False:
        warnings.append("Missing results/config_used.yaml for this run.")

    fusion_best = _safe_json(args.fusion_best)
    tuning_rows = _csv_row_count(args.fusion_tuning)

    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "machine": {
            "node": platform.node(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": sys.version,
        },
        "torch": torch_info,
        "git": {
            "commit": _run_git(["rev-parse", "HEAD"]),
            "branch": _run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
            "dirty": bool(_run_git(["status", "--porcelain"])),
        },
        "seed": args.seed,
        "paths": {
            "episodes": str(args.episodes),
            "features": str(args.features),
            "tsms_state": str(args.tsms_state),
            "dronedetect_state": str(args.dronedetect_state),
            "label_stats": str(args.label_stats),
            "fusion_best": str(args.fusion_best),
            "fusion_tuning": str(args.fusion_tuning),
            "config_used": str(args.config_used),
        },
        "dataset_state": {
            "raw_size_gb": round(_dir_size_bytes(Path("data/raw")) / (1024**3), 3),
            "processed_size_gb": round(_dir_size_bytes(Path("data/processed")) / (1024**3), 3),
            "episodes_exists": bool(args.episodes.exists()),
            "features_exists": bool(args.features.exists()),
            "tsms_state": tsms_state,
            "dronedetect_state": dronedetect_state,
            "label_stats": _safe_json(args.label_stats),
            "full_dataset_verified": {
                "tsms": tsms_full,
                "dronedetect": dronedetect_full,
                "all_modalities_full": bool(tsms_full and dronedetect_full),
            },
        },
        "fusion": {
            "best_config": fusion_best,
            "tuning_csv_rows": tuning_rows,
            "tuning_csv_path": str(args.fusion_tuning),
        },
        "config_snapshot": _safe_json(args.config_snapshot),
        "final_run_eligible": bool(tsms_full and dronedetect_full and len(warnings) == 0),
        "warnings": warnings,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
