from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def _safe_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    for enc in ("utf-8", "utf-8-sig"):
        try:
            return json.loads(path.read_text(encoding=enc))
        except Exception:  # noqa: BLE001
            continue
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-snapshot", type=Path, default=Path("results/run_config_snapshot.json"))
    parser.add_argument("--fusion-best", type=Path, default=Path("results/fusion_best_config.json"))
    parser.add_argument("--tsms-state", type=Path, default=Path("results/tsms_state.json"))
    parser.add_argument("--dronedetect-state", type=Path, default=Path("results/dronedetect_state.json"))
    parser.add_argument("--out", type=Path, default=Path("results/config_used.yaml"))
    args = parser.parse_args()

    snap = _safe_json(args.config_snapshot)
    fusion_best = _safe_json(args.fusion_best)
    tsms_state = _safe_json(args.tsms_state)
    dronedetect_state = _safe_json(args.dronedetect_state)
    payload = {
        "run_config_snapshot": snap,
        "fusion_selection": fusion_best,
        "dataset_verification": {
            "tsms_complete": bool(isinstance(tsms_state, dict) and tsms_state.get("complete", False)),
            "dronedetect_complete": bool(
                isinstance(dronedetect_state, dict) and dronedetect_state.get("complete", False)
            ),
            "tsms_state_path": str(args.tsms_state),
            "dronedetect_state_path": str(args.dronedetect_state),
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
