from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


REQUIRED_DEMO_PATHS: Tuple[str, ...] = (
    "demo_assets/sample_video.mp4",
    "demo_assets/sample_audio.wav",
    "demo_assets/sample_radar.csv",
    "demo_assets/sample_metadata.json",
    "demo_outputs/predictions_per_frame.csv",
    "demo_outputs/gating_weights_per_frame.csv",
    "demo_outputs/bboxes_per_frame.csv",
    "final_isef_package/tables/metrics_unimodal.csv",
    "final_isef_package/tables/metrics_fusion.csv",
    "final_isef_package/tables/stress_audio_snr.csv",
    "final_isef_package/tables/stress_vision_corruptions.csv",
    "final_isef_package/tables/stress_radar_clutter.csv",
    "final_isef_package/tables/latency_summary.json",
)

PREDICTION_COLUMNS: Tuple[str, ...] = (
    "scenario",
    "frame_idx",
    "t_sec",
    "y_true",
    "p_vis",
    "p_aud",
    "p_rad",
    "p_avg",
    "p_static",
    "p_gate",
    "p_gate_noq",
    "w_vis",
    "w_aud",
    "w_rad",
    "vision_condition",
    "audio_snr_db",
    "radar_clutter_level",
    "miss_vis",
    "miss_aud",
    "miss_rad",
)


@dataclass(frozen=True)
class ModeResolution:
    mode: str
    badge_text: str
    details: str


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def read_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: Dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def poster_mode_missing_paths(root: Path) -> List[str]:
    missing: List[str] = []
    for rel in REQUIRED_DEMO_PATHS:
        if not (root / rel).exists():
            missing.append(rel)
    return missing


def resolve_mode(preferred_mode: str, root: Path, models_dir: Path) -> ModeResolution:
    missing_paths = poster_mode_missing_paths(root)
    if missing_paths:
        return ModeResolution(
            mode="error",
            badge_text="Poster Mode: Missing committed assets",
            details="Missing files: " + ", ".join(missing_paths),
        )

    normalized = preferred_mode.strip().lower()
    if normalized == "full":
        try:
            from model_adapters.full_mode import FullModeAdapter
        except ModuleNotFoundError:
            from demo_app.model_adapters.full_mode import FullModeAdapter

        adapter = FullModeAdapter(models_dir=models_dir)
        ok, reason = adapter.is_available()
        if ok:
            return ModeResolution(
                mode="full",
                badge_text="Full Mode: Live inference",
                details=reason,
            )
        return ModeResolution(
            mode="poster",
            badge_text="Poster Mode: Precomputed outputs",
            details=f"Full mode unavailable, auto-fallback enabled. {reason}",
        )

    return ModeResolution(
        mode="poster",
        badge_text="Poster Mode: Precomputed outputs",
        details="Runs offline with committed demo_assets + demo_outputs.",
    )


def scripted_steps() -> List[Dict[str, object]]:
    return [
        {
            "name": "Clean",
            "scenario": "clean",
            "audio_snr_db": 30.0,
            "vision_severity": 0.0,
            "radar_clutter_level": 0.10,
            "miss_vis": 0,
            "miss_aud": 0,
            "miss_rad": 0,
        },
        {
            "name": "Audio Degraded",
            "scenario": "audio_degraded",
            "audio_snr_db": 0.0,
            "vision_severity": 0.0,
            "radar_clutter_level": 0.10,
            "miss_vis": 0,
            "miss_aud": 0,
            "miss_rad": 0,
        },
        {
            "name": "Vision Degraded",
            "scenario": "vision_degraded",
            "audio_snr_db": 30.0,
            "vision_severity": 2.0,
            "radar_clutter_level": 0.10,
            "miss_vis": 0,
            "miss_aud": 0,
            "miss_rad": 0,
        },
        {
            "name": "Radar Clutter",
            "scenario": "radar_clutter",
            "audio_snr_db": 30.0,
            "vision_severity": 0.0,
            "radar_clutter_level": 0.90,
            "miss_vis": 0,
            "miss_aud": 0,
            "miss_rad": 0,
        },
        {
            "name": "Missing Audio",
            "scenario": "missing_audio",
            "audio_snr_db": -10.0,
            "vision_severity": 0.0,
            "radar_clutter_level": 0.30,
            "miss_vis": 0,
            "miss_aud": 1,
            "miss_rad": 0,
        },
    ]


def softmax_masked(values: Sequence[float], mask: Sequence[float]) -> List[float]:
    masked = []
    for value, keep in zip(values, mask):
        if keep > 0.5:
            masked.append(value)
        else:
            masked.append(-1e9)
    max_value = max(masked)
    exps = [math.exp(v - max_value) if v > -1e8 else 0.0 for v in masked]
    denom = sum(exps)
    if denom <= 0.0:
        return [0.0, 0.0, 0.0]
    return [v / denom for v in exps]


def clamp_prob(value: float) -> float:
    return max(1e-4, min(1.0 - 1e-4, value))


def sf(row: Dict[str, object], key: str, default: float = 0.0) -> float:
    raw = row.get(key, default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def recompute_row_with_controls(
    row: Dict[str, object],
    audio_snr_db: float,
    vision_severity: float,
    radar_clutter_level: float,
    miss_vis: int,
    miss_aud: int,
    miss_rad: int,
) -> Dict[str, float]:
    p_vis = sf(row, "p_vis", 0.5)
    p_aud = sf(row, "p_aud", 0.5)
    p_rad = sf(row, "p_rad", 0.5)

    snr_factor = max(0.15, min(1.0, (audio_snr_db + 10.0) / 40.0))
    vis_factor = max(0.20, 1.0 - 0.20 * vision_severity)
    rad_factor = max(0.25, 1.0 - 0.55 * radar_clutter_level)

    p_vis = clamp_prob(0.5 + (p_vis - 0.5) * vis_factor)
    p_aud = clamp_prob(0.5 + (p_aud - 0.5) * snr_factor)
    p_rad = clamp_prob(0.5 + (p_rad - 0.5) * rad_factor)

    mask_vis = 0.0 if int(miss_vis) else 1.0
    mask_aud = 0.0 if int(miss_aud) else 1.0
    mask_rad = 0.0 if int(miss_rad) else 1.0
    mask = [mask_vis, mask_aud, mask_rad]

    if mask_vis < 0.5:
        p_vis = 0.5
    if mask_aud < 0.5:
        p_aud = 0.5
    if mask_rad < 0.5:
        p_rad = 0.5

    q_vis = max(0.0, min(1.0, 1.0 - 0.22 * vision_severity))
    q_aud = max(0.0, min(1.0, (audio_snr_db + 10.0) / 40.0))
    q_rad = max(0.0, min(1.0, 1.0 - 0.60 * radar_clutter_level))

    gate_logits = [
        3.0 * (q_vis - 0.5) + 0.40 * (p_vis - 0.5),
        3.0 * (q_aud - 0.5) + 0.40 * (p_aud - 0.5),
        3.0 * (q_rad - 0.5) + 0.40 * (p_rad - 0.5),
    ]
    w_gate = softmax_masked(gate_logits, mask)
    p_gate = clamp_prob(
        w_gate[0] * p_vis + w_gate[1] * p_aud + w_gate[2] * p_rad
    )

    avg_weights = softmax_masked([0.0, 0.0, 0.0], mask)
    p_avg = clamp_prob(
        avg_weights[0] * p_vis + avg_weights[1] * p_aud + avg_weights[2] * p_rad
    )

    static_logits = [0.20, 0.45, 0.35]
    w_static = softmax_masked(static_logits, mask)
    p_static = clamp_prob(
        w_static[0] * p_vis + w_static[1] * p_aud + w_static[2] * p_rad
    )

    gate_noq_logits = [
        3.5 * (p_vis - 0.5),
        3.5 * (p_aud - 0.5),
        3.5 * (p_rad - 0.5),
    ]
    w_gate_noq = softmax_masked(gate_noq_logits, mask)
    p_gate_noq = clamp_prob(
        w_gate_noq[0] * p_vis + w_gate_noq[1] * p_aud + w_gate_noq[2] * p_rad
    )

    return {
        "p_vis": p_vis,
        "p_aud": p_aud,
        "p_rad": p_rad,
        "p_avg": p_avg,
        "p_static": p_static,
        "p_gate": p_gate,
        "p_gate_noq": p_gate_noq,
        "w_vis": w_gate[0],
        "w_aud": w_gate[1],
        "w_rad": w_gate[2],
        "w_static_vis": w_static[0],
        "w_static_aud": w_static[1],
        "w_static_rad": w_static[2],
        "audio_snr_db": audio_snr_db,
        "vision_severity": vision_severity,
        "radar_clutter_level": radar_clutter_level,
        "miss_vis": mask_vis,
        "miss_aud": mask_aud,
        "miss_rad": mask_rad,
    }


def summarize_export_caption(payload: Dict[str, object]) -> str:
    return (
        "This export demonstrates condition-adaptive multimodal fusion using "
        "vision, audio, and radar. The gating weights shift with quality and "
        "missing-modality controls while fused confidence remains interpretable."
        f" Mode={payload.get('mode')}, scenario={payload.get('scenario')}."
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_files_recursive(root: Path) -> List[str]:
    return sorted(
        [str(path.relative_to(root)).replace("\\", "/") for path in root.rglob("*") if path.is_file()]
    )


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
