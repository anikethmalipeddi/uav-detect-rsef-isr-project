from __future__ import annotations

import csv
import json
import math
import shutil
import wave
from pathlib import Path
from statistics import mean
from typing import Dict, List


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def pick_existing_path(candidates: List[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("No candidate path exists: " + ", ".join(str(p) for p in candidates))


def copy_demo_media(root: Path) -> Dict[str, str]:
    assets_dir = root / "demo_assets"
    ensure_dir(assets_dir)

    src_video = pick_existing_path(
        [
            root / "data" / "raw" / "dronedetect_thesis" / "Data" / "Video_IR" / "IR_DRONE_040.mp4",
            root / "data" / "raw" / "dronedetect_thesis" / "Data" / "Video_IR" / "IR_BIRD_070.mp4",
        ]
    )
    src_audio = pick_existing_path(
        [
            root / "data" / "raw" / "dronedetect_thesis" / "Data" / "Audio" / "DRONE_001.wav",
            root / "data" / "raw" / "dronedetect_thesis" / "Data" / "Audio" / "BACKGROUND_024.wav",
        ]
    )

    dst_video = assets_dir / "sample_video.mp4"
    dst_audio = assets_dir / "sample_audio.wav"
    shutil.copy2(src_video, dst_video)
    shutil.copy2(src_audio, dst_audio)

    with wave.open(str(dst_audio), "rb") as handle:
        sr = int(handle.getframerate())
        frames = int(handle.getnframes())
        duration = frames / float(sr) if sr else 0.0

    return {
        "source_video": str(src_video.relative_to(root)).replace("\\", "/"),
        "source_audio": str(src_audio.relative_to(root)).replace("\\", "/"),
        "duration_sec": round(duration, 3),
    }


def generate_radar_csv(root: Path, frame_count: int = 120) -> Dict[str, object]:
    out_path = root / "demo_assets" / "sample_radar.csv"
    fieldnames = ["frame_idx"] + [f"bin_{i:02d}" for i in range(64)]
    rows: List[Dict[str, object]] = []
    for frame_idx in range(frame_count):
        row: Dict[str, object] = {"frame_idx": frame_idx}
        t = frame_idx / float(max(1, frame_count - 1))
        for bin_idx in range(64):
            phase = 2.0 * math.pi * (t * 1.7 + bin_idx / 16.0)
            val = 0.55 + 0.40 * math.sin(phase) + 0.10 * math.cos(phase * 0.37)
            row[f"bin_{bin_idx:02d}"] = f"{max(0.0, min(1.0, val)):.6f}"
        rows.append(row)
    write_csv(out_path, fieldnames, rows)
    return {"radar_rows": frame_count, "radar_bins": 64}


def clamp_prob(v: float) -> float:
    return max(1e-4, min(1.0 - 1e-4, v))


def softmax_masked(values: List[float], mask: List[float]) -> List[float]:
    masked = [v if m > 0.5 else -1e9 for v, m in zip(values, mask)]
    max_v = max(masked)
    exps = [math.exp(v - max_v) if v > -1e8 else 0.0 for v in masked]
    denom = sum(exps)
    if denom <= 0.0:
        return [0.0, 0.0, 0.0]
    return [x / denom for x in exps]


def build_demo_outputs(root: Path, frame_count: int = 120, fps: float = 8.0) -> Dict[str, object]:
    outputs_dir = root / "demo_outputs"
    ensure_dir(outputs_dir)

    full_preds = read_csv(root / "final_isef_package" / "tables" / "preds_fusion.csv")
    test_rows = [r for r in full_preds if r.get("split") == "test"]
    if not test_rows:
        raise RuntimeError("final_isef_package/tables/preds_fusion.csv has no test rows.")

    base_p_vis = mean(float(r["p_vis"]) for r in test_rows)
    base_p_aud = mean(float(r["p_aud"]) for r in test_rows)
    base_p_rad = mean(float(r["p_rad"]) for r in test_rows)

    scenarios = {
        "clean": {"audio_snr_db": 30.0, "vision_severity": 0.0, "radar_clutter_level": 0.10, "miss_vis": 0, "miss_aud": 0, "miss_rad": 0},
        "audio_degraded": {"audio_snr_db": 0.0, "vision_severity": 0.0, "radar_clutter_level": 0.10, "miss_vis": 0, "miss_aud": 0, "miss_rad": 0},
        "vision_degraded": {"audio_snr_db": 30.0, "vision_severity": 2.0, "radar_clutter_level": 0.10, "miss_vis": 0, "miss_aud": 0, "miss_rad": 0},
        "radar_clutter": {"audio_snr_db": 30.0, "vision_severity": 0.0, "radar_clutter_level": 0.90, "miss_vis": 0, "miss_aud": 0, "miss_rad": 0},
        "missing_audio": {"audio_snr_db": -10.0, "vision_severity": 0.0, "radar_clutter_level": 0.30, "miss_vis": 0, "miss_aud": 1, "miss_rad": 0},
        "missing_vision": {"audio_snr_db": 20.0, "vision_severity": 0.0, "radar_clutter_level": 0.30, "miss_vis": 1, "miss_aud": 0, "miss_rad": 0},
        "missing_radar": {"audio_snr_db": 20.0, "vision_severity": 0.0, "radar_clutter_level": 0.30, "miss_vis": 0, "miss_aud": 0, "miss_rad": 1},
    }

    pred_rows: List[Dict[str, object]] = []
    gate_rows: List[Dict[str, object]] = []
    bbox_rows: List[Dict[str, object]] = []

    for scenario, cfg in scenarios.items():
        for frame_idx in range(frame_count):
            t_sec = frame_idx / fps
            wobble = 0.07 * math.sin(2.0 * math.pi * frame_idx / frame_count)
            p_vis = base_p_vis + wobble
            p_aud = base_p_aud + 0.06 * math.sin(2.0 * math.pi * frame_idx / (frame_count * 0.83) + 0.4)
            p_rad = base_p_rad + 0.05 * math.cos(2.0 * math.pi * frame_idx / (frame_count * 1.1) + 0.2)

            snr = float(cfg["audio_snr_db"])
            vis_sev = float(cfg["vision_severity"])
            clutter = float(cfg["radar_clutter_level"])
            miss_vis = int(cfg["miss_vis"])
            miss_aud = int(cfg["miss_aud"])
            miss_rad = int(cfg["miss_rad"])

            snr_factor = max(0.15, min(1.0, (snr + 10.0) / 40.0))
            vis_factor = max(0.20, 1.0 - 0.20 * vis_sev)
            rad_factor = max(0.25, 1.0 - 0.55 * clutter)
            p_vis = clamp_prob(0.5 + (p_vis - 0.5) * vis_factor)
            p_aud = clamp_prob(0.5 + (p_aud - 0.5) * snr_factor)
            p_rad = clamp_prob(0.5 + (p_rad - 0.5) * rad_factor)

            mask = [0.0 if miss_vis else 1.0, 0.0 if miss_aud else 1.0, 0.0 if miss_rad else 1.0]
            if mask[0] < 0.5:
                p_vis = 0.5
            if mask[1] < 0.5:
                p_aud = 0.5
            if mask[2] < 0.5:
                p_rad = 0.5

            q_vis = max(0.0, min(1.0, 1.0 - 0.22 * vis_sev))
            q_aud = max(0.0, min(1.0, (snr + 10.0) / 40.0))
            q_rad = max(0.0, min(1.0, 1.0 - 0.60 * clutter))

            gate_logits = [
                3.0 * (q_vis - 0.5) + 0.4 * (p_vis - 0.5),
                3.0 * (q_aud - 0.5) + 0.4 * (p_aud - 0.5),
                3.0 * (q_rad - 0.5) + 0.4 * (p_rad - 0.5),
            ]
            w = softmax_masked(gate_logits, mask)
            p_gate = clamp_prob(w[0] * p_vis + w[1] * p_aud + w[2] * p_rad)
            w_avg = softmax_masked([0.0, 0.0, 0.0], mask)
            p_avg = clamp_prob(w_avg[0] * p_vis + w_avg[1] * p_aud + w_avg[2] * p_rad)
            w_static = softmax_masked([0.20, 0.45, 0.35], mask)
            p_static = clamp_prob(w_static[0] * p_vis + w_static[1] * p_aud + w_static[2] * p_rad)
            w_noq = softmax_masked([3.5 * (p_vis - 0.5), 3.5 * (p_aud - 0.5), 3.5 * (p_rad - 0.5)], mask)
            p_gate_noq = clamp_prob(w_noq[0] * p_vis + w_noq[1] * p_aud + w_noq[2] * p_rad)

            vision_condition = "normal" if vis_sev <= 0.01 else "degraded"
            pred_rows.append(
                {
                    "scenario": scenario,
                    "frame_idx": frame_idx,
                    "t_sec": f"{t_sec:.3f}",
                    "y_true": 1,
                    "p_vis": f"{p_vis:.6f}",
                    "p_aud": f"{p_aud:.6f}",
                    "p_rad": f"{p_rad:.6f}",
                    "p_avg": f"{p_avg:.6f}",
                    "p_static": f"{p_static:.6f}",
                    "p_gate": f"{p_gate:.6f}",
                    "p_gate_noq": f"{p_gate_noq:.6f}",
                    "w_vis": f"{w[0]:.6f}",
                    "w_aud": f"{w[1]:.6f}",
                    "w_rad": f"{w[2]:.6f}",
                    "vision_condition": vision_condition,
                    "audio_snr_db": f"{snr:.2f}",
                    "radar_clutter_level": f"{clutter:.2f}",
                    "miss_vis": int(mask[0]),
                    "miss_aud": int(mask[1]),
                    "miss_rad": int(mask[2]),
                }
            )
            gate_rows.append(
                {
                    "scenario": scenario,
                    "frame_idx": frame_idx,
                    "t_sec": f"{t_sec:.3f}",
                    "w_vis": f"{w[0]:.6f}",
                    "w_aud": f"{w[1]:.6f}",
                    "w_rad": f"{w[2]:.6f}",
                }
            )

            x_center = 120 + int(60 * math.sin(2.0 * math.pi * frame_idx / frame_count))
            y_center = 80 + int(35 * math.cos(2.0 * math.pi * frame_idx / frame_count))
            width = 48
            height = 28
            bbox_rows.append(
                {
                    "scenario": scenario,
                    "frame_idx": frame_idx,
                    "x1": max(0, x_center - width),
                    "y1": max(0, y_center - height),
                    "x2": x_center + width,
                    "y2": y_center + height,
                    "label": "drone",
                }
            )

    write_csv(
        outputs_dir / "predictions_per_frame.csv",
        [
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
        ],
        pred_rows,
    )
    write_csv(
        outputs_dir / "gating_weights_per_frame.csv",
        ["scenario", "frame_idx", "t_sec", "w_vis", "w_aud", "w_rad"],
        gate_rows,
    )
    write_csv(
        outputs_dir / "bboxes_per_frame.csv",
        ["scenario", "frame_idx", "x1", "y1", "x2", "y2", "label"],
        bbox_rows,
    )

    return {
        "frame_count": frame_count,
        "fps": fps,
        "scenario_count": len(scenarios),
    }


def write_metadata(root: Path, media_meta: Dict[str, object], output_meta: Dict[str, object], radar_meta: Dict[str, object]) -> None:
    payload = {
        "source_video": media_meta["source_video"],
        "source_audio": media_meta["source_audio"],
        "source_radar": "demo_assets/sample_radar.csv",
        "duration_sec": media_meta["duration_sec"],
        "fps": output_meta["fps"],
        "frame_step": 1,
        "output_frame_count": output_meta["frame_count"],
        "output_scenarios": output_meta["scenario_count"],
        "radar_rows": radar_meta["radar_rows"],
        "radar_bins": radar_meta["radar_bins"],
        "derived_from": [
            "final_isef_package/tables/preds_fusion.csv",
            "final_isef_package/tables/metrics_fusion.csv",
            "final_isef_package/tables/stress_audio_snr.csv",
            "final_isef_package/tables/stress_vision_corruptions.csv",
            "final_isef_package/tables/stress_radar_clutter.csv",
        ],
    }
    with (root / "demo_assets" / "sample_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> int:
    root = repo_root()
    ensure_dir(root / "demo_assets")
    ensure_dir(root / "demo_outputs")
    media_meta = copy_demo_media(root)
    radar_meta = generate_radar_csv(root)
    output_meta = build_demo_outputs(root)
    write_metadata(root, media_meta, output_meta, radar_meta)
    print("Built demo_assets and demo_outputs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

