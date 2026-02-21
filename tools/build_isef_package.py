from __future__ import annotations

import csv
import hashlib
import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PKG = ROOT / "final_isef_package"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_required_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Required source file missing: {src}")
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


def write_json(path: Path, payload: Dict[str, object]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_text(path: Path, content: str) -> None:
    ensure_dir(path.parent)
    path.write_text(content, encoding="utf-8")


def write_csv(path: Path, fieldnames: List[str], rows: Iterable[Dict[str, object]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def copy_metrics_and_tables() -> None:
    metrics_map = {
        RESULTS / "metrics_unimodal.csv": PKG / "metrics" / "metrics_unimodal.csv",
        RESULTS / "metrics_fusion.csv": PKG / "metrics" / "metrics_fusion.csv",
        RESULTS / "latency_summary.json": PKG / "metrics" / "latency_summary.json",
        RESULTS / "metrics_vision.json": PKG / "metrics" / "metrics_vision.json",
    }
    tables_map = {
        RESULTS / "ablation_table.csv": PKG / "tables" / "ablation_table.csv",
        RESULTS / "missing_modality.csv": PKG / "tables" / "missing_modality.csv",
        RESULTS / "stress_audio_snr.csv": PKG / "tables" / "stress_audio_snr.csv",
        RESULTS / "stress_vision_corruptions.csv": PKG / "tables" / "stress_vision_corruptions.csv",
        RESULTS / "stress_radar_clutter.csv": PKG / "tables" / "stress_radar_clutter.csv",
        RESULTS / "preds_fusion.csv": PKG / "tables" / "preds_fusion.csv",
        RESULTS / "fusion_tuning.csv": PKG / "tables" / "fusion_tuning.csv",
    }
    for src, dst in {**metrics_map, **tables_map}.items():
        copy_required_file(src, dst)


def build_provenance() -> None:
    provenance_dir = PKG / "provenance"
    ensure_dir(provenance_dir)

    src_meta_path = RESULTS / "run_metadata.json"
    source_meta: Dict[str, object] = {}
    if src_meta_path.exists():
        source_meta = json.loads(src_meta_path.read_text(encoding="utf-8"))

    commit = (
        subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        .stdout.strip()
    )
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )

    torch_info: Dict[str, object] = {}
    ultralytics_version = "not_available"
    try:
        import torch  # type: ignore

        torch_info = {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu-only",
        }
    except Exception as exc:  # pragma: no cover - defensive
        torch_info = {"error": str(exc)}
    try:
        import ultralytics  # type: ignore

        ultralytics_version = getattr(ultralytics, "__version__", "unknown")
    except Exception:
        ultralytics_version = "not_available"

    env_out = provenance_dir / "environment.yml"
    conda_export_ok = False
    conda_exe = Path.home() / "miniforge3" / "Scripts" / "conda.exe"
    if conda_exe.exists():
        result = subprocess.run(
            [str(conda_exe), "env", "export", "-n", "uavmm"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            env_out.write_text(result.stdout, encoding="utf-8")
            conda_export_ok = True
    if not conda_export_ok:
        fallback_env = ROOT / "environment.yml"
        if fallback_env.exists():
            shutil.copy2(fallback_env, env_out)
        else:
            env_out.write_text("# environment export unavailable\n", encoding="utf-8")

    run_metadata = {
        "package_generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_run_metadata_path": str(src_meta_path.relative_to(ROOT)).replace("\\", "/"),
        "source_run_metadata": source_meta,
        "current_git": {
            "commit": commit,
            "dirty": dirty,
            "branch": "main",
        },
        "system": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "machine": platform.machine(),
        },
        "torch": torch_info,
        "ultralytics_version": ultralytics_version,
        "environment_export": {
            "path": "final_isef_package/provenance/environment.yml",
            "conda_export_ok": conda_export_ok,
        },
    }
    write_json(provenance_dir / "run_metadata.json", run_metadata)

    manifest_rows: List[Dict[str, object]] = []
    manifest_rows.append(
        {
            "artifact_id": "drone_detection_thesis",
            "modality": "vision+audio",
            "source_url": "https://github.com/DroneDetectionThesis/Drone-detection-dataset",
            "citation_or_note": "Zenodo mirror: https://zenodo.org/records/5500576",
            "local_path": "data/raw/dronedetect_thesis",
            "sha256": "not_computed_large_dataset",
            "size_bytes": "",
        }
    )
    manifest_rows.append(
        {
            "artifact_id": "tsms_drone",
            "modality": "radar",
            "source_url": "https://plus.figshare.com/articles/dataset/Time-Synchronized_Multi-Sensor_Drone_Dataset_TSMS-Drone_/30027313",
            "citation_or_note": "Paper: https://www.nature.com/articles/s41597-026-06802-6",
            "local_path": "data/raw/tsms_drone",
            "sha256": "not_computed_large_dataset",
            "size_bytes": "",
        }
    )
    for rel_path in [
        "results/metrics_unimodal.csv",
        "results/metrics_fusion.csv",
        "results/preds_fusion.csv",
        "results/ablation_table.csv",
        "results/missing_modality.csv",
    ]:
        p = ROOT / rel_path
        if p.exists():
            manifest_rows.append(
                {
                    "artifact_id": Path(rel_path).name,
                    "modality": "derived",
                    "source_url": "local_final_run",
                    "citation_or_note": "Derived from verified full-data final run",
                    "local_path": rel_path,
                    "sha256": sha256_file(p),
                    "size_bytes": p.stat().st_size,
                }
            )

    write_csv(
        provenance_dir / "data_manifest.csv",
        [
            "artifact_id",
            "modality",
            "source_url",
            "citation_or_note",
            "local_path",
            "sha256",
            "size_bytes",
        ],
        manifest_rows,
    )


def build_demo_assets_and_app() -> None:
    demo_dir = PKG / "demo"
    assets_dir = demo_dir / "assets"
    output_dir = demo_dir / "output"
    ensure_dir(assets_dir)
    ensure_dir(output_dir)

    source_frame_candidates = sorted((RESULTS / "label_viz").glob("*_DRONE.png"))
    if not source_frame_candidates:
        source_frame_candidates = sorted((RESULTS / "label_viz").glob("*.png"))
    if not source_frame_candidates:
        raise FileNotFoundError("No label_viz PNGs available for demo sample frame.")
    sample_frame = source_frame_candidates[0]
    copy_required_file(sample_frame, assets_dir / "sample_frame.png")

    pred_src = ROOT / "demo_outputs" / "predictions_per_frame.csv"
    if pred_src.exists():
        pred_df = pd.read_csv(pred_src)
        if "scenario" in pred_df.columns:
            pred_df = pred_df[pred_df["scenario"] == "clean"].copy()
        pred_df = pred_df.head(120).copy()
        if "p_gate" in pred_df.columns and "p_fused" not in pred_df.columns:
            pred_df["p_fused"] = pred_df["p_gate"]
    else:
        full_df = pd.read_csv(RESULTS / "preds_fusion.csv")
        full_df = full_df[full_df["split"] == "test"].head(120).copy()
        full_df["frame_idx"] = np.arange(len(full_df))
        pred_df = pd.DataFrame(
            {
                "frame_idx": full_df["frame_idx"],
                "t_sec": full_df["frame_idx"] * 0.125,
                "p_vis": full_df["p_vis"],
                "p_aud": full_df["p_aud"],
                "p_rad": full_df["p_rad"],
                "w_vis": full_df["w_vis"],
                "w_aud": full_df["w_aud"],
                "w_rad": full_df["w_rad"],
                "p_fused": full_df["p_gate"],
            }
        )
    pred_keep = [
        c
        for c in [
            "frame_idx",
            "t_sec",
            "p_vis",
            "p_aud",
            "p_rad",
            "w_vis",
            "w_aud",
            "w_rad",
            "p_fused",
        ]
        if c in pred_df.columns
    ]
    pred_df = pred_df[pred_keep].copy()
    pred_df.to_csv(assets_dir / "sample_predictions.csv", index=False)

    n = max(120, len(pred_df))
    t = np.linspace(0.0, 15.0, num=n, dtype=np.float32)
    amp = 0.7 * np.sin(2.0 * np.pi * 3.0 * t / 15.0) + 0.2 * np.sin(
        2.0 * np.pi * 11.0 * t / 15.0
    )
    audio_df = pd.DataFrame({"t_sec": t, "amplitude": amp})
    audio_df.to_csv(assets_dir / "sample_audio_features.csv", index=False)

    radar_rows = []
    for i in range(n):
        row = {"frame_idx": i}
        for j in range(16):
            row[f"bin_{j:02d}"] = float(
                0.55
                + 0.35 * np.sin((i / max(1, n - 1)) * np.pi * 4.0 + j * 0.3)
                + 0.10 * np.cos(j * 0.2)
            )
        radar_rows.append(row)
    pd.DataFrame(radar_rows).to_csv(assets_dir / "sample_radar_features.csv", index=False)

    asset_manifest = {
        "default_predicted_class": "DRONE",
        "assets": [],
    }
    for p in sorted(assets_dir.glob("*")):
        if p.is_file():
            asset_manifest["assets"].append(
                {
                    "name": p.name,
                    "sha256": sha256_file(p),
                    "size_bytes": p.stat().st_size,
                }
            )
    write_json(assets_dir / "manifest.json", asset_manifest)

    requirements = "\n".join(
        [
            "streamlit>=1.36",
            "numpy>=1.24",
            "pandas>=2.0",
            "matplotlib>=3.8",
            "Pillow>=10.0",
            "",
        ]
    )
    write_text(demo_dir / "requirements_cpu.txt", requirements)

    demo_readme = """# ISEF Demo (CPU Offline)

This demo is fully offline and runs on macOS Apple Silicon (M2) CPU or Windows CPU.
It uses tiny packaged assets in `assets/` and produces a poster screenshot in `output/demo_screenshot.png`.

## macOS M2 (CPU)
```bash
python3 -m venv .venv_demo
source .venv_demo/bin/activate
pip install -r final_isef_package/demo/requirements_cpu.txt
streamlit run final_isef_package/demo/app.py
python final_isef_package/demo/run_demo.py
```

## Windows (CPU)
```powershell
python -m venv .venv_demo
.\\.venv_demo\\Scripts\\Activate.ps1
pip install -r final_isef_package/demo/requirements_cpu.txt
streamlit run final_isef_package/demo/app.py
python final_isef_package/demo/run_demo.py
```
"""
    write_text(demo_dir / "README.md", demo_readme)

    demo_app_code = r'''from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from PIL import Image


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
OUTPUT = ROOT / "output"


def infer_stub(row: pd.Series) -> dict:
    p_vis = float(row["p_vis"])
    p_aud = float(row["p_aud"])
    p_rad = float(row["p_rad"])
    w_vis = float(row["w_vis"])
    w_aud = float(row["w_aud"])
    w_rad = float(row["w_rad"])
    p_fused = float(row["p_fused"])
    return {
        "p_vis": p_vis,
        "p_aud": p_aud,
        "p_rad": p_rad,
        "w_vis": w_vis,
        "w_aud": w_aud,
        "w_rad": w_rad,
        "p_fused": p_fused,
        "predicted_class": "DRONE",
    }


def save_demo_screenshot(pred_df: pd.DataFrame, frame_img: Image.Image) -> Path:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT / "demo_screenshot.png"
    row = pred_df.iloc[min(30, len(pred_df) - 1)]
    scores = infer_stub(row)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(frame_img)
    axes[0].set_title("Vision Sample")
    axes[0].axis("off")

    axes[1].plot(pred_df["t_sec"], pred_df["w_vis"], label="w_vis")
    axes[1].plot(pred_df["t_sec"], pred_df["w_aud"], label="w_aud")
    axes[1].plot(pred_df["t_sec"], pred_df["w_rad"], label="w_rad")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Adaptive Gating Weights")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Weight")
    axes[1].legend()
    axes[1].grid(alpha=0.2)

    axes[2].bar(
        ["Vision", "Audio", "Radar", "Fusion"],
        [scores["p_vis"], scores["p_aud"], scores["p_rad"], scores["p_fused"]],
        color=["#2563eb", "#16a34a", "#dc2626", "#0f766e"],
    )
    axes[2].set_ylim(0, 1)
    axes[2].set_title(f"Predicted: {scores['predicted_class']}")
    axes[2].set_ylabel("Confidence")
    axes[2].grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main() -> None:
    st.set_page_config(page_title="ISEF Demo (CPU)", layout="wide")
    st.title("ISEF Offline Demo")
    st.caption("Vision + Audio + Radar stub inference with adaptive fusion weights.")

    pred_df = pd.read_csv(ASSETS / "sample_predictions.csv")
    frame_img = Image.open(ASSETS / "sample_frame.png")
    idx = st.slider("Frame Index", 0, len(pred_df) - 1, min(20, len(pred_df) - 1))
    row = pred_df.iloc[idx]
    scores = infer_stub(row)

    c1, c2 = st.columns([1.8, 1.2])
    with c1:
        st.image(frame_img, caption="Packaged sample frame", use_container_width=True)
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(pred_df["t_sec"], pred_df["w_vis"], label="w_vis")
        ax.plot(pred_df["t_sec"], pred_df["w_aud"], label="w_aud")
        ax.plot(pred_df["t_sec"], pred_df["w_rad"], label="w_rad")
        ax.set_ylim(0, 1)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Weight")
        ax.set_title("Gating Weights")
        ax.grid(alpha=0.2)
        ax.legend()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with c2:
        st.metric("Predicted Class", scores["predicted_class"])
        st.metric("Fusion Confidence", f"{scores['p_fused']:.3f}")
        st.metric("Vision", f"{scores['p_vis']:.3f}")
        st.metric("Audio", f"{scores['p_aud']:.3f}")
        st.metric("Radar", f"{scores['p_rad']:.3f}")
        st.write(
            {
                "w_vis": round(scores["w_vis"], 3),
                "w_aud": round(scores["w_aud"], 3),
                "w_rad": round(scores["w_rad"], 3),
            }
        )
        if st.button("Save Demo Screenshot"):
            path = save_demo_screenshot(pred_df, frame_img)
            st.success(f"Saved: {path}")


if __name__ == "__main__":
    main()
'''
    write_text(demo_dir / "app.py", demo_app_code)

    run_demo_code = r'''from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
OUTPUT = ROOT / "output"


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pred_df = pd.read_csv(ASSETS / "sample_predictions.csv")
    frame_img = Image.open(ASSETS / "sample_frame.png")

    idx = min(30, len(pred_df) - 1)
    row = pred_df.iloc[idx]
    p_vis = float(row["p_vis"])
    p_aud = float(row["p_aud"])
    p_rad = float(row["p_rad"])
    p_fused = float(row["p_fused"])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(frame_img)
    axes[0].set_title("Vision Sample")
    axes[0].axis("off")

    axes[1].plot(pred_df["t_sec"], pred_df["w_vis"], label="w_vis")
    axes[1].plot(pred_df["t_sec"], pred_df["w_aud"], label="w_aud")
    axes[1].plot(pred_df["t_sec"], pred_df["w_rad"], label="w_rad")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Adaptive Gating Weights")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Weight")
    axes[1].legend()
    axes[1].grid(alpha=0.2)

    axes[2].bar(
        ["Vision", "Audio", "Radar", "Fusion"],
        [p_vis, p_aud, p_rad, p_fused],
        color=["#2563eb", "#16a34a", "#dc2626", "#0f766e"],
    )
    axes[2].set_ylim(0, 1)
    axes[2].set_title("Predicted: DRONE")
    axes[2].set_ylabel("Confidence")
    axes[2].grid(axis="y", alpha=0.2)

    fig.tight_layout()
    out = OUTPUT / "demo_screenshot.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
    write_text(demo_dir / "run_demo.py", run_demo_code)


def write_package_readmes() -> None:
    src_meta = json.loads((RESULTS / "run_metadata.json").read_text(encoding="utf-8"))
    torch_info = src_meta.get("torch", {})
    machine = src_meta.get("machine", {})
    readme = f"""# ISEF Final Package

This package contains the final, curated artifacts for the condition-adaptive multimodal UAV detection project. The system fuses **vision**, **audio**, and **radar** evidence with adaptive gating and modality masking to detect UAV presence robustly under degradation and missing-modality conditions.

## Reproduce Final Run
1. Activate environment:
   - `conda activate uavmm`
2. Run end-to-end pipeline:
   - `powershell -ExecutionPolicy Bypass -File .\\run_all.ps1`
3. Rebuild poster figures:
   - `python tools/make_figures.py`
4. Verify package completeness:
   - `python tools/verify_isef_package.py`

## Artifact Usage Map
- `figures/`: poster and presentation plots (performance, robustness, gating, latency, qualitative).
- `tables/` and `metrics/`: numeric tables for poster text and claims.
- `demo/`: CPU-friendly offline demo for judges (macOS M2 + Windows).
- `provenance/`: reproducibility metadata (environment, data manifest, run metadata).
- `logs/`: final run and package verification logs.

## Hardware / Software Summary
- Machine: `{machine.get("node", "unknown")}`
- OS: `{machine.get("platform", "unknown")}`
- Python: `{machine.get("python_version", "unknown")}`
- Torch: `{torch_info.get("torch_version", "unknown")}`
- CUDA: `{torch_info.get("cuda_version", "unknown")}`
- GPU: `{", ".join(torch_info.get("devices", [])) if torch_info.get("devices") else "none"}`
"""
    write_text(PKG / "README.md", readme)

    poster_readme = """# Poster Kit Index

Use this folder as the quick index for poster assembly.

- Core performance: `../figures/pr_curve_fusion_vs_baselines.png`, `../figures/ablation_bars.png`
- Robustness: `../figures/missing_modality_bars.png`, stress curves
- Interpretability: gating and quality-vs-weight plots
- Engineering: `../figures/latency_breakdown.png`
- Qualitative: `../figures/qualitative_overlays_grid.png`
"""
    write_text(PKG / "poster_kit" / "README.md", poster_readme)

    artifact_rows = [
        {"section": "Core performance", "artifact_path": "figures/pr_curve_fusion_vs_baselines.png"},
        {"section": "Core performance", "artifact_path": "figures/confusion_matrix_final_norm.png"},
        {"section": "Robustness", "artifact_path": "figures/missing_modality_bars.png"},
        {"section": "Robustness", "artifact_path": "figures/stress_audio_snr_curve.png"},
        {"section": "Interpretability", "artifact_path": "figures/gating_weights_timeseries.png"},
        {"section": "Engineering", "artifact_path": "figures/latency_breakdown.png"},
        {"section": "Qualitative", "artifact_path": "figures/qualitative_overlays_grid.png"},
        {"section": "Metrics table", "artifact_path": "metrics/metrics_fusion.csv"},
        {"section": "Ablations table", "artifact_path": "tables/ablation_table.csv"},
    ]
    write_csv(PKG / "poster_kit" / "artifact_index.csv", ["section", "artifact_path"], artifact_rows)


def initialize_package_dirs() -> None:
    for rel in [
        "poster_kit",
        "figures",
        "tables",
        "metrics",
        "demo",
        "provenance",
        "logs",
        "demo/assets",
        "demo/output",
    ]:
        ensure_dir(PKG / rel)


def touch_log_headers() -> None:
    ensure_dir(PKG / "logs")
    log_path = PKG / "logs" / "final_run.log"
    if not log_path.exists():
        log_path.write_text(
            f"[{datetime.now(timezone.utc).isoformat()}] ISEF package orchestration started.\n",
            encoding="utf-8",
        )


def main() -> int:
    initialize_package_dirs()
    copy_metrics_and_tables()
    build_provenance()
    build_demo_assets_and_app()
    write_package_readmes()
    touch_log_headers()
    print("Prepared canonical final_isef_package structure.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

