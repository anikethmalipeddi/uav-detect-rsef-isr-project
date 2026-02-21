from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "final_isef_package"
REPORT_PATH = PKG / "logs" / "verification_report.txt"


REQUIRED_CLASSES = ["AIRPLANE", "BIRD", "DRONE", "HELICOPTER"]
REQUIRED_FIGURES = [
    "pr_curve_fusion_vs_baselines.png",
    "roc_curve_fusion_vs_baselines.png",
    "confusion_matrix_final_raw.png",
    "confusion_matrix_final_norm.png",
    "ablation_bars.png",
    "missing_modality_bars.png",
    "stress_audio_snr_curve.png",
    "stress_vision_corruption_curve.png",
    "stress_radar_clutter_curve.png",
    "gating_weights_timeseries.png",
    "quality_vs_weight_audio.png",
    "quality_vs_weight_vision.png",
    "quality_vs_weight_radar.png",
    "latency_breakdown.png",
    "qualitative_overlays_grid.png",
]


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def _check_file(path: Path, name: str) -> CheckResult:
    if not path.exists():
        return CheckResult(name=name, ok=False, detail=f"missing: {path}")
    if not path.is_file():
        return CheckResult(name=name, ok=False, detail=f"not a file: {path}")
    if path.stat().st_size <= 0:
        return CheckResult(name=name, ok=False, detail=f"empty file: {path}")
    return CheckResult(name=name, ok=True, detail=f"ok ({path.stat().st_size} bytes)")


def _check_csv_schema(path: Path, required_cols: Sequence[str], name: str) -> CheckResult:
    file_check = _check_file(path, name)
    if not file_check.ok:
        return file_check
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        cols = reader.fieldnames or []
        missing = [c for c in required_cols if c not in cols]
        rows = list(reader)
    if missing:
        return CheckResult(name=name, ok=False, detail=f"missing columns {missing} in {path}")
    if len(rows) == 0:
        return CheckResult(name=name, ok=False, detail=f"no rows in {path}")
    return CheckResult(name=name, ok=True, detail=f"rows={len(rows)}, cols={len(cols)}")


def _check_json(path: Path, required_keys: Sequence[str], name: str) -> CheckResult:
    file_check = _check_file(path, name)
    if not file_check.ok:
        return file_check
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return CheckResult(name=name, ok=False, detail=f"invalid json: {exc}")
    missing = [k for k in required_keys if k not in payload]
    if missing:
        return CheckResult(name=name, ok=False, detail=f"missing keys {missing} in {path}")
    return CheckResult(name=name, ok=True, detail=f"keys ok: {required_keys}")


def _check_image_readable(path: Path, name: str) -> CheckResult:
    file_check = _check_file(path, name)
    if not file_check.ok:
        return file_check
    try:
        with Image.open(path) as img:
            img.verify()
    except Exception as exc:
        return CheckResult(name=name, ok=False, detail=f"image unreadable {path}: {exc}")
    return CheckResult(name=name, ok=True, detail=f"image readable: {path.name}")


def _check_dataset_classes() -> CheckResult:
    yaml_path = ROOT / "data" / "processed" / "vision_yolo" / "dataset.yaml"
    file_check = _check_file(yaml_path, "dataset.yaml present")
    if not file_check.ok:
        return file_check
    payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    names = payload.get("names")
    if not isinstance(names, list):
        return CheckResult("dataset classes", False, "dataset.yaml names is not a list")
    if sorted(names) != sorted(REQUIRED_CLASSES):
        return CheckResult(
            "dataset classes",
            False,
            f"class mismatch. expected={REQUIRED_CLASSES}, found={names}",
        )
    if len(set(names)) < 4:
        return CheckResult("dataset classes", False, f"not multi-class: {names}")
    return CheckResult("dataset classes", True, f"4-class labels confirmed: {names}")


def _check_figures_png_only(fig_dir: Path) -> CheckResult:
    if not fig_dir.exists():
        return CheckResult("figures dir exists", False, f"missing {fig_dir}")
    bad = []
    for path in fig_dir.iterdir():
        if path.is_file() and path.suffix.lower() != ".png":
            bad.append(path.name)
    if bad:
        return CheckResult("figures png-only", False, f"non-PNG files present: {bad}")
    return CheckResult("figures png-only", True, "all figure files are PNG")


def run_checks() -> List[CheckResult]:
    results: List[CheckResult] = []

    for rel in [
        "README.md",
        "poster_kit/README.md",
        "poster_kit/artifact_index.csv",
        "provenance/run_metadata.json",
        "provenance/environment.yml",
        "provenance/data_manifest.csv",
        "demo/README.md",
        "demo/assets/manifest.json",
        "demo/output/demo_screenshot.png",
        "demo/app.py",
        "demo/run_demo.py",
        "logs/final_run.log",
    ]:
        results.append(_check_file(PKG / rel, f"required file: {rel}"))

    req_demo_env = PKG / "demo" / "requirements_cpu.txt"
    req_demo_conda = PKG / "demo" / "environment_cpu.yml"
    if req_demo_env.exists() or req_demo_conda.exists():
        chosen = req_demo_env if req_demo_env.exists() else req_demo_conda
        results.append(_check_file(chosen, "demo env/requirements"))
    else:
        results.append(
            CheckResult(
                name="demo env/requirements",
                ok=False,
                detail="missing both demo/requirements_cpu.txt and demo/environment_cpu.yml",
            )
        )

    results.append(_check_dataset_classes())

    results.append(
        _check_csv_schema(
            PKG / "metrics" / "metrics_unimodal.csv",
            ["run_id", "split", "pr_auc", "roc_auc"],
            "metrics_unimodal schema",
        )
    )
    results.append(
        _check_csv_schema(
            PKG / "metrics" / "metrics_fusion.csv",
            ["run_id", "split", "pr_auc", "roc_auc"],
            "metrics_fusion schema",
        )
    )
    results.append(
        _check_json(
            PKG / "metrics" / "latency_summary.json",
            ["methods"],
            "latency_summary schema",
        )
    )
    metrics_vision_path = PKG / "metrics" / "metrics_vision.json"
    if metrics_vision_path.exists():
        results.append(
            _check_json(
                metrics_vision_path,
                ["map50_95", "map50"],
                "metrics_vision schema",
            )
        )
    else:
        results.append(
            CheckResult(
                "metrics_vision presence",
                False,
                "missing final_isef_package/metrics/metrics_vision.json",
            )
        )

    results.append(
        _check_csv_schema(
            PKG / "tables" / "ablation_table.csv",
            ["run_id", "split", "pr_auc", "roc_auc"],
            "ablation_table schema",
        )
    )
    results.append(
        _check_csv_schema(
            PKG / "tables" / "missing_modality.csv",
            ["run_id", "split", "scenario", "pr_auc"],
            "missing_modality schema",
        )
    )
    results.append(
        _check_csv_schema(
            PKG / "tables" / "stress_audio_snr.csv",
            ["run_id", "split", "snr_db", "pr_auc"],
            "stress_audio_snr schema",
        )
    )
    results.append(
        _check_csv_schema(
            PKG / "tables" / "stress_vision_corruptions.csv",
            ["run_id", "split", "corruption", "severity", "pr_auc"],
            "stress_vision_corruptions schema",
        )
    )
    results.append(
        _check_csv_schema(
            PKG / "tables" / "stress_radar_clutter.csv",
            ["run_id", "split", "clutter_level", "pr_auc"],
            "stress_radar_clutter schema",
        )
    )

    results.append(
        _check_csv_schema(
            PKG / "provenance" / "data_manifest.csv",
            ["artifact_id", "modality", "source_url", "local_path", "sha256"],
            "data_manifest schema",
        )
    )
    results.append(
        _check_json(
            PKG / "provenance" / "run_metadata.json",
            ["current_git", "system", "torch", "environment_export"],
            "provenance run_metadata schema",
        )
    )

    fig_dir = PKG / "figures"
    results.append(_check_figures_png_only(fig_dir))
    for fig_name in REQUIRED_FIGURES:
        results.append(_check_image_readable(fig_dir / fig_name, f"figure: {fig_name}"))

    # Explicit confusion matrix checks for final model.
    results.append(
        _check_image_readable(
            fig_dir / "confusion_matrix_final_raw.png",
            "confusion_matrix_final_raw exists and readable",
        )
    )
    results.append(
        _check_image_readable(
            fig_dir / "confusion_matrix_final_norm.png",
            "confusion_matrix_final_norm exists and readable",
        )
    )

    return results


def render_report(results: Sequence[CheckResult]) -> str:
    lines = []
    lines.append("ISEF PACKAGE VERIFICATION")
    lines.append(f"Package path: {PKG}")
    lines.append("")
    pass_count = sum(1 for r in results if r.ok)
    fail_count = sum(1 for r in results if not r.ok)
    for res in results:
        mark = "PASS" if res.ok else "FAIL"
        lines.append(f"[{mark}] {res.name}: {res.detail}")
    lines.append("")
    lines.append(f"SUMMARY: PASS={pass_count} FAIL={fail_count}")
    lines.append("FINAL RESULT: PASS" if fail_count == 0 else "FINAL RESULT: FAIL")
    return "\n".join(lines) + "\n"


def main() -> int:
    checks = run_checks()
    report = render_report(checks)
    print(report, end="")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    return 0 if all(c.ok for c in checks) else 1


if __name__ == "__main__":
    sys.exit(main())

