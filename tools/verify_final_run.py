from __future__ import annotations

import csv
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PKG = ROOT / "final_isef_package"


@dataclass
class Check:
    check_id: str
    title: str
    status: str
    evidence: list[str]
    reason: str | None = None
    minimal_fix: str | None = None
    rerun_command: str | None = None


def read_text(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except Exception:
            continue
    return ""


def read_json(path: Path) -> Any:
    txt = read_text(path)
    if not txt:
        return None
    try:
        return json.loads(txt)
    except Exception:
        return None


def parse_kv(txt: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in txt.splitlines():
        line = raw.strip()
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def to_int(x: str | None, default: int = -1) -> int:
    try:
        return int(str(x))
    except Exception:
        return default


def to_float(x: str | None, default: float = -1.0) -> float:
    try:
        return float(str(x))
    except Exception:
        return default


def mtime_utc(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def run(cmd: list[str]) -> tuple[int, str, str]:
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def csv_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return next(csv.reader(f))


def csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.reader(f)
        _ = next(r, None)
        return sum(1 for _ in r)


def add(
    checks: list[Check],
    cid: str,
    title: str,
    ok: bool,
    evidence: list[str],
    reason: str,
    fix: str,
    cmd: str,
) -> None:
    checks.append(
        Check(
            check_id=cid,
            title=title,
            status="PASS" if ok else "FAIL",
            evidence=evidence,
            reason=None if ok else reason,
            minimal_fix=None if ok else fix,
            rerun_command=None if ok else cmd,
        )
    )


def verify() -> tuple[bool, list[Check], dict[str, Any]]:
    checks: list[Check] = []

    tsms_txt_p = RESULTS / "tsms_state.txt"
    tsms_txt = read_text(tsms_txt_p)
    tsms = parse_kv(tsms_txt)
    req = to_int(tsms.get("required_entries"))
    req_ok = to_int(tsms.get("required_extracted_ok_count"))
    bad_arch = to_int(tsms.get("bad_archive_count"))
    total_gb = to_float(tsms.get("total_size_gb"))
    min_gb = to_float(tsms.get("min_total_gb"))
    tsms_ok = (
        tsms_txt_p.exists()
        and "TSMS_STATE=COMPLETE" in tsms_txt
        and "TSMS_DATASET_COMPLETE=TRUE" in tsms_txt
        and req > 0
        and req_ok == req
        and bad_arch == 0
        and total_gb >= min_gb > 0
    )
    add(
        checks,
        "1.1",
        "TSMS full verification evidence is COMPLETE",
        tsms_ok,
        [
            f"path={tsms_txt_p.as_posix()} exists={tsms_txt_p.exists()}",
            f"required_extracted_ok_count={req_ok}/{req}",
            f"bad_archive_count={bad_arch}",
            f"total_size_gb={total_gb} min_total_gb={min_gb}",
        ],
        "TSMS state markers or thresholds are invalid/missing.",
        "Re-run TSMS verify.",
        "conda run -n uavmm python -m src.preprocess.verify_tsms --root data/raw/tsms_drone --out-txt results/tsms_state.txt --out-json results/tsms_state.json --complete-flag results/tsms_download_complete.flag --min-total-gb 150.0 --min-extracted-files-per-archive 1000",
    )

    flag = RESULTS / "tsms_download_complete.flag"
    logs = [RESULTS / "logs" / "download_tsms_full.log", RESULTS / "tsms_download_final.log", RESULTS / "tsms_download.log"]
    hits = []
    for p in logs:
        if not p.exists():
            continue
        txt = read_text(p)
        if ("TSMS_DOWNLOAD_COMPLETE=TRUE" in txt) or bool(re.search(r"(?m)^COMPLETE\s*$", txt)):
            hits.append(p.as_posix())
    add(
        checks,
        "1.2",
        "TSMS complete flag and COMPLETE log marker exist",
        flag.exists() and len(hits) > 0,
        [f"flag_exists={flag.exists()}", f"complete_logs={hits}"],
        "TSMS complete flag/log marker missing.",
        "Re-run TSMS downloader and retain log.",
        "powershell -NoProfile -ExecutionPolicy Bypass -File scripts/download_tsms_full.ps1 -Root data/raw/tsms_drone -Extract -DeleteArchive -RetryCount 3 -BackoffBaseSec 15 -MinExtractedFiles 1000 -CompleteFlag results/tsms_download_complete.flag",
    )

    dd_txt_p = RESULTS / "dronedetect_state.txt"
    dd_txt = read_text(dd_txt_p)
    dd = parse_kv(dd_txt)
    vv, vir, lv, lir, aud = (
        to_int(dd.get("video_v_count")),
        to_int(dd.get("video_ir_count")),
        to_int(dd.get("label_v_count")),
        to_int(dd.get("label_ir_count")),
        to_int(dd.get("audio_count")),
    )
    dd_ok = (
        dd_txt_p.exists()
        and "DRONEDETECT_STATE=COMPLETE" in dd_txt
        and "FULL_DRONEDETECT_PRESENT=YES" in dd_txt
        and vv >= 285
        and vir >= 365
        and lv >= 285
        and lir >= 365
        and aud >= 90
        and to_float(dd.get("total_size_gb")) >= 0.30
    )
    add(
        checks,
        "1.3",
        "DroneDetectionThesis full verification evidence is COMPLETE",
        dd_ok,
        [f"video_v={vv}", f"video_ir={vir}", f"label_v={lv}", f"label_ir={lir}", f"audio={aud}"],
        "DroneDetectionThesis full-data evidence is invalid/missing.",
        "Re-run DroneDetection verifier.",
        "conda run -n uavmm python -m src.preprocess.verify_dronedetect --root data/raw/dronedetect_thesis/Data --min-video-v 285 --min-video-ir 365 --min-label-v 285 --min-label-ir 365 --min-audio 90 --min-total-gb 0.30 --expect-video-v 285 --expect-video-ir 365 --expect-label-v 285 --expect-label-ir 365 --expect-audio 90 --out-txt results/dronedetect_state.txt --out-json results/dronedetect_state.json",
    )

    labels_csv = ROOT / "data/processed/vision_labels/labels_long.csv"
    labels_jsonl = ROOT / "data/processed/vision_labels/labels_long.jsonl"
    videos_csv = ROOT / "data/processed/vision_labels/videos.csv"
    viz_dir = RESULTS / "label_viz"
    viz_png = len(list(viz_dir.glob("*.png"))) if viz_dir.exists() else 0
    ls_json = RESULTS / "label_stats.json"
    ls_txt = RESULTS / "label_stats.txt"
    ls = read_json(ls_json) if ls_json.exists() else {}
    boxes = int(ls.get("boxes", 0)) if isinstance(ls, dict) else 0
    inv = float(ls.get("box_quality", {}).get("invalid_fraction", 1.0)) if isinstance(ls, dict) else 1.0
    cc = ls.get("class_counts", {}) if isinstance(ls, dict) else {}
    nonzero_classes = sum(1 for v in cc.values() if int(v) > 0) if isinstance(cc, dict) else 0
    label_ok = (
        labels_csv.exists()
        and labels_csv.stat().st_size > 50000
        and labels_jsonl.exists()
        and labels_jsonl.stat().st_size > 50000
        and videos_csv.exists()
        and viz_png >= 20
        and ls_json.exists()
        and ls_txt.exists()
        and boxes > 0
        and inv <= 0.01
        and nonzero_classes >= 2
    )
    add(
        checks,
        "2.1",
        "Label conversion artifacts and validation sanity checks pass",
        label_ok,
        [f"labels_csv={labels_csv.exists()} size={labels_csv.stat().st_size if labels_csv.exists() else 0}", f"labels_jsonl={labels_jsonl.exists()} size={labels_jsonl.stat().st_size if labels_jsonl.exists() else 0}", f"videos_csv={videos_csv.exists()}", f"label_viz_png_count={viz_png}", f"boxes={boxes}", f"invalid_fraction={inv}", f"class_counts={cc}"],
        "Label conversion/validation artifacts are missing or suspicious.",
        "Re-run label conversion and stats/overlay generation.",
        "conda run -n uavmm python -m src.preprocess.convert_mat_labels --mat-root data/raw/dronedetect_thesis/Data --out-label-csv data/processed/vision_labels/labels_long.csv --out-label-json data/processed/vision_labels/labels_long.jsonl --out-video-csv data/processed/vision_labels/videos.csv --out-stats results/label_stats.json --out-stats-txt results/label_stats.txt --overlay-dir results/label_viz --overlay-samples 80 --yolo-out data/processed/vision_yolo --yolo-frame-step 4 --seed 42",
    )

    ep = ROOT / "data/processed/episodes/episodes.csv"
    split_rpt = RESULTS / "split_integrity_report.txt"
    split_txt = read_text(split_rpt)
    split_pass = split_rpt.exists() and "SPLIT_INTEGRITY=PASS" in split_txt
    overlap_ok = False
    overlap_ev: list[str] = []
    if ep.exists():
        with ep.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
            cols = set(rows[0].keys()) if rows else set()
        needed = {"episode_id", "split", "src_video_id", "src_audio_id", "src_radar_id"}
        if needed.issubset(cols):
            by = {s: [r for r in rows if r["split"] == s] for s in ("train", "val", "test")}
            uniq_ep = len({r["episode_id"] for r in rows}) == len(rows)
            leaks = 0
            for c in ("src_video_id", "src_audio_id", "src_radar_id"):
                a, b, d = ({r[c] for r in by["train"]}, {r[c] for r in by["val"]}, {r[c] for r in by["test"]})
                ab, ad, bd = a & b, a & d, b & d
                leaks += len(ab) + len(ad) + len(bd)
                overlap_ev.append(f"{c}: train∩val={len(ab)} train∩test={len(ad)} val∩test={len(bd)}")
            overlap_ok = uniq_ep and leaks == 0 and all(by[s] for s in ("train", "val", "test"))
    add(
        checks,
        "3.1",
        "Split integrity report PASS and no src_* overlap across train/val/test",
        split_pass and overlap_ok,
        [f"split_report_exists={split_rpt.exists()}", f"split_report_pass={split_pass}", *overlap_ev],
        "Split leakage or schema issue detected.",
        "Rebuild split-safe episodes and rerun split integrity.",
        "conda run -n uavmm python -m src.eval.split_integrity --episodes data/processed/episodes/episodes.csv --out-txt results/split_integrity_report.txt --out-json results/split_integrity_report.json",
    )

    req_files = [
        "metrics_unimodal.csv",
        "metrics_fusion.csv",
        "ablation_table.csv",
        "missing_modality.csv",
        "stress_audio_snr.csv",
        "stress_vision_corruptions.csv",
        "stress_radar_clutter.csv",
        "latency_summary.json",
        "preds_fusion.csv",
        "significance_pr_auc.txt",
        "fusion_tuning.csv",
    ]
    req_figs = [
        "ablation_comparison.png",
        "stress_curves.png",
        "missing_modality_comparison.png",
        "gate_weight_behavior.png",
        "latency_plot.png",
    ]
    missing = [f for f in req_files if not (RESULTS / f).exists()]
    missing_figs = [f for f in req_figs if not (RESULTS / "figures" / f).exists()]
    pred_cols_ok = False
    pred_ev: list[str] = []
    pred_path = RESULTS / "preds_fusion.csv"
    need_pred = {"episode_id", "split", "y_true", "p_avg", "p_static", "p_gate", "p_gate_noq", "w_vis", "w_aud", "w_rad", "vision_condition", "audio_snr_db", "radar_clutter_level"}
    if pred_path.exists():
        h = set(csv_header(pred_path))
        pred_cols_ok = need_pred.issubset(h) and csv_rows(pred_path) > 0
        pred_ev = [f"pred_rows={csv_rows(pred_path)}", f"missing_pred_cols={sorted(list(need_pred - h))}"]
    add(
        checks,
        "4.1",
        "All final artifacts exist and preds_fusion has required columns",
        len(missing) == 0 and len(missing_figs) == 0 and pred_cols_ok,
        [f"missing_artifacts={missing}", f"missing_figures={missing_figs}", *pred_ev],
        "One or more final artifacts are missing or malformed.",
        "Regenerate missing eval artifacts.",
        "powershell -NoProfile -ExecutionPolicy Bypass -File run_all.ps1",
    )

    rm = read_json(RESULTS / "run_metadata.json")
    rm_ok = isinstance(rm, dict)
    full_yes = bool(rm.get("dataset_state", {}).get("full_dataset_verified", {}).get("all_modalities_full", False)) if rm_ok else False
    warn_none = bool(len(rm.get("warnings", [])) == 0) if rm_ok else False
    ts_ok = False
    freshness_ok = False
    fresh_ev: list[str] = []
    if rm_ok and isinstance(rm.get("timestamp_utc"), str):
        try:
            meta_ts = datetime.fromisoformat(rm["timestamp_utc"])
            if meta_ts.tzinfo is None:
                meta_ts = meta_ts.replace(tzinfo=timezone.utc)
            ts_ok = True
            gates = [RESULTS / "tsms_state.txt", RESULTS / "dronedetect_state.txt", RESULTS / "split_integrity_report.txt"]
            gate_t = max(mtime_utc(p) for p in gates if p.exists())
            arts = [RESULTS / f for f in req_files] + [RESULTS / "figures" / f for f in req_figs]
            old = [p.name for p in arts if p.exists() and mtime_utc(p) < gate_t]
            new = [p.name for p in arts if p.exists() and mtime_utc(p) > meta_ts]
            freshness_ok = len(old) == 0 and len(new) == 0
            fresh_ev = [f"gate_passed_at={gate_t.isoformat()}", f"run_metadata_timestamp={meta_ts.isoformat()}", f"artifacts_older_than_gate={old}", f"artifacts_newer_than_run_metadata={new}"]
        except Exception:
            pass
    add(
        checks,
        "4.2",
        "run_metadata confirms full datasets + no fallback; artifact timestamps are coherent",
        rm_ok and full_yes and warn_none and ts_ok and freshness_ok,
        [f"run_metadata_exists={rm_ok}", f"full_datasets_verified={full_yes}", f"warnings_empty={warn_none}", f"timestamp_parse_ok={ts_ok}", *fresh_ev],
        "Metadata/freshness validation failed.",
        "Regenerate run_metadata after final artifacts.",
        "conda run -n uavmm python -m src.utils.run_metadata --out results/run_metadata.json --episodes data/processed/episodes/episodes.csv --features data/processed/episodes/features_cache.npz --tsms-state results/tsms_state.json --dronedetect-state results/dronedetect_state.json --label-stats results/label_stats.json --fusion-best results/fusion_best_config.json --fusion-tuning results/fusion_tuning.csv --fallback-warnings results/fallback_warnings.txt --config-snapshot results/run_config_snapshot.json --config-used results/config_used.yaml --seed 42",
    )

    uni = RESULTS / "metrics_unimodal.csv"
    uni_ids = set()
    if uni.exists():
        with uni.open("r", encoding="utf-8-sig", newline="") as f:
            uni_ids = {r.get("run_id", "") for r in csv.DictReader(f)}
    sep_scripts = all((ROOT / p).exists() for p in ["src/train/train_audio.py", "src/train/train_radar.py", "src/train/train_vision_yolo.py"])
    add(
        checks,
        "5.1",
        "Vision, Audio, Radar are separate modalities in scripts and outputs",
        sep_scripts and {"UNI-V", "UNI-A", "UNI-R"}.issubset(uni_ids),
        [f"separate_train_scripts={sep_scripts}", f"unimodal_run_ids={sorted(list(uni_ids))}"],
        "Modalities are not clearly separated in scripts/metrics.",
        "Restore separate modality train/eval scripts and rerun unimodal eval.",
        "conda run -n uavmm python -m src.eval.eval_unimodal --episodes data/processed/episodes/episodes.csv --audio-ckpt results/checkpoints/best_audio.pt --radar-ckpt results/checkpoints/best_radar.pt --vision-csv data/processed/episodes/vision_presence.csv --out results/metrics_unimodal.csv",
    )

    tune_py = read_text(ROOT / "src/train/tune_fusion.py")
    tune_csv = RESULTS / "fusion_tuning.csv"
    tune_h = csv_header(tune_csv) if tune_csv.exists() else []
    tune_ok = (
        'idx_train = np.where(split == "train")[0]' in tune_py
        and 'idx_val = np.where(split == "val")[0]' in tune_py
        and "val_pr_auc_gate" in tune_py
        and "idx_test" not in tune_py
        and tune_csv.exists()
        and not any("test" in c.lower() for c in tune_h)
    )
    mf = RESULTS / "metrics_fusion.csv"
    mf_splits = set()
    if mf.exists():
        with mf.open("r", encoding="utf-8-sig", newline="") as f:
            mf_splits = {r.get("split", "") for r in csv.DictReader(f)}
    best_logged = bool(rm.get("fusion", {}).get("best_config")) if rm_ok else False
    add(
        checks,
        "6.1",
        "Fusion tuning uses train/val only; test used for frozen final eval; best config logged",
        tune_ok and mf_splits == {"test"} and best_logged,
        [f"tune_columns={tune_h}", f"metrics_fusion_splits={sorted(list(mf_splits))}", f"best_config_logged={best_logged}"],
        "Fusion tuning/eval split discipline not proven.",
        "Rerun tune_fusion (train/val) then eval_fusion (--split test).",
        "conda run -n uavmm python -m src.train.tune_fusion --episodes data/processed/episodes/episodes.csv --features data/processed/episodes/features_cache.npz --vision-csv data/processed/episodes/vision_presence.csv --audio-ckpt results/checkpoints/best_audio.pt --radar-ckpt results/checkpoints/best_radar.pt --seed 42 --epochs 260 --lr-grid 0.0005 --drop-one-grid 0.10 --drop-two-grid 0.03 --aux-weight-grid 0.0 --max-configs 16 --out-csv results/fusion_tuning.csv --best-json results/fusion_best_config.json --train-final",
    )

    script_ok = (ROOT / "scripts/run_all.ps1").exists() and (ROOT / "run_all.ps1").exists()
    cfg_ok = (RESULTS / "config_used.yaml").exists()
    git_commit_present = bool(rm.get("git", {}).get("commit")) if rm_ok else False
    add(
        checks,
        "7.1",
        "Reproducibility artifacts exist (run_all.ps1, run_metadata with git hash, config_used.yaml)",
        script_ok and cfg_ok and rm_ok and git_commit_present,
        [f"scripts/run_all.ps1_exists={(ROOT / 'scripts/run_all.ps1').exists()}", f"run_all.ps1_exists={(ROOT / 'run_all.ps1').exists()}", f"run_metadata_exists={rm_ok}", f"git_commit_present={git_commit_present}", f"config_used_exists={cfg_ok}"],
        "Missing reproducibility artifacts or git hash in run_metadata.",
        "Commit code and regenerate run_metadata/config_used.",
        "conda run -n uavmm python -m src.utils.run_metadata --out results/run_metadata.json --episodes data/processed/episodes/episodes.csv --features data/processed/episodes/features_cache.npz --tsms-state results/tsms_state.json --dronedetect-state results/dronedetect_state.json --label-stats results/label_stats.json --fusion-best results/fusion_best_config.json --fusion-tuning results/fusion_tuning.csv --fallback-warnings results/fallback_warnings.txt --config-snapshot results/run_config_snapshot.json --config-used results/config_used.yaml --seed 42",
    )

    pkg_req = [
        PKG / "README_CHECKLIST.txt",
        PKG / "tables/metrics_unimodal.csv",
        PKG / "tables/metrics_fusion.csv",
        PKG / "tables/ablation_table.csv",
        PKG / "tables/missing_modality.csv",
        PKG / "tables/stress_audio_snr.csv",
        PKG / "tables/stress_vision_corruptions.csv",
        PKG / "tables/stress_radar_clutter.csv",
        PKG / "tables/preds_fusion.csv",
        PKG / "tables/fusion_tuning.csv",
        PKG / "verification/tsms_state.txt",
        PKG / "verification/dronedetect_state.txt",
        PKG / "verification/split_integrity_report.txt",
        PKG / "verification/label_stats.txt",
        PKG / "metadata/run_metadata.json",
        PKG / "metadata/config_used.yaml",
        PKG / "metadata/significance_pr_auc.txt",
    ]
    pkg_missing = [p.as_posix() for p in pkg_req if not p.exists()]
    has_data = False
    if PKG.exists():
        for p in PKG.rglob("*"):
            if p.is_file() and p.relative_to(PKG).as_posix().lower().startswith("data/"):
                has_data = True
                break
    add(
        checks,
        "8.1",
        "final_isef_package is complete and excludes raw/processed datasets",
        PKG.exists() and len(pkg_missing) == 0 and not has_data,
        [f"final_isef_package_exists={PKG.exists()}", f"missing={pkg_missing}", f"contains_data_tree={has_data}"],
        "Poster package missing required files or includes dataset tree.",
        "Rebuild package script output.",
        "powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build_final_isef_package.ps1",
    )

    gi = ROOT / ".gitignore"
    gi_txt = read_text(gi)
    needed = ["data/", "datasets/", "runs/", "*.zip", "*.7z", "results/"]
    miss = [p for p in needed if p not in gi_txt]
    rc1, out1, err1 = run(["git", "status", "--short"])
    rc2, out2, err2 = run(["git", "ls-files"])
    tracked = [x.strip() for x in out2.splitlines() if x.strip()]
    bad = [p for p in tracked if re.search(r"^(data/|datasets/|runs/|results/|final_isef_package/)|\.(zip|7z|pt|pth|npy|npz|mat|mp4|avi|wav)$", p.replace("\\", "/"), re.I)]
    huge = []
    for p in tracked:
        fp = ROOT / p
        if fp.exists() and fp.is_file() and fp.stat().st_size > 50 * 1024 * 1024:
            huge.append(f"{p} ({fp.stat().st_size} bytes)")
    add(
        checks,
        "9.1",
        "GitHub-ready ignore/tracking audit passes (no forbidden or huge tracked files)",
        gi.exists() and len(miss) == 0 and rc1 == 0 and rc2 == 0 and len(bad) == 0 and len(huge) == 0,
        [f".gitignore_exists={gi.exists()}", f"missing_ignore_patterns={miss}", "git status --short:", out1 if out1 else "(clean)", f"tracked_files={len(tracked)}", f"forbidden_tracked={bad}", f"huge_tracked={huge}", f"git_errors={[err1, err2]}"],
        "Git tracking/ignore policy violated.",
        "Untrack forbidden files and recommit.",
        "git rm -r --cached data results runs final_isef_package ; git commit -m \"Remove forbidden tracked artifacts\"",
    )

    pass_n = sum(1 for c in checks if c.status == "PASS")
    fail_n = sum(1 for c in checks if c.status == "FAIL")
    ok = fail_n == 0
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(ROOT),
        "overall_status": "PASS" if ok else "FAIL",
        "pass_count": pass_n,
        "fail_count": fail_n,
        "checks": [asdict(c) for c in checks],
    }
    return ok, checks, summary


def write_reports(ok: bool, checks: list[Check], summary: dict[str, Any]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    txt = RESULTS / "verification_report.txt"
    js = RESULTS / "verification_report.json"
    lines = [
        "ISEF FINAL RUN VERIFICATION REPORT",
        f"Generated UTC: {summary['generated_utc']}",
        f"Overall: {summary['overall_status']}",
        f"Pass={summary['pass_count']} Fail={summary['fail_count']}",
        "",
    ]
    for c in checks:
        lines.append(f"[{c.status}] {c.check_id} {c.title}")
        for ev in c.evidence:
            lines.append(f"  - {ev}")
        if c.status == "FAIL":
            lines.append(f"  - reason: {c.reason}")
            lines.append(f"  - minimal_fix: {c.minimal_fix}")
            lines.append(f"  - rerun_command: {c.rerun_command}")
        lines.append("")
    txt.write_text("\n".join(lines), encoding="utf-8")
    js.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {txt}")
    print(f"Wrote {js}")
    print("Verification PASSED." if ok else "Verification FAILED.")


def main() -> int:
    ok, checks, summary = verify()
    write_reports(ok, checks, summary)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
