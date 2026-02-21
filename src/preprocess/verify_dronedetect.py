from __future__ import annotations

import argparse
import json
from pathlib import Path


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def list_files(path: Path, exts: set[str]) -> list[Path]:
    if not path.exists():
        return []
    return [p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in exts]


def list_label_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(path.rglob("*_LABELS.mat"))


def stem_without_labels(name: str) -> str:
    return name.replace("_LABELS", "")


def verify_dronedetect(
    root: Path,
    min_video_v: int,
    min_video_ir: int,
    min_label_v: int,
    min_label_ir: int,
    min_audio: int,
    min_total_gb: float,
    expect_video_v: int | None = None,
    expect_video_ir: int | None = None,
    expect_label_v: int | None = None,
    expect_label_ir: int | None = None,
    expect_audio: int | None = None,
) -> tuple[bool, dict]:
    video_v_dir = root / "Video_V"
    video_ir_dir = root / "Video_IR"
    audio_dir = root / "Audio"

    required_dirs = [video_v_dir, video_ir_dir, audio_dir]
    required_dirs_ok = all(d.exists() and d.is_dir() for d in required_dirs)

    videos_v = list_files(video_v_dir, VIDEO_EXTS)
    videos_ir = list_files(video_ir_dir, VIDEO_EXTS)
    labels_v = list_label_files(video_v_dir)
    labels_ir = list_label_files(video_ir_dir)
    audio_files = list_files(audio_dir, AUDIO_EXTS)

    video_v_stems = {p.stem for p in videos_v}
    video_ir_stems = {p.stem for p in videos_ir}
    label_v_stems = {stem_without_labels(p.stem) for p in labels_v}
    label_ir_stems = {stem_without_labels(p.stem) for p in labels_ir}

    missing_video_v_labels = sorted(video_v_stems - label_v_stems)
    missing_video_ir_labels = sorted(video_ir_stems - label_ir_stems)
    orphan_v_labels = sorted(label_v_stems - video_v_stems)
    orphan_ir_labels = sorted(label_ir_stems - video_ir_stems)

    total_size_gb = dir_size_bytes(root) / (1024**3)
    min_size_ok = total_size_gb >= float(min_total_gb)

    counts_ok = (
        len(videos_v) >= min_video_v
        and len(videos_ir) >= min_video_ir
        and len(labels_v) >= min_label_v
        and len(labels_ir) >= min_label_ir
        and len(audio_files) >= min_audio
    )
    exact_count_checks = {
        "video_v_exact_ok": True if expect_video_v is None else (len(videos_v) == int(expect_video_v)),
        "video_ir_exact_ok": True if expect_video_ir is None else (len(videos_ir) == int(expect_video_ir)),
        "label_v_exact_ok": True if expect_label_v is None else (len(labels_v) == int(expect_label_v)),
        "label_ir_exact_ok": True if expect_label_ir is None else (len(labels_ir) == int(expect_label_ir)),
        "audio_exact_ok": True if expect_audio is None else (len(audio_files) == int(expect_audio)),
    }
    exact_counts_ok = bool(all(exact_count_checks.values()))
    pairing_ok = (
        len(missing_video_v_labels) == 0
        and len(missing_video_ir_labels) == 0
        and len(orphan_v_labels) == 0
        and len(orphan_ir_labels) == 0
    )
    complete = required_dirs_ok and counts_ok and exact_counts_ok and pairing_ok and min_size_ok

    state = {
        "root": str(root),
        "required_dirs_ok": required_dirs_ok,
        "video_v_count": len(videos_v),
        "video_ir_count": len(videos_ir),
        "label_v_count": len(labels_v),
        "label_ir_count": len(labels_ir),
        "audio_count": len(audio_files),
        "min_video_v": min_video_v,
        "min_video_ir": min_video_ir,
        "min_label_v": min_label_v,
        "min_label_ir": min_label_ir,
        "min_audio": min_audio,
        "expect_video_v": expect_video_v,
        "expect_video_ir": expect_video_ir,
        "expect_label_v": expect_label_v,
        "expect_label_ir": expect_label_ir,
        "expect_audio": expect_audio,
        "exact_count_checks": exact_count_checks,
        "exact_counts_ok": exact_counts_ok,
        "missing_video_v_labels": missing_video_v_labels[:30],
        "missing_video_ir_labels": missing_video_ir_labels[:30],
        "orphan_v_labels": orphan_v_labels[:30],
        "orphan_ir_labels": orphan_ir_labels[:30],
        "total_size_gb": round(float(total_size_gb), 3),
        "min_total_gb": float(min_total_gb),
        "min_size_ok": bool(min_size_ok),
        "complete": bool(complete),
    }
    return complete, state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/raw/dronedetect_thesis/Data"))
    parser.add_argument("--min-video-v", type=int, default=285)
    parser.add_argument("--min-video-ir", type=int, default=365)
    parser.add_argument("--min-label-v", type=int, default=285)
    parser.add_argument("--min-label-ir", type=int, default=365)
    parser.add_argument("--min-audio", type=int, default=90)
    parser.add_argument("--min-total-gb", type=float, default=40.0)
    parser.add_argument("--expect-video-v", type=int, default=None)
    parser.add_argument("--expect-video-ir", type=int, default=None)
    parser.add_argument("--expect-label-v", type=int, default=None)
    parser.add_argument("--expect-label-ir", type=int, default=None)
    parser.add_argument("--expect-audio", type=int, default=None)
    parser.add_argument("--out-txt", type=Path, default=Path("results/dronedetect_state.txt"))
    parser.add_argument("--out-json", type=Path, default=Path("results/dronedetect_state.json"))
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    complete, state = verify_dronedetect(
        root=args.root,
        min_video_v=args.min_video_v,
        min_video_ir=args.min_video_ir,
        min_label_v=args.min_label_v,
        min_label_ir=args.min_label_ir,
        min_audio=args.min_audio,
        min_total_gb=args.min_total_gb,
        expect_video_v=args.expect_video_v,
        expect_video_ir=args.expect_video_ir,
        expect_label_v=args.expect_label_v,
        expect_label_ir=args.expect_label_ir,
        expect_audio=args.expect_audio,
    )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_txt.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(state, indent=2), encoding="utf-8")

    lines = [
        "DroneDetectionThesis Verification",
        f"root={state['root']}",
        f"required_dirs_ok={state['required_dirs_ok']}",
        f"video_v_count={state['video_v_count']}",
        f"video_ir_count={state['video_ir_count']}",
        f"label_v_count={state['label_v_count']}",
        f"label_ir_count={state['label_ir_count']}",
        f"audio_count={state['audio_count']}",
        f"min_video_v={state['min_video_v']}",
        f"min_video_ir={state['min_video_ir']}",
        f"min_label_v={state['min_label_v']}",
        f"min_label_ir={state['min_label_ir']}",
        f"min_audio={state['min_audio']}",
        f"expect_video_v={state['expect_video_v']}",
        f"expect_video_ir={state['expect_video_ir']}",
        f"expect_label_v={state['expect_label_v']}",
        f"expect_label_ir={state['expect_label_ir']}",
        f"expect_audio={state['expect_audio']}",
        f"exact_count_checks={state['exact_count_checks']}",
        f"exact_counts_ok={state['exact_counts_ok']}",
        f"total_size_gb={state['total_size_gb']}",
        f"min_total_gb={state['min_total_gb']}",
        f"min_size_ok={state['min_size_ok']}",
    ]
    if complete:
        lines.append("DRONEDETECT_STATE=COMPLETE")
        lines.append("FULL_DRONEDETECT_PRESENT=YES")
        lines.append("COMPLETE")
    else:
        lines.append("DRONEDETECT_STATE=INCOMPLETE")
        lines.append("FULL_DRONEDETECT_PRESENT=NO")
        lines.append("INCOMPLETE")
        if state["missing_video_v_labels"]:
            lines.append("Missing labels for Video_V clips (first 30):")
            lines.extend(f"- {x}" for x in state["missing_video_v_labels"])
        if state["missing_video_ir_labels"]:
            lines.append("Missing labels for Video_IR clips (first 30):")
            lines.extend(f"- {x}" for x in state["missing_video_ir_labels"])
        if state["orphan_v_labels"]:
            lines.append("Orphan Video_V labels without matching video (first 30):")
            lines.extend(f"- {x}" for x in state["orphan_v_labels"])
        if state["orphan_ir_labels"]:
            lines.append("Orphan Video_IR labels without matching video (first 30):")
            lines.extend(f"- {x}" for x in state["orphan_ir_labels"])

    args.out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.out_txt.read_text(encoding="utf-8"))

    if not complete and not args.allow_incomplete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
