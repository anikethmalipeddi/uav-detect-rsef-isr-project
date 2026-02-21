from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd

import matio
import scipy.io as sio
from matio.subsystem.subsys import MatSubsystem
from matio.utils.matclass import MatlabOpaque

try:
    import mat73  # type: ignore
except Exception:  # noqa: BLE001
    mat73 = None

try:
    import h5py  # type: ignore
except Exception:  # noqa: BLE001
    h5py = None


CLASS_NAMES = ["AIRPLANE", "BIRD", "DRONE", "HELICOPTER"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASS_NAMES)}
VIDEO_EXT = [".mp4", ".avi", ".mov", ".mkv"]


def patch_matio_subsystem() -> None:
    """Patch mat-io for a known None-handling issue in check_unknowns()."""
    if getattr(MatSubsystem, "_uavmm_patched", False):
        return

    original = MatSubsystem.check_unknowns

    def _safe_check_unknowns(self: MatSubsystem, cell2: object) -> None:
        if self._u6_metadata is None:
            self._u6_metadata = np.empty(0, dtype=np.uint8)
        if self._u7_metadata is None:
            self._u7_metadata = np.empty(0, dtype=np.uint8)
        original(self, cell2)

    MatSubsystem.check_unknowns = _safe_check_unknowns  # type: ignore[assignment]
    MatSubsystem._uavmm_patched = True  # type: ignore[attr-defined]


def stable_split(key: str) -> str:
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()
    bucket = int(h[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "val"
    return "test"


def resolve_video_path(mat_path: Path) -> Path:
    base = mat_path.stem.replace("_LABELS", "")
    for ext in VIDEO_EXT:
        candidate = mat_path.with_name(base + ext)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve video for label file: {mat_path}")


def to_string_scalar(x: object) -> str:
    if isinstance(x, np.ndarray):
        if x.size == 0:
            return ""
        return str(x.flat[0])
    return str(x)


@dataclass
class VideoLabelParseResult:
    video_id: str
    video_path: str
    label_path: str
    split: str
    frame_count: int
    width: int
    height: int
    fps: float
    label_rows: int
    total_boxes: int
    boxes: list[dict]


def _parse_ground_truth_matio(mat_path: Path) -> tuple[np.ndarray, np.ndarray]:
    patch_matio_subsystem()
    data = matio.load_from_mat(str(mat_path), raw_data=False)
    if "gTruth" not in data:
        raise ValueError(f"gTruth variable not found in {mat_path}")
    gt = data["gTruth"]
    if not isinstance(gt, MatlabOpaque):
        raise ValueError(f"gTruth is not MatlabOpaque in {mat_path}")
    if not isinstance(gt.properties, dict) or "any" not in gt.properties:
        raise ValueError(f"gTruth.any missing in {mat_path}")

    any_struct = gt.properties["any"]
    if not isinstance(any_struct, np.ndarray) or any_struct.size == 0 or any_struct.dtype.names is None:
        raise ValueError(f"gTruth.any has unexpected structure in {mat_path}")
    required = {"DataSource", "LabelDefinitions", "LabelData"}
    if not required.issubset(set(any_struct.dtype.names)):
        raise ValueError(f"gTruth.any missing fields {sorted(required)} in {mat_path}")

    cell = any_struct.flat[0]
    label_defs = cell["LabelDefinitions"]
    label_data = cell["LabelData"]
    if not isinstance(label_defs, np.ndarray) or label_defs.dtype.names is None:
        raise ValueError(f"LabelDefinitions malformed in {mat_path}")
    if not isinstance(label_data, np.ndarray) or label_data.dtype.names is None:
        raise ValueError(f"LabelData malformed in {mat_path}")
    return label_defs, label_data


def _to_string(x: object) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore")
    if isinstance(x, np.ndarray):
        if x.size == 0:
            return ""
        return _to_string(x.flat[0])
    return str(x)


def _to_list(x: object) -> list:
    if isinstance(x, np.ndarray):
        return list(x.flat)
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _collect_named_values(obj: object, key_name: str) -> list[object]:
    out: list[object] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k) == key_name:
                out.append(v)
            out.extend(_collect_named_values(v, key_name))
    elif isinstance(obj, np.ndarray):
        if obj.dtype.names is not None and key_name in obj.dtype.names:
            out.extend(_to_list(obj[key_name]))
        for v in obj.flat:
            out.extend(_collect_named_values(v, key_name))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out.extend(_collect_named_values(v, key_name))
    elif hasattr(obj, "__dict__"):
        out.extend(_collect_named_values(vars(obj), key_name))
    return out


def _find_mapping_with_keys(obj: object, keys: set[str]) -> dict | None:
    if isinstance(obj, dict):
        if keys.issubset(set(obj.keys())):
            return obj
        for v in obj.values():
            found = _find_mapping_with_keys(v, keys)
            if found is not None:
                return found
    elif isinstance(obj, np.ndarray):
        for v in obj.flat:
            found = _find_mapping_with_keys(v, keys)
            if found is not None:
                return found
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            found = _find_mapping_with_keys(v, keys)
            if found is not None:
                return found
    elif hasattr(obj, "__dict__"):
        return _find_mapping_with_keys(vars(obj), keys)
    return None


def _build_label_arrays_from_records(class_names: list[str], frame_records: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray]:
    if not class_names:
        raise ValueError("No class names found in fallback MAT parser.")
    n_frames = len(frame_records)
    if n_frames <= 0:
        raise ValueError("No frame label records found in fallback MAT parser.")

    defs_dtype = np.dtype([("Name", object)])
    label_defs = np.empty((len(class_names), 1), dtype=defs_dtype)
    for i, cname in enumerate(class_names):
        label_defs[i, 0]["Name"] = cname

    data_dtype = np.dtype([(cname, object) for cname in class_names])
    label_data = np.empty((n_frames, 1), dtype=data_dtype)
    for i, rec in enumerate(frame_records):
        for cname in class_names:
            label_data[i, 0][cname] = rec.get(cname, np.empty((0, 4), dtype=np.float32))
    return label_defs, label_data


def _extract_generic_gtruth_to_arrays(obj: object) -> tuple[np.ndarray, np.ndarray]:
    required = {"LabelDefinitions", "LabelData"}
    g = _find_mapping_with_keys(obj, required)
    if g is None:
        raise ValueError("Could not find LabelDefinitions/LabelData in generic gTruth structure.")

    label_defs_obj = g["LabelDefinitions"]
    label_data_obj = g["LabelData"]
    raw_names = _collect_named_values(label_defs_obj, "Name")
    class_names: list[str] = []
    for n in raw_names:
        s = _to_string(n).strip().upper()
        if s and s not in class_names:
            class_names.append(s)
    if not class_names:
        # conservative fallback for expected classes
        class_names = CLASS_NAMES.copy()

    frame_records: list[dict[str, object]] = []
    if isinstance(label_data_obj, dict):
        # common shape: dict[class_name] -> sequence over frames
        keys = [k for k in class_names if k in label_data_obj]
        if not keys:
            keys = [k for k in label_data_obj.keys() if str(k).upper() in class_names]
        if not keys:
            raise ValueError("LabelData dict does not contain expected class keys.")
        seqs = {str(k).upper(): _to_list(label_data_obj[k]) for k in keys}
        n = max(len(v) for v in seqs.values())
        for i in range(n):
            rec: dict[str, object] = {}
            for k, seq in seqs.items():
                if i < len(seq):
                    rec[k] = seq[i]
            frame_records.append(rec)
    elif isinstance(label_data_obj, (list, tuple, np.ndarray)):
        rows = _to_list(label_data_obj)
        for row in rows:
            rec: dict[str, object] = {}
            if isinstance(row, dict):
                for cname in class_names:
                    if cname in row:
                        rec[cname] = row[cname]
                    elif cname.lower() in row:
                        rec[cname] = row[cname.lower()]
            elif isinstance(row, np.void) and row.dtype.names is not None:
                for cname in class_names:
                    if cname in row.dtype.names:
                        rec[cname] = row[cname]
            frame_records.append(rec)
    else:
        raise ValueError("Unsupported LabelData structure in generic gTruth parser.")

    return _build_label_arrays_from_records(class_names, frame_records)


def _h5_to_python(node: object, file_obj: object) -> object:
    if h5py is None:
        raise RuntimeError("h5py is not installed")
    if isinstance(node, h5py.Dataset):
        arr = node[()]
        if isinstance(arr, np.ndarray) and arr.dtype == object:
            vals = []
            for ref in arr.flat:
                if isinstance(ref, h5py.Reference):
                    vals.append(_h5_to_python(file_obj[ref], file_obj))
                else:
                    vals.append(ref)
            return vals
        if isinstance(arr, np.ndarray) and arr.dtype.kind in {"S", "U"}:
            return [_to_string(x) for x in arr.flat]
        return arr
    if isinstance(node, h5py.Group):
        return {k: _h5_to_python(node[k], file_obj) for k in node.keys()}
    return node


def parse_ground_truth_mat(mat_path: Path) -> tuple[np.ndarray, np.ndarray]:
    errors: list[str] = []
    try:
        return _parse_ground_truth_matio(mat_path)
    except Exception as e:  # noqa: BLE001
        errors.append(f"matio: {e}")

    # scipy supports classic MAT files and gives a useful error for v7.3.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = sio.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
        g = d.get("gTruth", None)
        if g is not None:
            return _extract_generic_gtruth_to_arrays(g)
    except Exception as e:  # noqa: BLE001
        errors.append(f"scipy: {e}")

    # Optional v7.3 parser via mat73 if installed.
    if mat73 is not None:
        try:
            d = mat73.loadmat(str(mat_path))
            g = d.get("gTruth", d)
            return _extract_generic_gtruth_to_arrays(g)
        except Exception as e:  # noqa: BLE001
            errors.append(f"mat73: {e}")
    else:
        errors.append("mat73: not installed")

    # Fallback generic v7.3 parsing via h5py.
    if h5py is not None:
        try:
            with h5py.File(str(mat_path), "r") as f:
                payload = {k: _h5_to_python(f[k], f) for k in f.keys()}
            g = payload.get("gTruth", payload)
            return _extract_generic_gtruth_to_arrays(g)
        except Exception as e:  # noqa: BLE001
            errors.append(f"h5py: {e}")
    else:
        errors.append("h5py: not installed")

    joined = " | ".join(errors)
    raise RuntimeError(f"Failed to parse MAT labels for {mat_path}. Parsers attempted: {joined}")


def normalize_boxes(arr: object) -> np.ndarray:
    if isinstance(arr, np.void):
        if arr.dtype.names is None:
            return np.empty((0, 4), dtype=np.float32)
        if "Position" in arr.dtype.names:
            return normalize_boxes(arr["Position"])
        parts = []
        for name in arr.dtype.names:
            part = normalize_boxes(arr[name])
            if part.size:
                parts.append(part)
        if parts:
            return np.concatenate(parts, axis=0).astype(np.float32)
        return np.empty((0, 4), dtype=np.float32)

    if isinstance(arr, (list, tuple)):
        parts = []
        for item in arr:
            part = normalize_boxes(item)
            if part.size:
                parts.append(part)
        if parts:
            return np.concatenate(parts, axis=0).astype(np.float32)
        return np.empty((0, 4), dtype=np.float32)

    if not isinstance(arr, np.ndarray):
        return np.empty((0, 4), dtype=np.float32)
    if arr.size == 0:
        return np.empty((0, 4), dtype=np.float32)
    if arr.dtype.names is not None:
        # Some files store boxes as struct with fields: Position, LabelUIDs.
        if "Position" in arr.dtype.names:
            parts = []
            for item in arr.flat:
                part = normalize_boxes(np.asarray(item["Position"]))
                if part.size:
                    parts.append(part)
            if parts:
                return np.concatenate(parts, axis=0).astype(np.float32)
            return np.empty((0, 4), dtype=np.float32)

        parts = []
        for name in arr.dtype.names:
            part = normalize_boxes(np.asarray(arr[name]))
            if part.size:
                parts.append(part)
        if parts:
            return np.concatenate(parts, axis=0).astype(np.float32)
        return np.empty((0, 4), dtype=np.float32)

    if arr.dtype == object:
        parts = []
        for item in arr.flat:
            part = normalize_boxes(np.asarray(item))
            if part.size:
                parts.append(part)
        if parts:
            return np.concatenate(parts, axis=0).astype(np.float32)
        return np.empty((0, 4), dtype=np.float32)

    try:
        numeric = np.asarray(arr, dtype=np.float32)
    except (TypeError, ValueError):
        try:
            numeric = np.asarray(arr.tolist(), dtype=np.float32)
        except Exception:  # noqa: BLE001
            return np.empty((0, 4), dtype=np.float32)

    if numeric.ndim == 1:
        if numeric.size == 4:
            return numeric.reshape(1, 4).astype(np.float32)
        if numeric.size % 4 == 0:
            return numeric.reshape(-1, 4).astype(np.float32)
        return np.empty((0, 4), dtype=np.float32)

    if numeric.ndim >= 2:
        if numeric.shape[-1] == 4:
            return numeric.reshape(-1, 4).astype(np.float32)
        if numeric.size % 4 == 0:
            return numeric.reshape(-1, 4).astype(np.float32)
    return np.empty((0, 4), dtype=np.float32)


def parse_video_labels(mat_path: Path) -> VideoLabelParseResult:
    video_path = resolve_video_path(mat_path)
    video_id = video_path.stem
    split = stable_split(video_id)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if frame_count <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video metadata for {video_path} (frames={frame_count}, w={width}, h={height})")

    label_defs, label_data = parse_ground_truth_mat(mat_path)
    class_names: list[str] = []
    for i in range(label_defs.shape[0]):
        rec = label_defs[i, 0] if label_defs.ndim == 2 else label_defs[i]
        cname = to_string_scalar(rec["Name"]).upper()
        class_names.append(cname)

    boxes: list[dict] = []
    label_rows = int(label_data.shape[0])
    for frame_idx in range(label_rows):
        rec = label_data[frame_idx, 0] if label_data.ndim == 2 else label_data[frame_idx]
        for cname in class_names:
            if cname not in label_data.dtype.names:
                continue
            b = normalize_boxes(rec[cname])
            if b.size == 0:
                continue
            for row in b:
                boxes.append(
                    {
                        "video_id": video_id,
                        "video_path": str(video_path),
                        "label_path": str(mat_path),
                        "split": split,
                        "frame_idx": int(frame_idx),
                        "class_name": cname,
                        "x": float(row[0]),
                        "y": float(row[1]),
                        "w": float(row[2]),
                        "h": float(row[3]),
                        "frame_w": int(width),
                        "frame_h": int(height),
                    }
                )

    return VideoLabelParseResult(
        video_id=video_id,
        video_path=str(video_path),
        label_path=str(mat_path),
        split=split,
        frame_count=frame_count,
        width=width,
        height=height,
        fps=fps,
        label_rows=label_rows,
        total_boxes=len(boxes),
        boxes=boxes,
    )


def validate_boxes(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"total_boxes": 0, "invalid_boxes": 0, "invalid_fraction": 1.0}

    x0 = df["x"].to_numpy(dtype=np.float32)
    y0 = df["y"].to_numpy(dtype=np.float32)
    w = df["w"].to_numpy(dtype=np.float32)
    h = df["h"].to_numpy(dtype=np.float32)
    x1 = x0 + w
    y1 = y0 + h
    fw = df["frame_w"].to_numpy(dtype=np.float32)
    fh = df["frame_h"].to_numpy(dtype=np.float32)

    invalid = (
        (w <= 0)
        | (h <= 0)
        | (x0 < 0)
        | (y0 < 0)
        | (x1 > fw)
        | (y1 > fh)
    )
    n_invalid = int(np.sum(invalid))
    total = int(len(df))
    return {
        "total_boxes": total,
        "invalid_boxes": n_invalid,
        "invalid_fraction": float(n_invalid / max(1, total)),
    }


def draw_overlays(
    labels_df: pd.DataFrame,
    out_dir: Path,
    n_samples: int = 40,
    seed: int = 42,
) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if labels_df.empty:
        return []

    rng = np.random.default_rng(seed)
    unique_frames = labels_df[["video_id", "video_path", "frame_idx", "class_name"]].drop_duplicates()
    n = min(n_samples, len(unique_frames))
    sample_idx = rng.choice(len(unique_frames), size=n, replace=False)
    samples = unique_frames.iloc[sample_idx].reset_index(drop=True)

    outputs: list[str] = []
    for _, row in samples.iterrows():
        video_path = Path(row["video_path"])
        frame_idx = int(row["frame_idx"])
        video_id = str(row["video_id"])
        class_name = str(row["class_name"])
        frame_boxes = labels_df[
            (labels_df["video_id"] == video_id)
            & (labels_df["frame_idx"] == frame_idx)
        ]
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            continue
        for _, b in frame_boxes.iterrows():
            x = int(round(float(b["x"])))
            y = int(round(float(b["y"])))
            w = int(round(float(b["w"])))
            h = int(round(float(b["h"])))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (20, 240, 20), 2)
            cv2.putText(
                frame,
                str(b["class_name"]),
                (x, max(12, y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (20, 240, 20),
                1,
                cv2.LINE_AA,
            )
        out_path = out_dir / f"{video_id}_f{frame_idx:06d}_{class_name}.png"
        cv2.imwrite(str(out_path), frame)
        outputs.append(str(out_path))
    return outputs


def yolo_line(class_name: str, x: float, y: float, w: float, h: float, iw: int, ih: int) -> str:
    cls = CLASS_TO_ID[class_name]
    cx = (x + w / 2.0) / float(iw)
    cy = (y + h / 2.0) / float(ih)
    nw = w / float(iw)
    nh = h / float(ih)
    return f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def build_yolo_dataset(
    video_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    out_root: Path,
    frame_step: int = 3,
) -> dict:
    if frame_step < 1:
        raise ValueError("frame_step must be >= 1")
    for split in ["train", "val", "test"]:
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    index_rows: list[dict] = []
    total_images = 0
    total_positive = 0

    grouped = labels_df.groupby(["video_id", "frame_idx"])
    for _, v in video_df.iterrows():
        video_id = str(v["video_id"])
        video_path = str(v["video_path"])
        split = str(v["split"])
        frame_count = int(v["frame_count"])
        width = int(v["width"])
        height = int(v["height"])

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video for YOLO export: {video_path}")

        for frame_idx in range(0, frame_count, frame_step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            img_name = f"{video_id}_f{frame_idx:06d}.jpg"
            img_path = out_root / "images" / split / img_name
            lbl_path = out_root / "labels" / split / img_name.replace(".jpg", ".txt")
            cv2.imwrite(str(img_path), frame)

            has_box = False
            lines: list[str] = []
            key = (video_id, frame_idx)
            if key in grouped.groups:
                gdf = grouped.get_group(key)
                for _, b in gdf.iterrows():
                    cname = str(b["class_name"]).upper()
                    if cname not in CLASS_TO_ID:
                        continue
                    lines.append(
                        yolo_line(
                            cname,
                            float(b["x"]),
                            float(b["y"]),
                            float(b["w"]),
                            float(b["h"]),
                            iw=width,
                            ih=height,
                        )
                    )
                has_box = len(lines) > 0
            lbl_path.write_text("\n".join(lines), encoding="utf-8")

            total_images += 1
            total_positive += int(has_box)
            index_rows.append(
                {
                    "video_id": video_id,
                    "video_path": video_path,
                    "split": split,
                    "frame_idx": frame_idx,
                    "image_path": str(img_path),
                    "label_path": str(lbl_path),
                    "has_box": int(has_box),
                }
            )
        cap.release()

    dataset_yaml = (
        f"path: {out_root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"names: {CLASS_NAMES}\n"
    )
    (out_root / "dataset.yaml").write_text(dataset_yaml, encoding="utf-8")
    pd.DataFrame(index_rows).to_csv(out_root / "index.csv", index=False)

    if total_images <= 0 or total_positive <= 0:
        raise RuntimeError(
            f"YOLO export failed: total_images={total_images}, total_positive={total_positive}."
        )
    return {
        "total_images": int(total_images),
        "total_positive": int(total_positive),
        "positive_fraction": float(total_positive / max(1, total_images)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat-root", required=True, type=Path)
    parser.add_argument("--out-label-csv", type=Path, default=Path("data/processed/vision_labels/labels_long.csv"))
    parser.add_argument("--out-label-json", type=Path, default=Path("data/processed/vision_labels/labels_long.jsonl"))
    parser.add_argument("--out-video-csv", type=Path, default=Path("data/processed/vision_labels/videos.csv"))
    parser.add_argument("--out-stats", type=Path, default=Path("results/label_stats.json"))
    parser.add_argument("--out-stats-txt", type=Path, default=Path("results/label_stats.txt"))
    parser.add_argument("--overlay-dir", type=Path, default=Path("results/label_viz"))
    parser.add_argument("--overlay-samples", type=int, default=40)
    parser.add_argument("--yolo-out", type=Path, default=Path("data/processed/vision_yolo"))
    parser.add_argument("--yolo-frame-step", type=int, default=3)
    parser.add_argument("--max-mats", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    mats = sorted(args.mat_root.rglob("*_LABELS.mat"))
    if args.max_mats > 0:
        mats = mats[: args.max_mats]
    if not mats:
        raise FileNotFoundError(f"No *_LABELS.mat files found under {args.mat_root}")

    parsed: list[VideoLabelParseResult] = []
    errors: list[dict] = []
    for mat_path in mats:
        try:
            parsed.append(parse_video_labels(mat_path))
        except Exception as e:  # noqa: BLE001
            errors.append({"label_path": str(mat_path), "error": str(e)})

    if errors:
        err_path = Path("results/label_parse_errors.json")
        err_path.parent.mkdir(parents=True, exist_ok=True)
        err_path.write_text(json.dumps(errors, indent=2), encoding="utf-8")
        raise RuntimeError(f"Label parsing failed for {len(errors)} files. See {err_path}")

    video_rows = [
        {
            "video_id": p.video_id,
            "video_path": p.video_path,
            "label_path": p.label_path,
            "split": p.split,
            "frame_count": p.frame_count,
            "width": p.width,
            "height": p.height,
            "fps": p.fps,
            "label_rows": p.label_rows,
            "total_boxes": p.total_boxes,
        }
        for p in parsed
    ]
    all_boxes = [b for p in parsed for b in p.boxes]

    videos_df = pd.DataFrame(video_rows)
    labels_df = pd.DataFrame(all_boxes)
    if labels_df.empty:
        raise RuntimeError("Parsed labels are empty. Refusing to continue.")

    quality = validate_boxes(labels_df)
    if quality["invalid_fraction"] > 0.05:
        raise RuntimeError(
            f"Too many invalid boxes ({quality['invalid_boxes']}/{quality['total_boxes']})."
        )

    overlays = draw_overlays(labels_df, args.overlay_dir, n_samples=args.overlay_samples, seed=args.seed)
    if len(overlays) == 0:
        raise RuntimeError("No overlay images were produced. Label validation failed.")

    yolo_stats = build_yolo_dataset(videos_df, labels_df, out_root=args.yolo_out, frame_step=args.yolo_frame_step)

    labels_out = labels_df.copy()
    labels_out["clip_id"] = labels_out["video_id"].astype(str)
    labels_out["frame_id"] = labels_out["frame_idx"].astype(int)
    labels_out["class_id"] = labels_out["class_name"].map(CLASS_TO_ID).astype(int)
    labels_out["bbox_x1"] = labels_out["x"].astype(float)
    labels_out["bbox_y1"] = labels_out["y"].astype(float)
    labels_out["bbox_x2"] = (labels_out["x"] + labels_out["w"]).astype(float)
    labels_out["bbox_y2"] = (labels_out["y"] + labels_out["h"]).astype(float)
    labels_out["bbox_xyxy"] = labels_out.apply(
        lambda r: [float(r["bbox_x1"]), float(r["bbox_y1"]), float(r["bbox_x2"]), float(r["bbox_y2"])], axis=1
    )
    labels_out = labels_out[
        [
            "clip_id",
            "frame_id",
            "bbox_x1",
            "bbox_y1",
            "bbox_x2",
            "bbox_y2",
            "class_id",
            "class_name",
            "video_path",
            "label_path",
            "split",
            "frame_w",
            "frame_h",
            "bbox_xyxy",
        ]
    ]

    args.out_label_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_label_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_video_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_stats.parent.mkdir(parents=True, exist_ok=True)
    args.out_stats_txt.parent.mkdir(parents=True, exist_ok=True)
    labels_out.to_csv(args.out_label_csv, index=False)
    with args.out_label_json.open("w", encoding="utf-8") as f:
        for rec in labels_out.to_dict(orient="records"):
            f.write(json.dumps(rec) + "\n")
    videos_df.to_csv(args.out_video_csv, index=False)

    class_counts = labels_df["class_name"].value_counts().to_dict()
    split_counts = videos_df["split"].value_counts().to_dict()
    clip_labeled_frames = (
        labels_df[["video_id", "frame_idx"]]
        .drop_duplicates()
        .groupby("video_id", as_index=False)
        .size()
        .rename(columns={"size": "frames_with_boxes"})
    )
    clip_summary = videos_df.merge(clip_labeled_frames, on="video_id", how="left")
    clip_summary["total_boxes"] = clip_summary["total_boxes"].fillna(0).astype(int)
    clip_summary["frames_with_boxes"] = clip_summary["frames_with_boxes"].fillna(0).astype(int)
    clip_summary["label_rows"] = clip_summary["label_rows"].fillna(0).astype(int)
    clip_summary["empty_label_rows"] = (clip_summary["label_rows"] - clip_summary["frames_with_boxes"]).clip(lower=0)
    total_label_rows = int(clip_summary["label_rows"].sum())
    total_empty_rows = int(clip_summary["empty_label_rows"].sum())
    empty_rate = float(total_empty_rows / max(1, total_label_rows))
    clip_summary_path = Path("results/label_per_clip_summary.csv")
    clip_summary_path.parent.mkdir(parents=True, exist_ok=True)
    clip_summary[
        [
            "video_id",
            "split",
            "frame_count",
            "label_rows",
            "frames_with_boxes",
            "empty_label_rows",
            "total_boxes",
        ]
    ].to_csv(clip_summary_path, index=False)

    stats = {
        "parsed_label_files": int(len(parsed)),
        "videos": int(len(videos_df)),
        "boxes": int(len(labels_df)),
        "class_counts": class_counts,
        "video_split_counts": split_counts,
        "box_quality": quality,
        "overlay_count": int(len(overlays)),
        "overlay_dir": str(args.overlay_dir),
        "total_label_rows": total_label_rows,
        "empty_label_rows": total_empty_rows,
        "empty_label_rate": empty_rate,
        "per_clip_summary_csv": str(clip_summary_path),
        "yolo": yolo_stats,
        "yolo_dataset_yaml": str(args.yolo_out / "dataset.yaml"),
        "label_csv": str(args.out_label_csv),
        "label_jsonl": str(args.out_label_json),
        "video_csv": str(args.out_video_csv),
    }
    args.out_stats.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    txt_lines = [
        "Label Conversion Stats",
        f"mat_root={args.mat_root}",
        f"parsed_label_files={stats['parsed_label_files']}",
        f"videos={stats['videos']}",
        f"boxes={stats['boxes']}",
        f"invalid_boxes={stats['box_quality']['invalid_boxes']}",
        f"invalid_fraction={stats['box_quality']['invalid_fraction']}",
        f"total_label_rows={stats['total_label_rows']}",
        f"empty_label_rows={stats['empty_label_rows']}",
        f"empty_label_rate={stats['empty_label_rate']:.6f}",
        f"overlay_count={stats['overlay_count']}",
        f"overlay_dir={stats['overlay_dir']}",
        f"per_clip_summary_csv={stats['per_clip_summary_csv']}",
        f"label_csv={stats['label_csv']}",
        f"label_jsonl={stats['label_jsonl']}",
        f"video_csv={stats['video_csv']}",
        f"yolo_dataset_yaml={stats['yolo_dataset_yaml']}",
    ]
    txt_lines.append("Class counts:")
    for k, v in stats["class_counts"].items():
        txt_lines.append(f"- {k}: {v}")
    txt_lines.append("Video split counts:")
    for k, v in stats["video_split_counts"].items():
        txt_lines.append(f"- {k}: {v}")
    txt_lines.append("Per-clip summary sample (top 20 by boxes):")
    for _, row in clip_summary.sort_values(["total_boxes", "video_id"], ascending=[False, True]).head(20).iterrows():
        txt_lines.append(
            "- "
            + f"{row['video_id']} split={row['split']} boxes={int(row['total_boxes'])} "
            + f"label_rows={int(row['label_rows'])} frames_with_boxes={int(row['frames_with_boxes'])} "
            + f"empty_label_rows={int(row['empty_label_rows'])}"
        )
    args.out_stats_txt.write_text("\n".join(txt_lines) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
