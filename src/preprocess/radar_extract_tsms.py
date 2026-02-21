from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
import scipy.io as sio
from tqdm import tqdm

try:
    import mat73  # type: ignore
except Exception:  # noqa: BLE001
    mat73 = None

try:
    import h5py  # type: ignore
except Exception:  # noqa: BLE001
    h5py = None


def find_first_2d_array(obj: object) -> np.ndarray | None:
    if isinstance(obj, np.ndarray):
        if obj.ndim == 2 and np.issubdtype(obj.dtype, np.number):
            return obj
        if obj.ndim > 2 and np.issubdtype(obj.dtype, np.number):
            flat = obj.reshape((-1, obj.shape[-2], obj.shape[-1]))
            return flat[0]
        if obj.dtype == object:
            for v in obj.flat:
                found = find_first_2d_array(v)
                if found is not None:
                    return found
    if isinstance(obj, dict):
        for v in obj.values():
            found = find_first_2d_array(v)
            if found is not None:
                return found
    return None


def find_first_2d_h5(node: object) -> np.ndarray | None:
    if h5py is None:
        return None
    if isinstance(node, h5py.Dataset):
        try:
            arr = np.asarray(node[()])
            found = find_first_2d_array(arr)
            if found is not None:
                return found
        except Exception:  # noqa: BLE001
            return None
        return None
    if isinstance(node, h5py.Group):
        for k in node.keys():
            found = find_first_2d_h5(node[k])
            if found is not None:
                return found
    return None


def load_radar_array(path: Path) -> np.ndarray | None:
    ext = path.suffix.lower()
    try:
        if ext == ".npy":
            arr = np.load(path)
            if arr.ndim == 2:
                return arr
            if arr.ndim > 2:
                return arr.reshape((-1, arr.shape[-2], arr.shape[-1]))[0]
            return None
        if ext in {".png", ".jpg", ".jpeg", ".bmp"}:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None
            return img.astype(np.float32)
        if ext == ".mat":
            try:
                d = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)
                for k, v in d.items():
                    if k.startswith("__"):
                        continue
                    found = find_first_2d_array(v)
                    if found is not None:
                        return found.astype(np.float32)
            except NotImplementedError:
                pass
            except Exception:  # noqa: BLE001
                pass

            if mat73 is not None:
                try:
                    d73 = mat73.loadmat(str(path))
                    found = find_first_2d_array(d73)
                    if found is not None:
                        return found.astype(np.float32)
                except Exception:  # noqa: BLE001
                    pass

            if h5py is not None:
                try:
                    with h5py.File(str(path), "r") as f:
                        for name in f.keys():
                            found = find_first_2d_h5(f[name])
                            if found is not None:
                                return found.astype(np.float32)
                except Exception:  # noqa: BLE001
                    pass
    except Exception:  # noqa: BLE001
        return None
    return None


def normalize_and_resize(arr: np.ndarray, size: int = 128) -> np.ndarray:
    arr = arr.astype(np.float32)
    if arr.shape != (size, size):
        arr = cv2.resize(arr, (size, size), interpolation=cv2.INTER_LINEAR)
    mean = float(arr.mean())
    std = float(arr.std() + 1e-6)
    arr = (arr - mean) / std
    return arr.astype(np.float32)


def radar_quality(arr: np.ndarray) -> tuple[float, float, float]:
    h, w = arr.shape
    center = w // 2
    band = max(1, int(0.1 * w))
    clutter = np.mean(np.abs(arr[:, center - band : center + band]))
    total = np.mean(np.abs(arr)) + 1e-6
    clutter_ratio = float(clutter / total)

    peak = float(np.max(np.abs(arr)))
    noise_floor = float(np.median(np.abs(arr)) + 1e-6)
    peak_to_noise = float(peak / noise_floor)
    energy = float(np.mean(arr**2))
    return clutter_ratio, peak_to_noise, energy


def process_tree(
    input_root: Path,
    output_root: Path,
    out_size: int = 128,
    max_files: int = 0,
    extensions: set[str] | None = None,
    dataset_only: bool = True,
) -> pd.DataFrame:
    output_root.mkdir(parents=True, exist_ok=True)
    if extensions is None:
        extensions = {".npy", ".png", ".jpg", ".jpeg", ".bmp", ".mat"}

    def is_allowed_path(p: Path) -> bool:
        if p.suffix.lower() not in extensions:
            return False
        if not dataset_only:
            return True
        low = str(p).replace("/", "\\").lower()
        # Use only TSMS dataset sensor files; exclude sample/reconstruction scripts.
        return ("\\dataset\\" in low) and ("\\sample code\\" not in low)

    files = [p for p in input_root.rglob("*") if is_allowed_path(p)]
    if max_files > 0:
        files = files[:max_files]
    rows = []
    idx = 0
    for p in tqdm(files, desc="Radar extract"):
        arr = load_radar_array(p)
        if arr is None:
            continue
        arr = normalize_and_resize(arr, size=out_size)
        q0, q1, q2 = radar_quality(arr)
        out_path = output_root / f"rd_{idx:08d}.npy"
        np.save(out_path, arr)
        rows.append(
            {
                "radar_id": idx,
                "source_path": str(p),
                "npy_path": str(out_path),
                "q_clutter_ratio": q0,
                "q_peak_to_noise": q1,
                "q_energy": q2,
            }
        )
        idx += 1
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument(
        "--extensions",
        type=str,
        default=".npy,.png,.jpg,.jpeg,.bmp,.mat",
        help="Comma-separated extensions, e.g. .png,.jpg",
    )
    parser.add_argument(
        "--include-non-dataset",
        action="store_true",
        help="Include non-dataset paths (e.g., sample code). Disabled by default for strict runs.",
    )
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/radar_rd/manifest.csv"))
    args = parser.parse_args()

    exts = {e.strip().lower() for e in args.extensions.split(",") if e.strip()}
    df = process_tree(
        args.input,
        args.output,
        out_size=args.size,
        max_files=args.max_files,
        extensions=exts,
        dataset_only=not args.include_non_dataset,
    )
    if df.empty:
        raise RuntimeError(f"No radar files were extracted from {args.input}.")
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.manifest, index=False)
    print(f"Wrote {len(df)} rows to {args.manifest}")


if __name__ == "__main__":
    main()
