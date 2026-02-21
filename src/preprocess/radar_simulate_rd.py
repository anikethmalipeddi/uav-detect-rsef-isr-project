from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import radarsimpy  # noqa: F401

    HAS_RADARSIMPY = True
except Exception:
    HAS_RADARSIMPY = False


def synthetic_rd_map(size: int = 128, snr_db: float = 20.0, clutter_level: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng()
    base = rng.normal(0.0, 1.0, size=(size, size)).astype(np.float32)

    # Target blob.
    cx = rng.integers(low=10, high=size - 10)
    cy = rng.integers(low=10, high=size - 10)
    xx, yy = np.meshgrid(np.arange(size), np.arange(size))
    blob = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * 4.0**2))).astype(np.float32)
    rd = base + 8.0 * blob

    # Near-zero Doppler clutter band.
    c0 = size // 2
    band = int(size * 0.08)
    rd[:, c0 - band : c0 + band] += clutter_level * rng.normal(0.0, 1.0, size=(size, 2 * band)).astype(np.float32)

    # AWGN from requested SNR.
    signal_power = np.mean(rd**2) + 1e-6
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    rd += rng.normal(0.0, np.sqrt(noise_power), size=rd.shape).astype(np.float32)

    rd = (rd - rd.mean()) / (rd.std() + 1e-6)
    return rd.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--label", type=int, default=1)
    parser.add_argument("--prefix", type=str, default="sim")
    parser.add_argument("--manifest-name", type=str, default="manifest_sim.csv")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    rows = []
    for i in range(args.count):
        snr_db = float(rng.choice([30, 20, 10, 0, -10]))
        clutter_level = float(rng.choice([1, 2, 4]))
        rd = synthetic_rd_map(size=args.size, snr_db=snr_db, clutter_level=clutter_level)
        p = args.output / f"{args.prefix}_{i:07d}.npy"
        np.save(p, rd)
        rows.append(
            {
                "radar_id": i,
                "npy_path": str(p),
                "label_drone": int(args.label),
                "snr_db": snr_db,
                "clutter_level": clutter_level,
                "source": "radarsimpy" if HAS_RADARSIMPY else "synthetic_fallback",
            }
        )

    manifest = args.output / args.manifest_name
    pd.DataFrame(rows).to_csv(manifest, index=False)
    print(f"Wrote {len(rows)} simulated RD maps to {manifest}")


if __name__ == "__main__":
    main()
