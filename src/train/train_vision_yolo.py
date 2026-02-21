from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/processed/vision_yolo/dataset.yaml"))
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", type=str, default="uavmm_yolov8n")
    parser.add_argument("--out-metrics", type=Path, default=Path("results/metrics_vision.json"))
    args = parser.parse_args()

    if YOLO is None:
        raise RuntimeError("Ultralytics is not installed. Install `ultralytics` first.")
    if not args.data.exists():
        raise FileNotFoundError(f"YOLO dataset config not found: {args.data}")

    model = YOLO(args.model)
    train_results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=args.device,
        seed=args.seed,
        project="runs/detect",
        name=args.run_name,
        exist_ok=True,
    )
    metrics = model.val(data=str(args.data), imgsz=args.imgsz, device=args.device)
    save_dir = getattr(train_results, "save_dir", None)
    if save_dir is None and getattr(model, "trainer", None) is not None:
        save_dir = getattr(model.trainer, "save_dir", None)
    if save_dir is None:
        save_dir = Path("runs/detect") / args.run_name
    else:
        save_dir = Path(str(save_dir))
    weights_path = save_dir / "weights" / "best.pt"

    out = {
        "map50_95": float(metrics.box.map),
        "map50": float(metrics.box.map50),
        "run_name": args.run_name,
        "weights_path": str(weights_path),
        "save_dir": str(save_dir),
    }
    args.out_metrics.parent.mkdir(parents=True, exist_ok=True)
    import json

    args.out_metrics.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {args.out_metrics}")


if __name__ == "__main__":
    main()
