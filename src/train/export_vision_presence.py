from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None


def logit_from_prob(p: float, eps: float = 1e-4) -> float:
    p = float(np.clip(p, eps, 1.0 - eps))
    return float(np.log(p / (1.0 - p)))


def frame_quality(frame: np.ndarray) -> tuple[float, float, float]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    contrast = float(gray.std())
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return brightness, contrast, blur


def run_video_yolo(video_path: Path, t0: float, t1: float, model: object | None) -> tuple[float, np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0.0, np.zeros((5,), dtype=np.float32)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    start = int(max(0, round(t0 * fps)))
    end = int(max(start + 1, round(t1 * fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    probs = []
    q = []
    idx = start
    while idx < end:
        ok, frame = cap.read()
        if not ok:
            break
        b, c, bl = frame_quality(frame)
        conf_max = 0.0
        conf_mean = 0.0
        if model is not None:
            res = model.predict(frame, verbose=False, conf=0.05)
            dets = res[0].boxes
            if dets is not None and len(dets) > 0:
                conf = dets.conf.detach().cpu().numpy()
                cls = dets.cls.detach().cpu().numpy().astype(int)
                names = res[0].names
                drone_conf = [float(conf[i]) for i in range(len(conf)) if "drone" in str(names.get(int(cls[i]), "")).lower()]
                if not drone_conf:
                    drone_conf = [float(c_) for c_ in conf]
                conf_max = float(max(drone_conf))
                conf_mean = float(np.mean(drone_conf))
        else:
            # Fallback heuristic when YOLO is not installed.
            conf_max = 0.8 if "drone" in str(video_path).lower() else 0.2
            conf_mean = conf_max
        probs.append(conf_max)
        q.append([b, c, bl, conf_mean, conf_max])
        idx += 1

    cap.release()
    if not probs:
        return 0.0, np.zeros((5,), dtype=np.float32)
    return float(np.max(probs)), np.mean(np.array(q, dtype=np.float32), axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, default=Path("data/processed/episodes/episodes.csv"))
    parser.add_argument("--model", type=str, default="results/checkpoints/best.pt")
    parser.add_argument("--out", type=Path, default=Path("data/processed/episodes/vision_presence.csv"))
    parser.add_argument("--allow-fallback", action="store_true")
    parser.add_argument("--warning-file", type=Path, default=Path("results/fallback_warnings.txt"))
    args = parser.parse_args()

    df = pd.read_csv(args.episodes, dtype={"episode_id": str})
    model = None
    if YOLO is not None and Path(args.model).exists():
        model = YOLO(str(args.model))
    else:
        reason = []
        if YOLO is None:
            reason.append("ultralytics_not_installed")
        if not Path(args.model).exists():
            reason.append(f"model_missing:{args.model}")
        msg = f"Vision export fallback requested because {','.join(reason)}"
        if not args.allow_fallback:
            raise RuntimeError(
                f"{msg}. Re-run with a valid YOLO model, or pass --allow-fallback explicitly."
            )
        args.warning_file.parent.mkdir(parents=True, exist_ok=True)
        args.warning_file.write_text(msg + "\n", encoding="utf-8")
        print(f"WARNING: {msg}")

    rows = []
    for row in tqdm(df.to_dict(orient="records"), desc="Vision logits"):
        video_path = Path(row["video_path"])
        p_vis, q_vis = run_video_yolo(video_path, float(row["t0_sec"]), float(row["t1_sec"]), model)
        rows.append(
            {
                "episode_id": row["episode_id"],
                "logit_vis": logit_from_prob(p_vis),
                "q_vis_0": float(q_vis[0]),
                "q_vis_1": float(q_vis[1]),
                "q_vis_2": float(q_vis[2]),
                "q_vis_3": float(q_vis[3]),
                "q_vis_4": float(q_vis[4]),
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
