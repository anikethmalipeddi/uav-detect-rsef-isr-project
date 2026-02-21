from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.thesis_dataset import AudioNPYDataset
from src.models.audio_cnn import AudioCNN
from src.utils.metrics import metrics_row, plot_pr_curve
from src.utils.seed import set_seed


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true = []
    p_score = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logit = model(x)
            p = torch.sigmoid(logit)
            y_true.append(y.cpu().numpy().reshape(-1))
            p_score.append(p.cpu().numpy().reshape(-1))
    return np.concatenate(y_true), np.concatenate(p_score)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, default=Path("data/processed/episodes/episodes.csv"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-ckpt", type=Path, default=Path("results/checkpoints/best_audio.pt"))
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_train = AudioNPYDataset(args.episodes, split="train")
    ds_val = AudioNPYDataset(args.episodes, split="val")
    ds_test = AudioNPYDataset(args.episodes, split="test")

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=0)
    dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = AudioCNN().to(device)
    y_train = np.array([ds_train.df.iloc[i]["label_drone"] for i in range(len(ds_train))], dtype=np.float32)
    pos = max(1.0, float(y_train.sum()))
    neg = max(1.0, float(len(y_train) - y_train.sum()))
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val = -1.0
    args.out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(dl_train, desc=f"audio epoch {epoch + 1}/{args.epochs}")
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logit = model(x)
            loss = criterion(logit, y)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        yv, pv = evaluate(model, dl_val, device=device)
        from sklearn.metrics import average_precision_score

        val_ap = float(average_precision_score(yv, pv))
        if val_ap > best_val:
            best_val = val_ap
            torch.save({"model": model.state_dict(), "val_ap": val_ap}, args.out_ckpt)

    ckpt = torch.load(args.out_ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    yt, pt = evaluate(model, dl_test, device=device)
    row = metrics_row("UNI-A", "test", yt, pt, thr=0.5)

    import pandas as pd

    out_csv = Path("results/metrics_unimodal.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_csv.exists():
        old = pd.read_csv(out_csv)
        out = pd.concat([old[old["run_id"] != "UNI-A"], pd.DataFrame([row])], ignore_index=True)
    else:
        out = pd.DataFrame([row])
    out.to_csv(out_csv, index=False)

    plot_pr_curve(yt, pt, "results/figures/pr_curves/audio_pr.png", "Audio CNN PR Curve")
    print(f"Saved {args.out_ckpt} and updated {out_csv}")


if __name__ == "__main__":
    main()
