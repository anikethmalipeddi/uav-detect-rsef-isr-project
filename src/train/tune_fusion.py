from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

from src.models.fusion_gate import average_fusion, build_fusion_features
from src.train.train_fusion import eval_gate, eval_static, infer_unimodal_logits, train_gate_model
from src.utils.io import save_npz
from src.utils.seed import set_seed


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _build_configs(args: argparse.Namespace) -> list[dict]:
    cfgs: list[dict] = []
    for h1, h2, drop, wd, lr, d1, d2, aux in itertools.product(
        _parse_int_list(args.hidden1_grid),
        _parse_int_list(args.hidden2_grid),
        _parse_float_list(args.gate_dropout_grid),
        _parse_float_list(args.weight_decay_grid),
        _parse_float_list(args.lr_grid),
        _parse_float_list(args.drop_one_grid),
        _parse_float_list(args.drop_two_grid),
        _parse_float_list(args.aux_weight_grid),
    ):
        cfgs.append(
            {
                "hidden1": h1,
                "hidden2": h2,
                "gate_dropout": drop,
                "weight_decay": wd,
                "lr": lr,
                "drop_one": d1,
                "drop_two": d2,
                "aux_weight": aux,
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
            }
        )
    return cfgs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, default=Path("data/processed/episodes/episodes.csv"))
    parser.add_argument("--features", type=Path, default=Path("data/processed/episodes/features_cache.npz"))
    parser.add_argument("--vision-csv", type=Path, default=Path("data/processed/episodes/vision_presence.csv"))
    parser.add_argument("--audio-ckpt", type=Path, default=Path("results/checkpoints/best_audio.pt"))
    parser.add_argument("--radar-ckpt", type=Path, default=Path("results/checkpoints/best_radar.pt"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden1-grid", type=str, default="64,96")
    parser.add_argument("--hidden2-grid", type=str, default="32,48")
    parser.add_argument("--gate-dropout-grid", type=str, default="0.15,0.25")
    parser.add_argument("--weight-decay-grid", type=str, default="0.0001,0.0005")
    parser.add_argument("--lr-grid", type=str, default="0.0003,0.0005")
    parser.add_argument("--drop-one-grid", type=str, default="0.10,0.15")
    parser.add_argument("--drop-two-grid", type=str, default="0.03,0.05")
    parser.add_argument("--aux-weight-grid", type=str, default="0.0,0.1")
    parser.add_argument("--max-configs", type=int, default=16)
    parser.add_argument("--out-csv", type=Path, default=Path("results/fusion_tuning.csv"))
    parser.add_argument("--best-json", type=Path, default=Path("results/fusion_best_config.json"))
    parser.add_argument("--train-final", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not args.episodes.exists():
        raise FileNotFoundError(f"Missing episodes CSV: {args.episodes}")
    episodes = pd.read_csv(args.episodes, dtype={"episode_id": str})
    y = episodes["label_drone"].to_numpy(dtype=np.int64)
    split = episodes["split"].astype(str).to_numpy()
    idx_train = np.where(split == "train")[0]
    idx_val = np.where(split == "val")[0]
    if idx_train.size == 0 or idx_val.size == 0:
        raise RuntimeError("Fusion tuning requires non-empty train and val splits.")

    mask = episodes[["miss_vis", "miss_aud", "miss_rad"]].to_numpy(dtype=np.float32)
    if np.all(mask == 0):
        mask[:] = 1.0

    logit_vis, logit_aud, logit_rad, q_vis, q_aud, q_rad = infer_unimodal_logits(
        episodes,
        device=device,
        audio_ckpt=args.audio_ckpt,
        radar_ckpt=args.radar_ckpt,
        vision_csv=args.vision_csv,
    )
    logits = np.stack([logit_vis, logit_aud, logit_rad], axis=1).astype(np.float32)
    x = build_fusion_features(logit_vis, logit_aud, logit_rad, q_vis, q_aud, q_rad, mask).astype(np.float32)

    # Persist aligned feature cache with strict schema/order.
    save_npz(
        args.features,
        episode_id=episodes["episode_id"].to_numpy(dtype=np.str_),
        y=y.astype(np.int64),
        logit_vis=logit_vis.astype(np.float32),
        logit_aud=logit_aud.astype(np.float32),
        logit_rad=logit_rad.astype(np.float32),
        q_vis=q_vis.astype(np.float32),
        q_aud=q_aud.astype(np.float32),
        q_rad=q_rad.astype(np.float32),
        mask=mask.astype(np.float32),
    )

    # Baselines on validation split only.
    with torch.no_grad():
        lval = torch.from_numpy(logits[idx_val].astype(np.float32))
        mval = torch.from_numpy(mask[idx_val].astype(np.float32))
        p_avg_val = torch.sigmoid(average_fusion(lval, mval)[0]).numpy().reshape(-1)
    y_val = y[idx_val].astype(np.float32)
    val_pr_auc_avg = float(average_precision_score(y_val, p_avg_val))
    p_static_all, alpha = eval_static(logits, mask, y[idx_train].astype(np.float32), idx_train)
    val_pr_auc_static = float(average_precision_score(y_val, p_static_all[idx_val]))

    configs = _build_configs(args)
    if len(configs) > args.max_configs:
        configs = configs[: args.max_configs]
    if not configs:
        raise RuntimeError("No fusion tuning configurations were generated.")

    rows: list[dict] = []
    best_idx = -1
    best_val = -1.0
    for i, cfg in enumerate(configs):
        cfg_seed = int(args.seed + i + 1)
        gate = train_gate_model(
            x[idx_train],
            logits[idx_train],
            y[idx_train].astype(np.float32),
            mask[idx_train],
            x[idx_val],
            logits[idx_val],
            y[idx_val].astype(np.float32),
            mask[idx_val],
            include_quality=True,
            seed=cfg_seed,
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            lr=cfg["lr"],
            hidden1=cfg["hidden1"],
            hidden2=cfg["hidden2"],
            gate_dropout=cfg["gate_dropout"],
            weight_decay=cfg["weight_decay"],
            drop_one=cfg["drop_one"],
            drop_two=cfg["drop_two"],
            aux_weight=cfg["aux_weight"],
        )
        p_gate_val = eval_gate(gate, x[idx_val], logits[idx_val], mask[idx_val])
        pr_val_gate = float(average_precision_score(y_val, p_gate_val))

        gate_noq = train_gate_model(
            x[idx_train],
            logits[idx_train],
            y[idx_train].astype(np.float32),
            mask[idx_train],
            x[idx_val],
            logits[idx_val],
            y[idx_val].astype(np.float32),
            mask[idx_val],
            include_quality=False,
            seed=cfg_seed,
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            lr=cfg["lr"],
            hidden1=cfg["hidden1"],
            hidden2=cfg["hidden2"],
            gate_dropout=cfg["gate_dropout"],
            weight_decay=cfg["weight_decay"],
            drop_one=cfg["drop_one"],
            drop_two=cfg["drop_two"],
            aux_weight=cfg["aux_weight"],
        )
        x_noq_val = x[idx_val].copy()
        x_noq_val[:, 9:21] = 0.0
        p_gate_noq_val = eval_gate(gate_noq, x_noq_val, logits[idx_val], mask[idx_val])
        pr_val_gate_noq = float(average_precision_score(y_val, p_gate_noq_val))

        row = {
            "config_id": i,
            "seed": cfg_seed,
            **cfg,
            "val_pr_auc_avg": val_pr_auc_avg,
            "val_pr_auc_static": val_pr_auc_static,
            "val_pr_auc_gate": pr_val_gate,
            "val_pr_auc_gate_noq": pr_val_gate_noq,
            "val_delta_gate_vs_avg": pr_val_gate - val_pr_auc_avg,
            "val_delta_gate_vs_static": pr_val_gate - val_pr_auc_static,
        }
        rows.append(row)
        if pr_val_gate > best_val:
            best_val = pr_val_gate
            best_idx = i

    tuning_df = pd.DataFrame(rows).sort_values("val_pr_auc_gate", ascending=False).reset_index(drop=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    tuning_df.to_csv(args.out_csv, index=False)

    best_row = tuning_df.iloc[0].to_dict()
    best_payload = {
        "selected_config_id": int(best_row["config_id"]),
        "selection_metric": "val_pr_auc_gate",
        "selected_config": {
            "epochs": int(best_row["epochs"]),
            "batch_size": int(best_row["batch_size"]),
            "hidden1": int(best_row["hidden1"]),
            "hidden2": int(best_row["hidden2"]),
            "gate_dropout": float(best_row["gate_dropout"]),
            "weight_decay": float(best_row["weight_decay"]),
            "lr": float(best_row["lr"]),
            "drop_one": float(best_row["drop_one"]),
            "drop_two": float(best_row["drop_two"]),
            "aux_weight": float(best_row["aux_weight"]),
        },
        "val_metrics": {
            "avg": float(best_row["val_pr_auc_avg"]),
            "static": float(best_row["val_pr_auc_static"]),
            "gate": float(best_row["val_pr_auc_gate"]),
            "gate_noq": float(best_row["val_pr_auc_gate_noq"]),
        },
        "tuning_csv": str(args.out_csv),
    }
    args.best_json.parent.mkdir(parents=True, exist_ok=True)
    args.best_json.write_text(json.dumps(best_payload, indent=2), encoding="utf-8")

    if args.train_final:
        cfg = best_payload["selected_config"]
        gate = train_gate_model(
            x[idx_train],
            logits[idx_train],
            y[idx_train].astype(np.float32),
            mask[idx_train],
            x[idx_val],
            logits[idx_val],
            y[idx_val].astype(np.float32),
            mask[idx_val],
            include_quality=True,
            seed=args.seed,
            epochs=int(cfg["epochs"]),
            batch_size=int(cfg["batch_size"]),
            lr=float(cfg["lr"]),
            hidden1=int(cfg["hidden1"]),
            hidden2=int(cfg["hidden2"]),
            gate_dropout=float(cfg["gate_dropout"]),
            weight_decay=float(cfg["weight_decay"]),
            drop_one=float(cfg["drop_one"]),
            drop_two=float(cfg["drop_two"]),
            aux_weight=float(cfg["aux_weight"]),
        )
        gate_noq = train_gate_model(
            x[idx_train],
            logits[idx_train],
            y[idx_train].astype(np.float32),
            mask[idx_train],
            x[idx_val],
            logits[idx_val],
            y[idx_val].astype(np.float32),
            mask[idx_val],
            include_quality=False,
            seed=args.seed,
            epochs=int(cfg["epochs"]),
            batch_size=int(cfg["batch_size"]),
            lr=float(cfg["lr"]),
            hidden1=int(cfg["hidden1"]),
            hidden2=int(cfg["hidden2"]),
            gate_dropout=float(cfg["gate_dropout"]),
            weight_decay=float(cfg["weight_decay"]),
            drop_one=float(cfg["drop_one"]),
            drop_two=float(cfg["drop_two"]),
            aux_weight=float(cfg["aux_weight"]),
        )
        _, alpha = eval_static(logits, mask, y[idx_train].astype(np.float32), idx_train)

        ckpt_dir = Path("results/checkpoints")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(gate.state_dict(), ckpt_dir / "fusion_gate.pt")
        torch.save(gate_noq.state_dict(), ckpt_dir / "fusion_gate_noq.pt")
        np.save(ckpt_dir / "fusion_static_alpha.npy", alpha.astype(np.float32))

    print(f"Wrote {args.out_csv} and {args.best_json}")


if __name__ == "__main__":
    main()
