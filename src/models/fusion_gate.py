from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def binary_entropy_np(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def build_fusion_features(
    logit_vis: np.ndarray,
    logit_aud: np.ndarray,
    logit_rad: np.ndarray,
    q_vis: np.ndarray,
    q_aud: np.ndarray,
    q_rad: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    p_vis = sigmoid_np(logit_vis)
    p_aud = sigmoid_np(logit_aud)
    p_rad = sigmoid_np(logit_rad)

    e_vis = binary_entropy_np(p_vis)
    e_aud = binary_entropy_np(p_aud)
    e_rad = binary_entropy_np(p_rad)

    parts = [
        logit_vis[:, None],
        p_vis[:, None],
        e_vis[:, None],
        logit_aud[:, None],
        p_aud[:, None],
        e_aud[:, None],
        logit_rad[:, None],
        p_rad[:, None],
        e_rad[:, None],
        q_vis,
        q_aud,
        q_rad,
        mask,
    ]
    x = np.concatenate(parts, axis=1).astype(np.float32)
    if x.shape[1] != 24:
        raise ValueError(f"Expected fusion feature dim 24, got {x.shape[1]}")
    return x


class FusionGateMLP(nn.Module):
    def __init__(self, input_dim: int = 24, hidden1: int = 64, hidden2: int = 32, dropout: float = 0.2) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.register_buffer("x_mean", torch.zeros(input_dim, dtype=torch.float32))
        self.register_buffer("x_std", torch.ones(input_dim, dtype=torch.float32))
        self.register_buffer("x_norm_enabled", torch.tensor(0, dtype=torch.uint8))
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 3),
        )

    def set_input_norm_stats(self, mean: np.ndarray, std: np.ndarray) -> None:
        m = torch.as_tensor(mean, dtype=torch.float32).reshape(-1)
        s = torch.as_tensor(std, dtype=torch.float32).reshape(-1).clamp_min(1e-6)
        if m.numel() != self.input_dim or s.numel() != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} stats entries, got {m.numel()} and {s.numel()}")
        self.x_mean.copy_(m)
        self.x_std.copy_(s)
        self.x_norm_enabled.fill_(1)

    def forward(self, x: torch.Tensor, logits: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, 24], logits: [B, 3], mask: [B, 3] with 1 present and 0 missing.
        if bool(self.x_norm_enabled.item()):
            x = (x - self.x_mean[None, :]) / self.x_std[None, :]
        gate_logits = self.net(x)
        masked_gate_logits = gate_logits.masked_fill(mask <= 0.0, -1e9)
        w = torch.softmax(masked_gate_logits, dim=1)
        fused = (w * logits).sum(dim=1, keepdim=True)
        return fused, w


class StaticFusion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(3))

    def forward(self, logits: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a = self.alpha[None, :].expand_as(logits)
        a = a.masked_fill(mask <= 0.0, -1e9)
        w = torch.softmax(a, dim=1)
        fused = (w * logits).sum(dim=1, keepdim=True)
        return fused, w


def average_fusion(logits: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    w = mask / (mask.sum(dim=1, keepdim=True).clamp_min(1e-6))
    fused = (w * logits).sum(dim=1, keepdim=True)
    return fused, w
