from __future__ import annotations

import torch
import torch.nn as nn

from transformers import ASTModel


class AudioASTBinary(nn.Module):
    def __init__(self, model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593") -> None:
        super().__init__()
        self.encoder = ASTModel.from_pretrained(model_name)
        self.head = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 128, T]
        out = self.encoder(input_values=x)
        pooled = out.last_hidden_state.mean(dim=1)
        return self.head(pooled)

    def freeze_encoder(self, freeze: bool = True) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = not freeze
