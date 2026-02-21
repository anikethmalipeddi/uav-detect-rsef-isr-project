from __future__ import annotations

from pathlib import Path

import pandas as pd


class PosterModeAdapter:
    """Loads committed precomputed outputs for deterministic offline replay."""

    def __init__(self, outputs_dir: Path) -> None:
        self.outputs_dir = outputs_dir
        self._preds = pd.read_csv(outputs_dir / "predictions_per_frame.csv")
        self._gates = pd.read_csv(outputs_dir / "gating_weights_per_frame.csv")
        self._bboxes = pd.read_csv(outputs_dir / "bboxes_per_frame.csv")

    @property
    def predictions(self) -> pd.DataFrame:
        return self._preds.copy()

    @property
    def gate_weights(self) -> pd.DataFrame:
        return self._gates.copy()

    @property
    def bboxes(self) -> pd.DataFrame:
        return self._bboxes.copy()

