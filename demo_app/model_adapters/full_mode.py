from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple


class FullModeAdapter:
    """Optional live inference adapter for local, gitignored weights."""

    required_weight_names = (
        "vision_model.pt",
        "audio_model.pt",
        "radar_model.pt",
        "fusion_model.pt",
    )

    def __init__(self, models_dir: Path) -> None:
        self.models_dir = models_dir

    def missing_weights(self) -> List[str]:
        missing: List[str] = []
        for name in self.required_weight_names:
            if not (self.models_dir / name).exists():
                missing.append(name)
        return missing

    def dependency_ok(self) -> bool:
        return importlib.util.find_spec("ultralytics") is not None

    def is_available(self) -> Tuple[bool, str]:
        missing = self.missing_weights()
        if missing:
            return False, f"Missing local weights in models/: {', '.join(missing)}"
        if not self.dependency_ok():
            return False, "ultralytics is not installed in this environment"
        return True, "Full mode dependencies and weights detected"

    def infer(self, _: Dict[str, object]) -> Dict[str, object]:
        # Full-mode implementation is intentionally conservative for demo safety.
        # The app falls back to Poster Mode unless dependencies + weights are present.
        available, reason = self.is_available()
        if not available:
            raise RuntimeError(reason)
        raise NotImplementedError(
            "Full Mode adapter wiring is intentionally minimal for this repo; "
            "place local weights under models/ and extend this adapter."
        )

