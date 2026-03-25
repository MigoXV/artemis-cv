from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from artemis_cv.inferencers.smoothers import OneEuroPointSmoother

from .inferencer import YoloPointInferencer


class SharedYoloPointInferencer(YoloPointInferencer):
    """在多个 WebRTC 会话间共享一份模型实例。"""

    def __init__(
        self,
        model_dir: str | Path,
        device: str | torch.device = "cpu",
        imgsz: int = YoloPointInferencer.DEFAULT_IMGSZ,
        frequency: float = 30.0,
        min_cutoff: float = 1.2,
        beta: float = 0.08,
        d_cutoff: float = 1.0,
    ) -> None:
        super().__init__(model_dir=model_dir, device=device, imgsz=imgsz)
        self._lock = threading.Lock()
        self._smoother = OneEuroPointSmoother(
            frequency=frequency,
            min_cutoff=min_cutoff,
            beta=beta,
            d_cutoff=d_cutoff,
        )

    def infer(
        self, bgr: np.ndarray, score_threshold: float
    ) -> Optional[Tuple[Tuple[float, float], Tuple[float, float], float]]:
        with self._lock:
            px, py, score = super().infer(bgr)
        if score < score_threshold:
            self._smoother.reset()
            return None
        height, width = bgr.shape[:2]
        smooth_px, smooth_py = self._smoother.update(px, py)
        clamped_px = float(np.clip(smooth_px, 0.0, width - 1))
        clamped_py = float(np.clip(smooth_py, 0.0, height - 1))
        nx = clamped_px / width if width else 0.0
        ny = clamped_py / height if height else 0.0
        return (nx, ny), (clamped_px, clamped_py), score

    def set_frequency(self, frequency: float) -> None:
        with self._lock:
            self._smoother.set_frequency(frequency)

    def reset(self) -> None:
        with self._lock:
            self._smoother.reset()
