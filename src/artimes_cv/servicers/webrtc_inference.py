from __future__ import annotations

import threading
from math import pi
from pathlib import Path

import numpy as np

from artimes_cv.inferencers.yolo import YoloPointInferencer
from artimes_cv.protos.detector import common_pb2


class LowPassFilter:
    def __init__(self) -> None:
        self.initialized = False
        self.prev = 0.0

    def reset(self) -> None:
        self.initialized = False
        self.prev = 0.0

    def update(self, value: float, alpha: float) -> float:
        if not self.initialized:
            self.prev = float(value)
            self.initialized = True
            return self.prev

        self.prev = alpha * float(value) + (1.0 - alpha) * self.prev
        return self.prev


class OneEuroFilter:
    def __init__(self, frequency: float, min_cutoff: float, beta: float, d_cutoff: float) -> None:
        if frequency <= 0.0:
            raise ValueError(f"frequency 必须大于 0，收到: {frequency}")
        if min_cutoff <= 0.0:
            raise ValueError(f"min_cutoff 必须大于 0，收到: {min_cutoff}")
        if d_cutoff <= 0.0:
            raise ValueError(f"d_cutoff 必须大于 0，收到: {d_cutoff}")

        self.frequency = float(frequency)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()

    def reset(self) -> None:
        self.x_filter.reset()
        self.dx_filter.reset()

    def set_frequency(self, frequency: float) -> None:
        if frequency > 0.0:
            self.frequency = float(frequency)

    def _alpha(self, cutoff: float) -> float:
        te = 1.0 / self.frequency
        tau = 1.0 / (2.0 * pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def update(self, value: float) -> float:
        if self.x_filter.initialized:
            derivative = (float(value) - self.x_filter.prev) * self.frequency
        else:
            derivative = 0.0

        dx_hat = self.dx_filter.update(derivative, self._alpha(self.d_cutoff))
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        return self.x_filter.update(float(value), self._alpha(cutoff))


class OneEuroPointSmoother:
    def __init__(self, frequency: float, min_cutoff: float, beta: float, d_cutoff: float) -> None:
        self.x_filter = OneEuroFilter(
            frequency=frequency,
            min_cutoff=min_cutoff,
            beta=beta,
            d_cutoff=d_cutoff,
        )
        self.y_filter = OneEuroFilter(
            frequency=frequency,
            min_cutoff=min_cutoff,
            beta=beta,
            d_cutoff=d_cutoff,
        )

    def reset(self) -> None:
        self.x_filter.reset()
        self.y_filter.reset()

    def set_frequency(self, frequency: float) -> None:
        self.x_filter.set_frequency(frequency)
        self.y_filter.set_frequency(frequency)

    def update(self, px: float, py: float) -> tuple[float, float]:
        return self.x_filter.update(px), self.y_filter.update(py)


class SharedYoloPointInferencer:
    """在多个 WebRTC 会话间共享一份模型实例。"""

    def __init__(self, model_dir: str | Path, device: str) -> None:
        self.model_dir = Path(model_dir)
        self.device = device
        self._lock = threading.Lock()
        self._inferencer = YoloPointInferencer(model_dir=self.model_dir, device=device)

    def infer(self, bgr: np.ndarray) -> tuple[float, float, float]:
        with self._lock:
            return self._inferencer.infer(bgr)


class WebRtcVisionInference:
    """将视频帧推理为 Detection proto，并维护单流平滑状态。"""

    def __init__(
        self,
        shared_inferencer: SharedYoloPointInferencer,
        score_threshold: float = 0.0,
        frequency: float = 30.0,
        min_cutoff: float = 1.2,
        beta: float = 0.08,
        d_cutoff: float = 1.0,
    ) -> None:
        self._shared_inferencer = shared_inferencer
        self._score_threshold = float(score_threshold)
        self._smoother = OneEuroPointSmoother(
            frequency=frequency,
            min_cutoff=min_cutoff,
            beta=beta,
            d_cutoff=d_cutoff,
        )

    def set_frequency(self, frequency: float) -> None:
        self._smoother.set_frequency(frequency)

    def reset(self) -> None:
        self._smoother.reset()

    def get_detection(self, bgr: np.ndarray) -> common_pb2.Detection | None:
        height, width = bgr.shape[:2]
        px, py, score = self._shared_inferencer.infer(bgr)

        if score < self._score_threshold:
            self._smoother.reset()
            return None

        smooth_px, smooth_py = self._smoother.update(px, py)
        clamped_px = float(np.clip(smooth_px, 0.0, width - 1))
        clamped_py = float(np.clip(smooth_py, 0.0, height - 1))
        nx = clamped_px / width if width else 0.0
        ny = clamped_py / height if height else 0.0

        return common_pb2.Detection(
            class_name="yolo_point",
            class_id=0,
            score=float(score),
            geometry=common_pb2.DetectionGeometry(
                point=common_pb2.Point2D(x=clamped_px, y=clamped_py)
            ),
            normalized_geometry=common_pb2.DetectionGeometry(
                point=common_pb2.Point2D(x=nx, y=ny)
            ),
        )
