from __future__ import annotations

from math import pi


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
    def __init__(
        self, frequency: float, min_cutoff: float, beta: float, d_cutoff: float
    ) -> None:
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
    def __init__(
        self, frequency: float, min_cutoff: float, beta: float, d_cutoff: float
    ) -> None:
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
