import time

from artimes_cv.protos.detector import common_pb2


class StubInference:
    """生成沿旋转 45° 正方形（菱形）路径运动的检测点。"""

    def __init__(
        self,
        center_x: float = 0.5,
        center_y: float = 0.5,
        radius: float = 0.3,
        period: float = 4.0,
    ):
        self._cx = center_x
        self._cy = center_y
        self._r = radius
        self._period = period
        self._start = time.monotonic()

    def get_detection(self, width: int, height: int) -> common_pb2.Detection:
        elapsed = time.monotonic() - self._start
        t = (elapsed % self._period) / self._period  # 0..1

        # 菱形四个顶点: 上 → 右 → 下 → 左
        verts = [
            (self._cx, self._cy - self._r),
            (self._cx + self._r, self._cy),
            (self._cx, self._cy + self._r),
            (self._cx - self._r, self._cy),
        ]

        seg = min(int(t * 4), 3)
        lt = t * 4 - seg
        p1, p2 = verts[seg], verts[(seg + 1) % 4]
        nx = p1[0] + (p2[0] - p1[0]) * lt
        ny = p1[1] + (p2[1] - p1[1]) * lt

        return common_pb2.Detection(
            class_name="stub_point",
            class_id=0,
            score=1.0,
            geometry=common_pb2.DetectionGeometry(
                point=common_pb2.Point2D(x=nx * width, y=ny * height),
            ),
            normalized_geometry=common_pb2.DetectionGeometry(
                point=common_pb2.Point2D(x=nx, y=ny),
            ),
        )
