from fractions import Fraction
from math import pi
from pathlib import Path

import av
import cv2
from tqdm import tqdm

from artemis_cv.inferencers.yolo import YoloPointInferencer

ROOT = Path(__file__).resolve().parents[2]
HF_MODEL_DIR = ROOT / 'model-bin' / 'artimes-yolov8n-260323-1629'
VIDEO_PATH = ROOT / 'data-bin' / 'videos' / 'slow.mp4'
OUTPUT_DIR = ROOT / 'data-bin' / 'video-outputs'
FRAME_STRIDE = 4
MIN_CUTOFF = 1.0
BETA = 0.12
D_CUTOFF = 1.0


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
            raise ValueError(f'frequency 必须大于 0，收到: {frequency}')
        if min_cutoff <= 0.0:
            raise ValueError(f'min_cutoff 必须大于 0，收到: {min_cutoff}')
        if d_cutoff <= 0.0:
            raise ValueError(f'd_cutoff 必须大于 0，收到: {d_cutoff}')

        self.frequency = float(frequency)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()

    def reset(self) -> None:
        self.x_filter.reset()
        self.dx_filter.reset()

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

    def update(self, px: float, py: float) -> tuple[float, float]:
        return self.x_filter.update(px), self.y_filter.update(py)


def draw_point(frame, px, py, score, frame_idx, inferred):
    vis = frame.copy()
    cx, cy = int(px), int(py)
    cv2.circle(vis, (cx, cy), 6, (0, 0, 255), -1)
    cv2.circle(vis, (cx, cy), 8, (255, 255, 255), 2)
    label = f'frame={frame_idx} score={score:.3f}'
    status = 'infer' if inferred else 'hold'
    cv2.putText(
        vis,
        f'{label} {status}',
        (cx + 10, cy + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        1,
    )
    return vis


if __name__ == '__main__':
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    inferencer = YoloPointInferencer(HF_MODEL_DIR, device='cpu')

    container = av.open(str(VIDEO_PATH))
    video_stream = container.streams.video[0]
    total_frames = video_stream.frames
    fps = Fraction(video_stream.average_rate)
    smoother = OneEuroPointSmoother(
        frequency=float(fps),
        min_cutoff=MIN_CUTOFF,
        beta=BETA,
        d_cutoff=D_CUTOFF,
    )

    out_path = OUTPUT_DIR / (VIDEO_PATH.stem + '_stride4_one_euro.mp4')
    out_container = None
    out_stream = None
    last_result = None

    try:
        with tqdm(total=total_frames, desc='Inferencing', unit='frame') as pbar:
            for frame_idx, av_frame in enumerate(container.decode(video=0)):
                bgr = av_frame.to_ndarray(format='bgr24')

                if out_container is None:
                    h, w = bgr.shape[:2]
                    out_container = av.open(str(out_path), mode='w')
                    out_stream = out_container.add_stream('h264', rate=fps)
                    out_stream.width = w
                    out_stream.height = h
                    out_stream.pix_fmt = 'yuv420p'

                inferred = frame_idx % FRAME_STRIDE == 0 or last_result is None
                if inferred:
                    last_result = inferencer.infer(bgr)

                px, py, score = last_result
                smooth_px, smooth_py = smoother.update(px, py)
                vis = draw_point(
                    bgr,
                    smooth_px,
                    smooth_py,
                    score,
                    frame_idx=frame_idx,
                    inferred=inferred,
                )

                rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
                out_frame = av.VideoFrame.from_ndarray(rgb, format='rgb24')
                out_frame = out_frame.reformat(format='yuv420p')
                for packet in out_stream.encode(out_frame):
                    out_container.mux(packet)

                pbar.update(1)

        if out_stream is not None:
            for packet in out_stream.encode():
                out_container.mux(packet)
    finally:
        container.close()
        if out_container is not None:
            out_container.close()

    print(f'输出视频: {out_path}')
