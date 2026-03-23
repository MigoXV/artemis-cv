from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import av
import cv2
import numpy as np
from tqdm import tqdm

from artimes_cv.inferencers.yolo import YoloPointInferencer

ROOT = Path(__file__).resolve().parents[2]
HF_MODEL_DIR = ROOT / 'model-bin' / 'artimes-yolov8n-260323-1629'
VIDEO_PATH = ROOT / 'data-bin' / 'videos' / 'slow.mp4'
OUTPUT_DIR = ROOT / 'data-bin' / 'video-outputs'
TRACK_BOX_SIZE = 64.0
TRACK_DET_THRESH = 0.25
EMA_ALPHA = 0.2
TRACKER_ARGS = SimpleNamespace(
    track_high_thresh=0.25,
    track_low_thresh=0.1,
    new_track_thresh=0.25,
    track_buffer=30,
    match_thresh=0.8,
    fuse_score=True,
)


class PointTrackResults:
    """为 ByteTrack 适配单点检测结果。"""

    def __init__(self, xywh: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> None:
        self.xywh = xywh.astype(np.float32, copy=False)
        self.conf = conf.astype(np.float32, copy=False)
        self.cls = cls.astype(np.float32, copy=False)

        if len(self.xywh) == 0:
            self.xyxy = np.empty((0, 4), dtype=np.float32)
        else:
            half_wh = self.xywh[:, 2:4] / 2.0
            self.xyxy = np.concatenate(
                [self.xywh[:, 0:2] - half_wh, self.xywh[:, 0:2] + half_wh], axis=1
            ).astype(np.float32, copy=False)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, idx) -> 'PointTrackResults':
        return PointTrackResults(
            np.atleast_2d(self.xywh[idx]),
            np.atleast_1d(self.conf[idx]),
            np.atleast_1d(self.cls[idx]),
        )

    @classmethod
    def empty(cls) -> 'PointTrackResults':
        return cls(
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    @classmethod
    def from_point(
        cls,
        px: float,
        py: float,
        score: float,
        frame_w: int,
        frame_h: int,
        box_size: float,
    ) -> 'PointTrackResults':
        width = min(box_size, float(frame_w))
        height = min(box_size, float(frame_h))
        cx = float(np.clip(px, 0.0, frame_w - 1))
        cy = float(np.clip(py, 0.0, frame_h - 1))
        return cls(
            np.array([[cx, cy, width, height]], dtype=np.float32),
            np.array([score], dtype=np.float32),
            np.array([0], dtype=np.float32),
        )


class ExponentialPointSmoother:
    """对单点轨迹做指数平滑，抑制高频抖动。"""

    def __init__(self, alpha: float) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f'alpha 必须在 (0, 1] 范围内，收到: {alpha}')
        self.alpha = alpha
        self._state: tuple[float, float] | None = None

    def reset(self) -> None:
        self._state = None

    def update(self, px: float, py: float) -> tuple[float, float]:
        if self._state is None:
            self._state = (float(px), float(py))
            return self._state

        prev_x, prev_y = self._state
        next_x = self.alpha * float(px) + (1.0 - self.alpha) * prev_x
        next_y = self.alpha * float(py) + (1.0 - self.alpha) * prev_y
        self._state = (next_x, next_y)
        return self._state


def draw_point(frame, px, py, score, track_id=None):
    vis = frame.copy()
    cx, cy = int(px), int(py)
    cv2.circle(vis, (cx, cy), 6, (0, 0, 255), -1)
    cv2.circle(vis, (cx, cy), 8, (255, 255, 255), 2)
    label = f'{score:.3f}'
    if track_id is not None:
        label = f'id={track_id} {label}'
    cv2.putText(vis, label, (cx + 10, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return vis


if __name__ == '__main__':
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from ultralytics.trackers.byte_tracker import BYTETracker
    except ImportError as exc:
        raise ImportError(
            "该脚本现在依赖 ultralytics 的 ByteTrack。请先安装: pip install ultralytics"
        ) from exc

    container = av.open(str(VIDEO_PATH))
    video_stream = container.streams.video[0]
    total_frames = video_stream.frames
    fps = Fraction(video_stream.average_rate)

    inferencer = YoloPointInferencer(HF_MODEL_DIR, device='cpu')
    tracker = BYTETracker(args=TRACKER_ARGS, frame_rate=float(fps))
    smoother = ExponentialPointSmoother(alpha=EMA_ALPHA)
    active_track_id = None

    out_path = OUTPUT_DIR / (VIDEO_PATH.stem + '_tracked.mp4')
    out_container = None
    out_stream = None

    try:
        with tqdm(total=total_frames, desc='Inferencing', unit='frame') as pbar:
            for av_frame in container.decode(video=0):
                bgr = av_frame.to_ndarray(format='bgr24')

                if out_container is None:
                    h, w = bgr.shape[:2]
                    out_container = av.open(str(out_path), mode='w')
                    out_stream = out_container.add_stream('h264', rate=fps)
                    out_stream.width = w
                    out_stream.height = h
                    out_stream.pix_fmt = 'yuv420p'

                px, py, score = inferencer.infer(bgr)

                h, w = bgr.shape[:2]
                if score >= TRACK_DET_THRESH:
                    detections = PointTrackResults.from_point(
                        px, py, score, frame_w=w, frame_h=h, box_size=TRACK_BOX_SIZE
                    )
                else:
                    detections = PointTrackResults.empty()

                tracks = tracker.update(detections, bgr)

                track_px, track_py, track_score = px, py, score
                track_id = None
                if len(tracks) > 0:
                    if active_track_id is not None:
                        matched = tracks[tracks[:, 4] == active_track_id]
                    else:
                        matched = np.empty((0, tracks.shape[1]), dtype=tracks.dtype)

                    chosen = matched[0] if len(matched) > 0 else tracks[np.argmax(tracks[:, 5])]
                    x1, y1, x2, y2, track_id, track_score = chosen[:6]
                    track_px = float((x1 + x2) / 2.0)
                    track_py = float((y1 + y2) / 2.0)
                    active_track_id = int(track_id)
                    track_id = active_track_id
                elif score < TRACK_DET_THRESH:
                    active_track_id = None
                    smoother.reset()

                smooth_px, smooth_py = smoother.update(track_px, track_py)
                vis = draw_point(bgr, smooth_px, smooth_py, track_score, track_id=track_id)

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
