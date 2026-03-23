from fractions import Fraction
from pathlib import Path

import av
import cv2
from tqdm import tqdm

from artimes_cv.inferencers.yolo import YoloPointInferencer

ROOT = Path(__file__).resolve().parents[2]
HF_MODEL_DIR = ROOT / 'model-bin' / 'artimes-yolov8n-260323-1629'
VIDEO_PATH = ROOT / 'data-bin' / 'videos' / 'slow.mp4'
OUTPUT_DIR = ROOT / 'data-bin' / 'video-outputs'
FRAME_STRIDE = 4


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

    out_path = OUTPUT_DIR / (VIDEO_PATH.stem + '_stride4.mp4')
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
                vis = draw_point(bgr, px, py, score, frame_idx=frame_idx, inferred=inferred)

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
