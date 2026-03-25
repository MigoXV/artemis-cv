"""Demo WebRTC 推理客户端：发送本地视频，接收服务端 YOLO 点检测结果并叠加显示。"""

import asyncio
import fractions
import queue
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import av
import cv2
import grpc
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from av import VideoFrame

from artemis_cv.protos.detector import common_pb2
from artemis_cv.protos.detector import webrtc_detector_pb2 as pb2
from artemis_cv.protos.detector import webrtc_detector_pb2_grpc as pb2_grpc

ROOT = Path(__file__).resolve().parents[2]
SERVER_ADDR = "localhost:50052"
VIDEO_PATH = ROOT / "data-bin" / "videos" / "slow.mp4"
SCORE_THRESHOLD = 0.0
FRAME_DOWNSAMPLE = 1
SYNC_WAIT_SECONDS = 0.75
MAX_PENDING_FRAMES = 120
MAX_PENDING_DETECTIONS = 240
LATEST_DETECTION_HOLD_SECONDS = 1.0


class VideoFileSource:
    """本地视频文件读取器。"""

    def __init__(self, video_path: Path, loop: bool = True):
        self.video_path = video_path
        self.loop = loop
        self._container: av.container.InputContainer | None = None
        self._frames = None
        self._fps = 30.0
        self._time_base = fractions.Fraction(1, 30)
        self._source_frame_index = -1
        self._open_video()

    def _open_video(self) -> None:
        if self._container is not None:
            self._container.close()

        self._container = av.open(str(self.video_path))
        video_stream = self._container.streams.video[0]
        self._frames = self._container.decode(video=0)

        if video_stream.average_rate:
            self._fps = float(video_stream.average_rate)
        rounded_fps = max(1, round(self._fps))
        self._time_base = fractions.Fraction(1, rounded_fps)
        self._source_frame_index = -1

    def read_next(self) -> tuple[Any, int, int]:
        while True:
            try:
                av_frame = next(self._frames)
            except StopIteration:
                if not self.loop:
                    raise EOFError("video ended")
                self._open_video()
                continue

            self._source_frame_index += 1
            bgr = av_frame.to_ndarray(format="bgr24")
            pts_ms = self._pts_to_ms(self._source_frame_index, self._time_base)
            return bgr, self._source_frame_index, pts_ms

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def time_base(self) -> fractions.Fraction:
        return self._time_base

    @staticmethod
    def _pts_to_ms(pts: int, time_base: fractions.Fraction) -> int:
        return int(round(float(pts * time_base) * 1000.0))

    def close(self) -> None:
        if self._container is not None:
            self._container.close()
            self._container = None


class VideoFileTrack(MediaStreamTrack):
    """将本地视频文件包装成 WebRTC 视频轨。"""

    kind = "video"

    def __init__(self, video_path: Path, loop: bool = True, frame_downsample: int = 1):
        super().__init__()
        self.frame_downsample = max(1, frame_downsample)
        self.source = VideoFileSource(video_path=video_path, loop=loop)
        self._start = time.time()
        self._send_index = 0

    async def recv(self) -> VideoFrame:
        source_frame_index = self._send_index * self.frame_downsample
        target = self._start + source_frame_index / self.source.fps
        wait = target - time.time()
        if wait > 0:
            await asyncio.sleep(wait)

        bgr = None
        frame_index = 0
        for _ in range(self.frame_downsample):
            bgr, frame_index, _ = self.source.read_next()

        self._send_index += 1
        frame = VideoFrame.from_ndarray(bgr, format="bgr24")
        frame.pts = frame_index
        frame.time_base = self.source.time_base
        return frame

    def close(self) -> None:
        self.source.close()


class VideoDisplayPlayer:
    """本地全帧播放，用于显示与叠加检测结果。"""

    def __init__(self, video_path: Path, loop: bool = True):
        self.source = VideoFileSource(video_path=video_path, loop=loop)
        self.display_queue: queue.Queue = queue.Queue(maxsize=MAX_PENDING_FRAMES)
        self._start = time.time()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                img, frame_index, pts_ms = self.source.read_next()
            except EOFError:
                break

            target = self._start + frame_index / self.source.fps
            wait = target - time.time()
            if wait > 0:
                time.sleep(wait)

            try:
                self.display_queue.put_nowait(
                    (img, frame_index, pts_ms, time.monotonic())
                )
            except queue.Full:
                try:
                    self.display_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.display_queue.put_nowait(
                        (img, frame_index, pts_ms, time.monotonic())
                    )
                except queue.Full:
                    pass

    def close(self) -> None:
        self._stop_event.set()
        self.source.close()
        if self._thread is not None:
            self._thread.join(timeout=2)


async def run_client(
    server_addr: str = "localhost:50051",
    video_path: Path = VIDEO_PATH,
    score_threshold: float = 0.0,
    frame_downsample: int = 1,
):
    channel = grpc.aio.insecure_channel(server_addr)
    stub = pb2_grpc.WebRtcDetectorEngineStub(channel)

    print(f"Connecting to {server_addr} ...")
    reply = await stub.CreateStream(
        pb2.CreateStreamRequest(
            config=common_pb2.StreamConfig(
                video_codec="vp8",
                score_threshold=score_threshold,
            )
        )
    )
    stream_id = reply.stream_id
    print(f"Stream created: {stream_id}")

    pc = RTCPeerConnection()
    display_player = VideoDisplayPlayer(video_path=video_path, loop=True)
    video_track = VideoFileTrack(
        video_path=video_path,
        loop=True,
        frame_downsample=frame_downsample,
    )
    pc.addTrack(video_track)

    await pc.setRemoteDescription(
        RTCSessionDescription(sdp=reply.offer.sdp, type=reply.offer.type)
    )
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    while pc.iceGatheringState != "complete":
        await asyncio.sleep(0.1)

    await stub.UpdateStream(
        pb2.StreamSignal(
            stream_id=stream_id,
            answer=pb2.SessionDescription(
                type=pc.localDescription.type,
                sdp=pc.localDescription.sdp,
            ),
        )
    )
    print("WebRTC connected!")
    display_player.start()

    detection_by_sync_id: OrderedDict[int, pb2.StreamDetectionsReply] = OrderedDict()
    det_lock = threading.Lock()
    latest_detection: pb2.StreamDetectionsReply | None = None
    latest_detection_at = 0.0

    def pop_detection(sync_id: int) -> pb2.StreamDetectionsReply | None:
        return detection_by_sync_id.pop(sync_id, None)

    async def recv_detections():
        nonlocal latest_detection, latest_detection_at
        try:
            async for det_reply in stub.StreamDetections(
                pb2.StreamDetectionsRequest(stream_id=stream_id)
            ):
                with det_lock:
                    detection_by_sync_id[det_reply.frame_id] = det_reply
                    if det_reply.detections:
                        latest_detection = det_reply
                        latest_detection_at = time.monotonic()
                    while len(detection_by_sync_id) > MAX_PENDING_DETECTIONS:
                        detection_by_sync_id.popitem(last=False)
        except Exception as exc:
            print(f"Detection stream ended: {exc}")

    det_task = asyncio.ensure_future(recv_detections())
    stop_event = threading.Event()

    def display_loop():
        pending_frames: OrderedDict[int, tuple] = OrderedDict()

        while not stop_event.is_set():
            try:
                img, frame_index, pts_ms, queued_at = display_player.display_queue.get(
                    timeout=0.05
                )
                pending_frames[pts_ms] = (img, frame_index, queued_at)
                while len(pending_frames) > MAX_PENDING_FRAMES:
                    pending_frames.popitem(last=False)
            except queue.Empty:
                pass

            while True:
                try:
                    img, frame_index, pts_ms, queued_at = display_player.display_queue.get_nowait()
                    pending_frames[pts_ms] = (img, frame_index, queued_at)
                    while len(pending_frames) > MAX_PENDING_FRAMES:
                        pending_frames.popitem(last=False)
                except queue.Empty:
                    break

            if not pending_frames:
                continue

            sync_id, (img, display_frame_index, queued_at) = next(iter(pending_frames.items()))

            with det_lock:
                det_reply = pop_detection(sync_id)
                fallback_reply = latest_detection
                fallback_age = time.monotonic() - latest_detection_at

            waited = time.monotonic() - queued_at
            if det_reply is None and waited < SYNC_WAIT_SECONDS:
                continue

            pending_frames.pop(sync_id, None)

            display_reply = det_reply
            using_fallback = False
            if (
                display_reply is None
                and fallback_reply is not None
                and fallback_reply.detections
                and fallback_age <= LATEST_DETECTION_HOLD_SECONDS
            ):
                display_reply = fallback_reply
                using_fallback = True

            detection_frame_id = display_reply.frame_id if display_reply is not None else None
            sync_status = "fallback" if using_fallback else "synced"
            if display_reply is None:
                sync_status = "missing"

            if display_reply and display_reply.detections:
                for det in display_reply.detections:
                    geom = det.geometry
                    if geom.HasField("point"):
                        px = int(geom.point.x)
                        py = int(geom.point.y)
                        cv2.circle(img, (px, py), 10, (0, 0, 255), -1)
                        cv2.circle(img, (px, py), 12, (255, 255, 255), 2)
                        cv2.putText(
                            img,
                            f"{det.class_name} {det.score:.3f}",
                            (px + 15, py - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255),
                            1,
                        )
            else:
                cv2.putText(
                    img,
                    (
                        f"no synced detections sync_id={sync_id} "
                        f"wait={waited*1000:.0f}ms"
                    ),
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 200, 255),
                    2,
                )

            cv2.putText(
                img,
                f"display frame_index={display_frame_index} sync_id={sync_id}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                img,
                f"detection sync_id={detection_frame_id} status={sync_status}",
                (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255) if using_fallback else (0, 255, 0),
                2,
            )

            cv2.imshow("Demo Infer Client", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                stop_event.set()
                break

        cv2.destroyAllWindows()

    disp_thread = threading.Thread(target=display_loop, daemon=True)
    disp_thread.start()

    try:
        while not stop_event.is_set():
            await asyncio.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        det_task.cancel()
        display_player.close()
        video_track.close()
        await pc.close()
        await channel.close()
        stop_event.set()
        disp_thread.join(timeout=2)


def main():
    asyncio.run(
        run_client(
            server_addr=SERVER_ADDR,
            video_path=VIDEO_PATH,
            score_threshold=SCORE_THRESHOLD,
            frame_downsample=max(1, FRAME_DOWNSAMPLE),
        )
    )


if __name__ == "__main__":
    main()
