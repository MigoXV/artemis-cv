"""Demo WebRTC 推理客户端：发送本地视频，接收服务端 YOLO 点检测结果并叠加显示。"""

import asyncio
import fractions
import queue
import threading
import time
from collections import OrderedDict
from pathlib import Path

import av
import cv2
import grpc
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from av import VideoFrame

from artimes_cv.protos.detector import common_pb2
from artimes_cv.protos.detector import webrtc_detector_pb2 as pb2
from artimes_cv.protos.detector import webrtc_detector_pb2_grpc as pb2_grpc

ROOT = Path(__file__).resolve().parents[2]
VIDEO_PATH = ROOT / "data-bin" / "videos" / "slow.mp4"
SYNC_WAIT_SECONDS = 0.75
MAX_PENDING_FRAMES = 120
MAX_PENDING_DETECTIONS = 240
LATEST_DETECTION_HOLD_SECONDS = 1.0


class VideoFileTrack(MediaStreamTrack):
    """将本地视频文件包装成 WebRTC 视频轨。"""

    kind = "video"

    def __init__(self, video_path: Path, loop: bool = True):
        super().__init__()
        self.video_path = video_path
        self.loop = loop
        self.display_queue: queue.Queue = queue.Queue(maxsize=MAX_PENDING_FRAMES)
        self._container: av.container.InputContainer | None = None
        self._frames = None
        self._fps = 30.0
        self._time_base = fractions.Fraction(1, 30)
        self._start = time.time()
        self._pts = 0
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

    async def recv(self) -> VideoFrame:
        self._pts += 1
        target = self._start + self._pts / self._fps
        wait = target - time.time()
        if wait > 0:
            await asyncio.sleep(wait)

        while True:
            try:
                av_frame = next(self._frames)
                break
            except StopIteration:
                if not self.loop:
                    raise EOFError("video ended")
                self._open_video()

        bgr = av_frame.to_ndarray(format="bgr24")
        pts_ms = self._pts_to_ms(self._pts, self._time_base)

        try:
            self.display_queue.put_nowait(
                (bgr.copy(), self._pts, pts_ms, time.monotonic())
            )
        except queue.Full:
            pass

        frame = VideoFrame.from_ndarray(bgr, format="bgr24")
        frame.pts = self._pts
        frame.time_base = self._time_base
        return frame

    @staticmethod
    def _pts_to_ms(pts: int, time_base: fractions.Fraction) -> int:
        return int(round(float(pts * time_base) * 1000.0))

    def close(self) -> None:
        if self._container is not None:
            self._container.close()
            self._container = None


async def run_client(
    server_addr: str = "localhost:50051",
    video_path: Path = VIDEO_PATH,
    score_threshold: float = 0.0,
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
    video_track = VideoFileTrack(video_path=video_path, loop=True)
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
                img, frame_index, pts_ms, queued_at = video_track.display_queue.get(
                    timeout=0.05
                )
                pending_frames[pts_ms] = (img, frame_index, queued_at)
                while len(pending_frames) > MAX_PENDING_FRAMES:
                    pending_frames.popitem(last=False)
            except queue.Empty:
                pass

            while True:
                try:
                    img, frame_index, pts_ms, queued_at = video_track.display_queue.get_nowait()
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
        video_track.close()
        await pc.close()
        await channel.close()
        stop_event.set()
        disp_thread.join(timeout=2)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Demo WebRTC 推理客户端")
    parser.add_argument("--server", default="localhost:50052", help="gRPC 服务端地址")
    parser.add_argument("--video", type=Path, default=VIDEO_PATH, help="推流视频路径")
    parser.add_argument("--score-threshold", type=float, default=0.0, help="服务端分数阈值")
    args = parser.parse_args()
    asyncio.run(
        run_client(
            server_addr=args.server,
            video_path=args.video,
            score_threshold=args.score_threshold,
        )
    )


if __name__ == "__main__":
    main()
