"""单条 WebRTC 流会话的连接生命周期、视频接收与检测分发。"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Tuple

from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription

from artimes_cv.inferencers.yolo import SharedYoloPointInferencer

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PendingFrame:
    frame_id: int
    pts_ms: int
    fps: float
    img: object


class WebRtcSession:
    """封装单条 WebRTC 流会话的完整生命周期。"""

    def __init__(
        self,
        stream_id: str,
        inferencer: SharedYoloPointInferencer,
        score_threshold: float,
    ):
        self.stream_id = stream_id
        self.pc = RTCPeerConnection()
        self.inferencer = inferencer
        self.score_threshold = score_threshold
        self.running = True
        self._video_tasks: set[asyncio.Task] = set()
        self._pending_frame: PendingFrame | None = None
        self._frame_ready = asyncio.Condition()

        # 消费方（gRPC StreamDetections）挂载队列
        self.detection_queues: list[asyncio.Queue] = []

        self.pc.addTransceiver("video", direction="recvonly")
        self.pc.on("track", self._on_track)

    # ── 信令接口 ────────────────────────────────────────────────────

    async def create_offer(self) -> RTCSessionDescription:
        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)
        while self.pc.iceGatheringState != "complete":
            await asyncio.sleep(0.05)
        return self.pc.localDescription

    async def set_answer(self, sdp: str, sdp_type: str) -> None:
        await self.pc.setRemoteDescription(
            RTCSessionDescription(sdp=sdp, type=sdp_type)
        )
        logger.info("Answer set for stream %s", self.stream_id)

    # ── 订阅检测结果 ────────────────────────────────────────────────

    def attach_detection_queue(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=60)
        self.detection_queues.append(q)
        return q

    def detach_detection_queue(self, q: asyncio.Queue) -> None:
        self.detection_queues.remove(q)

    # ── 视频帧处理 ──────────────────────────────────────────────────

    def _on_track(self, track: MediaStreamTrack) -> None:
        if track.kind == "video":
            logger.info("Video track received for stream %s", self.stream_id)
            task = asyncio.create_task(self._process_video(track))
            self._video_tasks.add(task)
            task.add_done_callback(self._on_video_task_done)

    def _on_video_task_done(self, task: asyncio.Task) -> None:
        self._video_tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.exception(
                "Video processing task failed for stream %s",
                self.stream_id,
                exc_info=exc,
            )

    async def _process_video(self, track: MediaStreamTrack) -> None:
        receiver_task = asyncio.create_task(self._receive_frames(track))
        processor_task = asyncio.create_task(self._run_inference_loop())

        try:
            await asyncio.gather(receiver_task, processor_task)
        except Exception:
            logger.exception(
                "Unhandled error while processing stream %s", self.stream_id
            )
            raise
        finally:
            self.running = False
            receiver_task.cancel()
            processor_task.cancel()
            async with self._frame_ready:
                self._frame_ready.notify_all()

    async def _receive_frames(self, track: MediaStreamTrack) -> None:
        frame_count = 0
        loop = asyncio.get_running_loop()
        last_fps_time = loop.time()
        fps = 0.0

        while self.running:
            try:
                frame = await track.recv()
            except Exception:
                logger.info("Track ended for stream %s", self.stream_id)
                break

            img = frame.to_ndarray(format="bgr24")
            pts_ms = self._frame_pts_ms(frame)
            frame_id = pts_ms

            frame_count += 1
            now = loop.time()
            elapsed = now - last_fps_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                last_fps_time = now

            if fps > 0.0:
                self.inferencer.set_frequency(fps)

            pending = PendingFrame(
                frame_id=frame_id,
                pts_ms=pts_ms,
                fps=fps,
                img=img,
            )
            async with self._frame_ready:
                replaced = self._pending_frame is not None
                self._pending_frame = pending
                self._frame_ready.notify()

            if replaced:
                logger.debug(
                    "Dropped stale frame before inference: stream=%s frame_id=%s pts_ms=%s",
                    self.stream_id,
                    frame_id,
                    pts_ms,
                )

        self.running = False
        async with self._frame_ready:
            self._frame_ready.notify_all()

    async def _run_inference_loop(self) -> None:
        while self.running:
            async with self._frame_ready:
                while self.running and self._pending_frame is None:
                    await self._frame_ready.wait()
                if not self.running and self._pending_frame is None:
                    return
                pending = self._pending_frame
                self._pending_frame = None

            if pending is None:
                continue

            det = await asyncio.to_thread(
                self.inferencer.infer, pending.img, self.score_threshold
            )
            if det is not None:
                self._push_detection(pending.frame_id, pending.pts_ms, det)

    @staticmethod
    def _frame_pts_ms(frame) -> int:
        if frame.pts is None or frame.time_base is None:
            return 0
        return int(round(float(frame.pts * frame.time_base) * 1000.0))

    def _push_detection(
        self,
        frame_id: int,
        pts_ms: int,
        detection: Tuple[Tuple[float, float], Tuple[float, float], float],
    ):
        payload = (self.stream_id, str(frame_id), frame_id, pts_ms, detection)
        for q in self.detection_queues:
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                pass

    # ── 清理 ────────────────────────────────────────────────────────

    async def close(self) -> None:
        self.running = False
        await self.pc.close()
