"""单条 WebRTC 流会话的连接生命周期与视频帧处理。"""

import asyncio
import logging
import queue
import time
import uuid

from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack

from artimes_cv.protos.detector import common_pb2
from artimes_cv.servicers.stub.stub_inference import StubInference

logger = logging.getLogger(__name__)


class WebRtcSession:
    """封装单条 WebRTC 流会话的完整生命周期。"""

    def __init__(self, stream_id: str):
        self.stream_id = stream_id
        self.pc = RTCPeerConnection()
        self.inference = StubInference()
        self.running = True

        # 消费方（gRPC StreamDetections）挂载队列
        self.detection_queues: list[asyncio.Queue] = []
        # 显示线程消费队列
        self.display_queue: queue.Queue = queue.Queue(maxsize=2)

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
            asyncio.ensure_future(self._process_video(track))

    async def _process_video(self, track: MediaStreamTrack) -> None:
        frame_count = 0
        last_fps_time = time.monotonic()
        fps = 0.0

        while self.running:
            try:
                frame = await track.recv()
            except Exception:
                logger.info("Track ended for stream %s", self.stream_id)
                break

            img = frame.to_ndarray(format="bgr24")
            h, w = img.shape[:2]

            frame_count += 1
            now = time.monotonic()
            elapsed = now - last_fps_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                last_fps_time = now

            det = self.inference.get_detection(w, h)

            try:
                self.display_queue.put_nowait((img, fps, det))
            except queue.Full:
                pass

            self._push_detection(det)

        self.running = False

    def _push_detection(self, det: common_pb2.Detection) -> None:
        if not self.detection_queues:
            return
        payload = (self.stream_id, str(uuid.uuid4()), int(time.time() * 1000), det)
        for q in self.detection_queues:
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                pass

    # ── 清理 ────────────────────────────────────────────────────────

    async def close(self) -> None:
        self.running = False
        await self.pc.close()
