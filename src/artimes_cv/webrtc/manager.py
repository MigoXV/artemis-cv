"""管理多个 WebRTC 会话，并统一维护 OpenCV 显示线程。"""

import logging
import queue
import threading
import time
import uuid

import cv2

from artimes_cv.protos.detector import common_pb2
from artimes_cv.servicers.webrtc_inference import (
    SharedYoloPointInferencer,
    WebRtcVisionInference,
)
from artimes_cv.webrtc.session import WebRtcSession

logger = logging.getLogger(__name__)


class WebRtcSessionManager:
    def __init__(
        self,
        model_dir: str,
        device: str,
        enable_display: bool,
        initial_frequency: float,
        min_cutoff: float,
        beta: float,
        d_cutoff: float,
    ):
        self._sessions: dict[str, WebRtcSession] = {}
        self._enable_display = enable_display
        self._display_running = False
        self._display_thread: threading.Thread | None = None
        self._initial_frequency = initial_frequency
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._d_cutoff = d_cutoff
        self._shared_inferencer = SharedYoloPointInferencer(
            model_dir=model_dir,
            device=device,
        )

    # ── 会话管理 ────────────────────────────────────────────────────

    def create(self, config: common_pb2.StreamConfig | None = None) -> WebRtcSession:
        stream_id = str(uuid.uuid4())
        score_threshold = float(config.score_threshold) if config else 0.0
        inference = WebRtcVisionInference(
            self._shared_inferencer,
            score_threshold=score_threshold,
            frequency=self._initial_frequency,
            min_cutoff=self._min_cutoff,
            beta=self._beta,
            d_cutoff=self._d_cutoff,
        )
        session = WebRtcSession(stream_id, inference=inference)
        self._sessions[stream_id] = session
        if self._enable_display:
            self._ensure_display_thread()
        logger.info("Session created: %s", stream_id)
        return session

    def get(self, stream_id: str) -> WebRtcSession | None:
        return self._sessions.get(stream_id)

    async def remove(self, stream_id: str) -> None:
        session = self._sessions.pop(stream_id, None)
        if session:
            await session.close()

    # ── 显示线程 ────────────────────────────────────────────────────

    def _ensure_display_thread(self) -> None:
        if self._display_running:
            return
        self._display_running = True
        self._display_thread = threading.Thread(
            target=self._display_loop, daemon=True
        )
        self._display_thread.start()

    def _display_loop(self) -> None:
        while self._display_running:
            displayed = False
            for session in list(self._sessions.values()):
                try:
                    img, fps, _pts_ms, _frame_id, det = session.display_queue.get_nowait()
                except queue.Empty:
                    continue

                cv2.putText(
                    img, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2,
                )
                if det is not None:
                    g = det.geometry
                    if g.HasField("point"):
                        px, py = int(g.point.x), int(g.point.y)
                        cv2.circle(img, (px, py), 8, (0, 0, 255), -1)
                        cv2.circle(img, (px, py), 10, (255, 255, 255), 2)
                cv2.imshow(f"WebRTC [{session.stream_id[:8]}]", img)
                displayed = True

            cv2.waitKey(1)
            if not displayed:
                time.sleep(0.005)

        cv2.destroyAllWindows()

    def stop(self) -> None:
        self._display_running = False
        if self._display_thread:
            self._display_thread.join(timeout=2)
