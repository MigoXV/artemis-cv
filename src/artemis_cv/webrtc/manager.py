"""管理多个 WebRTC 会话与共享推理器。"""

import logging
import uuid

from artemis_cv.inferencers.yolo import SharedYoloPointInferencer
from artemis_cv.protos.detector import common_pb2
from artemis_cv.webrtc.session import WebRtcSession

logger = logging.getLogger(__name__)


class WebRtcSessionManager:
    def __init__(
        self,
        model_dir: str,
        device: str,
        initial_frequency: float,
        min_cutoff: float,
        beta: float,
        d_cutoff: float,
    ):
        self._sessions: dict[str, WebRtcSession] = {}

        self.inferencer = SharedYoloPointInferencer(
            model_dir=model_dir,
            device=device,
            frequency=initial_frequency,
            min_cutoff=min_cutoff,
            beta=beta,
            d_cutoff=d_cutoff,
        )

    # ── 会话管理 ────────────────────────────────────────────────────
    def create(self, config: common_pb2.StreamConfig | None = None) -> WebRtcSession:
        stream_id = str(uuid.uuid4())
        score_threshold = float(config.score_threshold) if config else 0.0
        session = WebRtcSession(stream_id, inferencer=self.inferencer, score_threshold=score_threshold)
        self._sessions[stream_id] = session
        logger.info("Session created: %s", stream_id)
        return session

    def get(self, stream_id: str) -> WebRtcSession | None:
        return self._sessions.get(stream_id)

    async def remove(self, stream_id: str) -> None:
        session = self._sessions.pop(stream_id, None)
        if session:
            await session.close()
