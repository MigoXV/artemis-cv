import asyncio
import logging
from collections.abc import AsyncIterator

import grpc
import grpc.aio

from artimes_cv.protos.detector import webrtc_detector_pb2 as pb2
from artimes_cv.protos.detector import webrtc_detector_pb2_grpc as pb2_grpc
from artimes_cv.webrtc import WebRtcSessionManager

logger = logging.getLogger(__name__)


class WebRtcDetectorServicer(pb2_grpc.WebRtcDetectorEngineServicer):
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
        self._manager = WebRtcSessionManager(
            model_dir=model_dir,
            device=device,
            enable_display=enable_display,
            initial_frequency=initial_frequency,
            min_cutoff=min_cutoff,
            beta=beta,
            d_cutoff=d_cutoff,
        )

    async def CreateStream(
        self,
        request: pb2.CreateStreamRequest,
        context: grpc.aio.ServicerContext,
    ) -> pb2.CreateStreamReply:
        session = self._manager.create(request.config)
        local_desc = await session.create_offer()

        return pb2.CreateStreamReply(
            stream_id=session.stream_id,
            offer=pb2.SessionDescription(
                type=local_desc.type,
                sdp=local_desc.sdp,
            ),
        )

    async def UpdateStream(
        self,
        request: pb2.StreamSignal,
        context: grpc.aio.ServicerContext,
    ) -> pb2.UpdateStreamReply:
        session = self._manager.get(request.stream_id)
        if session is None:
            await context.abort(grpc.StatusCode.NOT_FOUND, "Stream not found")

        if request.WhichOneof("signal") == "answer":
            await session.set_answer(request.answer.sdp, request.answer.type)

        return pb2.UpdateStreamReply()

    async def StreamDetections(
        self,
        request: pb2.StreamDetectionsRequest,
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[pb2.StreamDetectionsReply]:
        session = self._manager.get(request.stream_id)
        if session is None:
            await context.abort(grpc.StatusCode.NOT_FOUND, "Stream not found")

        det_queue = session.attach_detection_queue()
        try:
            while session.running:
                try:
                    stream_id, req_id, frame_id, pts_ms, det = await asyncio.wait_for(
                        det_queue.get(), timeout=1.0
                    )
                    yield pb2.StreamDetectionsReply(
                        stream_id=stream_id,
                        request_id=req_id,
                        pts_ms=pts_ms,
                        detections=[det] if det is not None else [],
                        frame_id=frame_id,
                    )
                except asyncio.TimeoutError:
                    continue
        except Exception:
            logger.exception("StreamDetections failed for stream %s", request.stream_id)
            raise
        finally:
            session.detach_detection_queue(det_queue)
