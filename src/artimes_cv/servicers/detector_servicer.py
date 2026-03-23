from __future__ import annotations

import io

import av
import cv2
import grpc
import grpc.aio
import numpy as np

from artimes_cv.protos.detector import common_pb2
from artimes_cv.protos.detector import detector_pb2
from artimes_cv.protos.detector import detector_pb2_grpc
from artimes_cv.servicers.webrtc_inference import SharedYoloPointInferencer


class DetectorServicer(detector_pb2_grpc.DetectorEngineServicer):
    def __init__(
        self,
        model_dir: str,
        device: str,
    ) -> None:
        self._shared_inferencer = SharedYoloPointInferencer(
            model_dir=model_dir,
            device=device,
        )

    async def Detect(
        self,
        request: detector_pb2.DetectRequest,
        context: grpc.aio.ServicerContext,
    ) -> detector_pb2.DetectReply:
        try:
            bgr = self._decode_frame(request.frame)
        except ValueError as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))

        detection = self._infer_detection(
            bgr=bgr,
            score_threshold=float(request.score_threshold),
        )
        detections: list[common_pb2.Detection] = []
        if detection is not None:
            detections.append(detection)

        return detector_pb2.DetectReply(
            request_id=request.request_id,
            detections=detections,
        )

    async def StreamDetect(
        self,
        request_iterator,
        context: grpc.aio.ServicerContext,
    ):
        await context.abort(grpc.StatusCode.UNIMPLEMENTED, "StreamDetect is not implemented")
        yield

    def _infer_detection(
        self,
        *,
        bgr: np.ndarray,
        score_threshold: float,
    ) -> common_pb2.Detection | None:
        height, width = bgr.shape[:2]
        px, py, score = self._shared_inferencer.infer(bgr)
        if float(score) < float(score_threshold):
            return None

        clamped_px = float(np.clip(px, 0.0, width - 1))
        clamped_py = float(np.clip(py, 0.0, height - 1))
        nx = clamped_px / width if width else 0.0
        ny = clamped_py / height if height else 0.0
        return common_pb2.Detection(
            class_name="yolo_point",
            class_id=0,
            score=float(score),
            geometry=common_pb2.DetectionGeometry(
                point=common_pb2.Point2D(x=clamped_px, y=clamped_py)
            ),
            normalized_geometry=common_pb2.DetectionGeometry(
                point=common_pb2.Point2D(x=nx, y=ny)
            ),
        )

    def _decode_frame(self, frame: detector_pb2.ImageFrame) -> np.ndarray:
        frame_format = frame.format.lower().strip()
        if frame_format == "bgr8":
            return self._decode_raw(frame.data, width=frame.width, height=frame.height, channels=3)
        if frame_format == "rgb8":
            rgb = self._decode_raw(frame.data, width=frame.width, height=frame.height, channels=3)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if frame_format in {"jpeg", "jpg", "png"}:
            image = cv2.imdecode(np.frombuffer(frame.data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Failed to decode image format: {frame.format}")
            return image
        if frame_format == "h264":
            return self._decode_h264(frame.data)
        raise ValueError(f"Unsupported frame format: {frame.format}")

    @staticmethod
    def _decode_raw(data: bytes, *, width: int, height: int, channels: int) -> np.ndarray:
        expected = int(width) * int(height) * int(channels)
        buffer = np.frombuffer(data, dtype=np.uint8)
        if buffer.size != expected:
            raise ValueError(
                f"Raw frame size mismatch: expected {expected} bytes, got {buffer.size}"
            )
        return buffer.reshape((int(height), int(width), int(channels)))

    @staticmethod
    def _decode_h264(data: bytes) -> np.ndarray:
        try:
            with av.open(io.BytesIO(data), format="h264", mode="r") as container:
                for frame in container.decode(video=0):
                    return frame.to_ndarray(format="bgr24")
        except av.AVError as exc:
            raise ValueError(f"Failed to decode h264 frame: {exc}") from exc
        raise ValueError("Failed to decode h264 frame: empty stream")
