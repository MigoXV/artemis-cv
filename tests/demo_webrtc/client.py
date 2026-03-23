"""Demo WebRTC 客户端：生成动画 → 通过 WebRTC 推流 → 接收检测结果并叠加显示。"""

import asyncio
import fractions
import math
import queue
import threading
import time

import cv2
import grpc
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from av import VideoFrame

from artimes_cv.protos.detector import common_pb2
from artimes_cv.protos.detector import webrtc_detector_pb2 as pb2
from artimes_cv.protos.detector import webrtc_detector_pb2_grpc as pb2_grpc


# ── 自定义动画视频轨道 ──────────────────────────────────────────────


class AnimationVideoTrack(MediaStreamTrack):
    """生成弹跳球 + 拖尾动画的虚拟摄像头。"""

    kind = "video"

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        super().__init__()
        self._w = width
        self._h = height
        self._fps = fps
        self._start = time.time()
        self._pts = 0
        self.display_queue: queue.Queue = queue.Queue(maxsize=2)

    async def recv(self) -> VideoFrame:
        self._pts += 1
        target = self._start + self._pts / self._fps
        wait = target - time.time()
        if wait > 0:
            await asyncio.sleep(wait)

        t = time.time() - self._start
        img = self._render(t)

        try:
            self.display_queue.put_nowait(img.copy())
        except queue.Full:
            pass

        frame = VideoFrame.from_ndarray(img, format="bgr24")
        frame.pts = self._pts
        frame.time_base = fractions.Fraction(1, self._fps)
        return frame

    def _render(self, t: float) -> np.ndarray:
        img = np.zeros((self._h, self._w, 3), dtype=np.uint8)

        # 网格
        for x in range(0, self._w, 80):
            cv2.line(img, (x, 0), (x, self._h), (30, 30, 30), 1)
        for y in range(0, self._h, 60):
            cv2.line(img, (0, y), (self._w, y), (30, 30, 30), 1)

        # 弹跳球
        cx = int(self._w / 2 + math.sin(t * 2.0) * self._w * 0.35)
        cy = int(self._h / 2 + math.cos(t * 3.0) * self._h * 0.35)
        radius = int(25 + 10 * math.sin(t * 5))
        r = int(127 + 127 * math.sin(t))
        g = int(127 + 127 * math.sin(t + 2.1))
        b = int(127 + 127 * math.sin(t + 4.2))
        cv2.circle(img, (cx, cy), radius, (b, g, r), -1)

        # 拖尾
        for i in range(1, 6):
            tt = t - i * 0.08
            tcx = int(self._w / 2 + math.sin(tt * 2.0) * self._w * 0.35)
            tcy = int(self._h / 2 + math.cos(tt * 3.0) * self._h * 0.35)
            alpha = max(0.0, 1.0 - i * 0.2)
            tr = int(radius * alpha)
            if tr > 0:
                cv2.circle(
                    img, (tcx, tcy), tr,
                    (int(b * alpha), int(g * alpha), int(r * alpha)), -1,
                )

        cv2.putText(
            img, f"t={t:.1f}s", (10, self._h - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1,
        )
        return img


# ── 主客户端逻辑 ────────────────────────────────────────────────────


async def run_client(server_addr: str = "localhost:50051"):
    channel = grpc.aio.insecure_channel(server_addr)
    stub = pb2_grpc.WebRtcDetectorEngineStub(channel)

    # 1) CreateStream — 获取 offer
    print(f"Connecting to {server_addr} ...")
    reply = await stub.CreateStream(
        pb2.CreateStreamRequest(
            config=common_pb2.StreamConfig(
                video_codec="vp8", width=640, height=480,
            )
        )
    )
    stream_id = reply.stream_id
    print(f"Stream created: {stream_id}")

    # 2) 本地 PeerConnection：添加视频轨 → 设置 offer → 创建 answer
    pc = RTCPeerConnection()
    video_track = AnimationVideoTrack()
    pc.addTrack(video_track)

    await pc.setRemoteDescription(
        RTCSessionDescription(sdp=reply.offer.sdp, type=reply.offer.type)
    )
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    while pc.iceGatheringState != "complete":
        await asyncio.sleep(0.1)

    # 3) UpdateStream — 发送 answer
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

    # 4) 后台订阅检测结果
    latest_detection = None
    det_lock = threading.Lock()

    async def recv_detections():
        nonlocal latest_detection
        try:
            async for det_reply in stub.StreamDetections(
                pb2.StreamDetectionsRequest(stream_id=stream_id)
            ):
                with det_lock:
                    latest_detection = det_reply
        except Exception as e:
            print(f"Detection stream ended: {e}")

    det_task = asyncio.ensure_future(recv_detections())

    # 5) 显示线程：动画 + 检测叠加
    stop_event = threading.Event()

    def display_loop():
        while not stop_event.is_set():
            try:
                img = video_track.display_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            # 绘制菱形路径参考线
            cx, cy = img.shape[1] // 2, img.shape[0] // 2
            r = int(0.3 * min(img.shape[1], img.shape[0]))
            diamond = np.array([
                [cx, cy - r], [cx + r, cy], [cx, cy + r], [cx - r, cy],
            ], np.int32)
            cv2.polylines(img, [diamond], True, (80, 80, 80), 1, cv2.LINE_AA)

            # 叠加检测点
            with det_lock:
                det = latest_detection
            if det and det.detections:
                for d in det.detections:
                    geom = d.geometry
                    if geom.HasField("point"):
                        px, py = int(geom.point.x), int(geom.point.y)
                        cv2.circle(img, (px, py), 10, (0, 0, 255), -1)
                        cv2.circle(img, (px, py), 12, (255, 255, 255), 2)
                        cv2.putText(
                            img, d.class_name, (px + 15, py - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
                        )

            cv2.imshow("Demo Client", img)
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
        await pc.close()
        await channel.close()
        stop_event.set()
        disp_thread.join(timeout=2)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Demo WebRTC 客户端")
    parser.add_argument("--server", default="localhost:50051", help="gRPC 服务端地址")
    args = parser.parse_args()
    asyncio.run(run_client(args.server))


if __name__ == "__main__":
    main()
