import asyncio
import logging
from pathlib import Path

import grpc
import typer

from artimes_cv.protos.detector import webrtc_detector_pb2_grpc as pb2_grpc
from artimes_cv.servicers.webrtc_servicer import WebRtcDetectorServicer

app = typer.Typer(name="artimes-cv")
DEFAULT_MODEL_DIR = (
    Path(__file__).resolve().parents[3] / "model-bin" / "artimes-yolov8n-260323-1629"
)


@app.command()
def serve(
    host: str = typer.Option(
        "0.0.0.0",
        envvar="ARTIMES_CV_HOST",
        help="监听地址",
    ),
    port: int = typer.Option(
        50051,
        envvar="ARTIMES_CV_PORT",
        help="监听端口",
    ),
    model_dir: str = typer.Option(
        str(DEFAULT_MODEL_DIR),
        envvar="ARTIMES_CV_MODEL_DIR",
        help="YOLO 模型目录",
    ),
    device: str = typer.Option(
        "cpu",
        envvar="ARTIMES_CV_DEVICE",
        help="推理设备，例如 cpu / cuda:0",
    ),
    initial_frequency: float = typer.Option(
        30.0,
        envvar="ARTIMES_CV_INITIAL_FREQUENCY",
        help="One Euro Filter 初始频率",
    ),
    min_cutoff: float = typer.Option(
        1.2,
        envvar="ARTIMES_CV_MIN_CUTOFF",
        help="One Euro Filter min_cutoff",
    ),
    beta: float = typer.Option(
        0.08,
        envvar="ARTIMES_CV_BETA",
        help="One Euro Filter beta",
    ),
    d_cutoff: float = typer.Option(
        1.0,
        envvar="ARTIMES_CV_D_CUTOFF",
        help="One Euro Filter d_cutoff",
    ),
):
    """启动 WebRTC 检测 gRPC 服务端。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    async def _run():
        server = grpc.aio.server()
        pb2_grpc.add_WebRtcDetectorEngineServicer_to_server(
            WebRtcDetectorServicer(
                model_dir=model_dir,
                device=device,
                initial_frequency=initial_frequency,
                min_cutoff=min_cutoff,
                beta=beta,
                d_cutoff=d_cutoff,
            ),
            server,
        )
        listen_addr = f"{host}:{port}"
        server.add_insecure_port(listen_addr)
        await server.start()
        typer.echo(f"gRPC server listening on {listen_addr}")
        await server.wait_for_termination()

    asyncio.run(_run())


if __name__ == "__main__":
    app()
