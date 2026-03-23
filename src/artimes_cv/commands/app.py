import asyncio
import logging

import grpc
import typer

from artimes_cv.protos.detector import webrtc_detector_pb2_grpc as pb2_grpc
from artimes_cv.servicers.stub.servicer import WebRtcDetectorServicer

app = typer.Typer(name="artimes-cv")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="监听地址"),
    port: int = typer.Option(50051, help="监听端口"),
):
    """启动 WebRTC 检测 gRPC 服务端。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    async def _run():
        server = grpc.aio.server()
        pb2_grpc.add_WebRtcDetectorEngineServicer_to_server(
            WebRtcDetectorServicer(), server
        )
        listen_addr = f"{host}:{port}"
        server.add_insecure_port(listen_addr)
        await server.start()
        typer.echo(f"gRPC server listening on {listen_addr}")
        await server.wait_for_termination()

    asyncio.run(_run())


if __name__ == "__main__":
    app()
