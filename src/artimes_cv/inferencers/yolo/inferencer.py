from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import AutoModel


class YoloPointInferencer:
    """基于 YOLOv8 CenterPoint 的单点推理器。"""

    DEFAULT_IMGSZ = 640
    DEFAULT_PAD_VALUE = 114

    def __init__(
        self,
        model_dir: str | Path,
        device: str | torch.device = "cpu",
        imgsz: int = DEFAULT_IMGSZ,
    ) -> None:
        self.imgsz = imgsz
        self.device = torch.device(device)

        model_dir = Path(model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(f"模型目录不存在: {model_dir}")

        self.model = AutoModel.from_pretrained(str(model_dir), trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # 图像预处理
    # ------------------------------------------------------------------

    def _letterbox(
        self, image: np.ndarray
    ) -> tuple[np.ndarray, float, int, int, int, int]:
        """将图像等比缩放并填充为正方形，返回 (canvas, scale, pad_w, pad_h, orig_w, orig_h)。"""
        h, w = image.shape[:2]
        scale = min(self.imgsz / h, self.imgsz / w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)

        canvas = np.full(
            (self.imgsz, self.imgsz, 3), self.DEFAULT_PAD_VALUE, dtype=np.uint8
        )
        pad_w = (self.imgsz - nw) // 2
        pad_h = (self.imgsz - nh) // 2
        canvas[pad_h : pad_h + nh, pad_w : pad_w + nw] = resized
        return canvas, scale, pad_w, pad_h, w, h

    def _preprocess(
        self, bgr: np.ndarray
    ) -> tuple[torch.Tensor, float, int, int, int, int]:
        """BGR → letterbox RGB 张量，同时返回还原坐标所需的元信息。"""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        canvas, scale, pad_w, pad_h, orig_w, orig_h = self._letterbox(rgb)
        tensor = (
            torch.from_numpy(
                np.transpose(canvas.astype(np.float32) / 255.0, (2, 0, 1))
            )
            .unsqueeze(0)
            .to(self.device)
        )
        return tensor, scale, pad_w, pad_h, orig_w, orig_h

    # ------------------------------------------------------------------
    # 后处理
    # ------------------------------------------------------------------

    @staticmethod
    @torch.no_grad()
    def _decode_single_point(
        outputs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """从多尺度热图输出中解码最优点，返回归一化坐标与置信度。"""
        best_points: list[torch.Tensor] = []
        best_scores: list[torch.Tensor] = []

        batch_size = outputs[0][0].shape[0]
        for batch_idx in range(batch_size):
            best_score = None
            best_point = None

            for pred_heatmap, pred_offset in outputs:
                heatmap = pred_heatmap[batch_idx, 0].sigmoid()
                h, w = heatmap.shape
                flat_idx = int(torch.argmax(heatmap).item())
                y = flat_idx // w
                x = flat_idx % w

                score = heatmap[y, x]
                dx = pred_offset[batch_idx, 0, y, x].sigmoid()
                dy = pred_offset[batch_idx, 1, y, x].sigmoid()
                point = torch.stack(((x + dx) / w, (y + dy) / h))

                if best_score is None or bool(score > best_score):
                    best_score = score
                    best_point = point

            best_points.append(best_point)
            best_scores.append(best_score)

        return torch.stack(best_points, dim=0), torch.stack(best_scores, dim=0)

    def _postprocess(
        self,
        outputs,
        scale: float,
        pad_w: int,
        pad_h: int,
        orig_w: int,
        orig_h: int,
    ) -> tuple[float, float, float]:
        """将模型输出解码为原图像素坐标和置信度，返回 (px, py, score)。"""
        pred_points, pred_scores = self._decode_single_point(outputs)

        nx_lb = float(pred_points[0, 0])
        ny_lb = float(pred_points[0, 1])
        score = float(pred_scores[0])

        px = (nx_lb * self.imgsz - pad_w) / (orig_w * scale) * orig_w
        py = (ny_lb * self.imgsz - pad_h) / (orig_h * scale) * orig_h
        return px, py, score

    # ------------------------------------------------------------------
    # 公开推理接口
    # ------------------------------------------------------------------

    @torch.no_grad()
    def infer(self, bgr: np.ndarray) -> tuple[float, float, float]:
        """对单张 BGR 图像进行推理，返回 (px, py, score)（像素坐标）。

        Args:
            bgr: BGR 格式的 numpy 图像，形状为 (H, W, 3)。

        Returns:
            (px, py, score) — 预测点在原图中的像素坐标及置信度。
        """
        tensor, scale, pad_w, pad_h, orig_w, orig_h = self._preprocess(bgr)
        outputs = self.model(pixel_values=tensor)
        return self._postprocess(outputs, scale, pad_w, pad_h, orig_w, orig_h)

    @torch.no_grad()
    def infer_normalized(self, bgr: np.ndarray) -> tuple[float, float, float]:
        """对单张 BGR 图像进行推理，返回 (nx, ny, score)（归一化坐标 0~1）。"""
        px, py, score = self.infer(bgr)
        h, w = bgr.shape[:2]
        return px / w, py / h, score
