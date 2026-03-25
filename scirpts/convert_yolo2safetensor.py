import argparse
import json
from pathlib import Path

import torch

from _lockin_yolov8n import YOLOv8CenterPoint


REMOTE_INIT = """from .configuration_yolov8_centerpoint import YOLOv8CenterPointConfig
from .modeling_yolov8_centerpoint import YOLOv8CenterPointModel

__all__ = ["YOLOv8CenterPointConfig", "YOLOv8CenterPointModel"]
"""

REMOTE_CONFIG = """from transformers import PretrainedConfig


class YOLOv8CenterPointConfig(PretrainedConfig):
    model_type = "yolov8_centerpoint"

    def __init__(
        self,
        yolo_backbone: str = "yolov8n.yaml",
        num_labels: int = 1,
        id2label: dict | None = None,
        label2id: dict | None = None,
        **kwargs,
    ) -> None:
        if id2label is None:
            id2label = {"0": "center"}
        if label2id is None:
            label2id = {"center": 0}

        super().__init__(
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            **kwargs,
        )
        self.yolo_backbone = yolo_backbone
"""

REMOTE_MODELING = """from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from .configuration_yolov8_centerpoint import YOLOv8CenterPointConfig


@contextmanager
def _suspend_detect_bias_init():
    from ultralytics.nn.modules.head import Detect

    original = Detect.bias_init
    Detect.bias_init = lambda self: None
    try:
        yield
    finally:
        Detect.bias_init = original


class ConvBNAct(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, k // 2, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class PointHead(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    ConvBNAct(c, c, 3, 1),
                    nn.Conv2d(c, 3, kernel_size=1, stride=1, padding=0),
                )
                for c in channels
            ]
        )

    def forward(self, features):
        outputs = []
        for feature, branch in zip(features, self.branches):
            pred = branch(feature)
            outputs.append((pred[:, :1], pred[:, 1:3]))
        return outputs


class YOLOv8CenterPointModel(PreTrainedModel):
    config_class = YOLOv8CenterPointConfig
    base_model_prefix = "model"
    main_input_name = "pixel_values"

    def __init__(self, config: YOLOv8CenterPointConfig):
        super().__init__(config)
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "This module requires `ultralytics`. Install it with `pip install ultralytics`."
            ) from exc

        with _suspend_detect_bias_init():
            base_model = YOLO(config.yolo_backbone).model
        self.backbone_neck = base_model.model
        self.save = list(base_model.save)
        self.names = dict(config.id2label)

        detect = self.backbone_neck[-1]
        channels = [branch[0].conv.in_channels for branch in detect.cv2]
        self.head = PointHead(channels)
        self._strides = None
        self.post_init()

    @property
    def strides(self):
        if self._strides is None:
            stride = self.backbone_neck[-1].stride
            if isinstance(stride, torch.Tensor):
                if stride.is_meta:
                    raise RuntimeError("YOLO strides are not materialized yet.")
                stride = stride.tolist()
            self._strides = tuple(int(s) for s in stride)
        return self._strides

    def _forward_backbone_neck(self, x: torch.Tensor):
        outputs = []
        features = None

        for module in self.backbone_neck:
            if module.f != -1:
                if isinstance(module.f, int):
                    module_input = outputs[module.f]
                else:
                    module_input = [x if src == -1 else outputs[src] for src in module.f]
            else:
                module_input = x

            if module is self.backbone_neck[-1]:
                features = module_input if isinstance(module_input, list) else [module_input]
                break

            x = module(module_input)
            outputs.append(x if module.i in self.save else None)

        if features is None:
            raise RuntimeError("Failed to collect backbone/neck features from YOLOv8.")
        return features

    def forward(self, pixel_values: torch.Tensor, **kwargs):
        del kwargs
        return self.head(self._forward_backbone_neck(pixel_values))
"""

REMOTE_UTILS = """import cv2
import numpy as np
import torch


def letterbox(
    image: np.ndarray, size: int, pad_value: int = 114
) -> tuple[np.ndarray, float, int, int, int, int]:
    h, w = image.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((size, size, 3), pad_value, dtype=np.uint8)
    pad_w = (size - nw) // 2
    pad_h = (size - nh) // 2
    canvas[pad_h : pad_h + nh, pad_w : pad_w + nw] = resized
    return canvas, scale, pad_w, pad_h, w, h


@torch.no_grad()
def decode_single_point(outputs):
    best_points = []
    best_scores = []

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
"""


def build_config(yolo_backbone: str) -> dict:
    return {
        "architectures": ["YOLOv8CenterPointModel"],
        "model_type": "yolov8_centerpoint",
        "yolo_backbone": yolo_backbone,
        "num_labels": 1,
        "id2label": {"0": "center"},
        "label2id": {"center": 0},
        "auto_map": {
            "AutoConfig": "configuration_yolov8_centerpoint.YOLOv8CenterPointConfig",
            "AutoModel": "modeling_yolov8_centerpoint.YOLOv8CenterPointModel",
        },
    }


def write_remote_code(output_dir: Path) -> None:
    files = {
        "__init__.py": REMOTE_INIT,
        "configuration_yolov8_centerpoint.py": REMOTE_CONFIG,
        "modeling_yolov8_centerpoint.py": REMOTE_MODELING,
        "utils.py": REMOTE_UTILS,
    }
    for filename, content in files.items():
        (output_dir / filename).write_text(content, encoding="utf-8")


def write_readme(output_dir: Path) -> None:
    (output_dir / "README.md").write_text(
        """# artimes-yolov8n-260323-1402

A local converted HF style repository for YOLOv8CenterPoint.

## Contents
- config.json
- pytorch_model.bin
- model.safetensors (optional)
- configuration_yolov8_centerpoint.py
- modeling_yolov8_centerpoint.py
- utils.py
- __init__.py

## Usage
```python
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "model-bin/artimes-yolov8n-260323-1402",
    trust_remote_code=True,
)
```
""",
        encoding="utf-8",
    )


def convert_pt_to_hf(
    source_ckpt: Path,
    yolo_backbone: Path,
    output_dir: Path,
    save_safetensors: bool = True,
) -> None:
    if not source_ckpt.exists():
        raise FileNotFoundError(f"source checkpoint not found: {source_ckpt}")
    if not yolo_backbone.exists():
        raise FileNotFoundError(f"YOLO backbone weights file not found: {yolo_backbone}")

    ckpt = torch.load(source_ckpt, map_location="cpu")
    if "model_state" not in ckpt:
        raise ValueError("Checkpoint does not contain 'model_state'.")

    model = YOLOv8CenterPoint(weights=str(yolo_backbone))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)

    write_remote_code(output_dir)
    write_readme(output_dir)

    config = build_config(yolo_backbone.with_suffix(".yaml").name)
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    state_dict = model.state_dict()
    torch.save(state_dict, output_dir / "pytorch_model.bin")

    if save_safetensors:
        from safetensors.torch import save_file

        save_file(state_dict, str(output_dir / "model.safetensors"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert YOLO lockin model checkpoint to HF-style repository")
    parser.add_argument("--source", type=Path, default=Path("model-bin/point-yolov8n-01/best.pt"), help="Source pt checkpoint (from model-bin)")
    parser.add_argument("--yolo-backbone", type=Path, default=Path("model-bin/point-yolov8n-01/yolov8n.pt"), help="Ultralytics YOLOv8 backbone weights")
    parser.add_argument("--output", type=Path, default=Path("model-bin/artimes-yolov8n-260323-1555"), help="Output HF-style directory")
    parser.add_argument("--no-safetensors", action="store_true", help="Skip saving model.safetensors")
    args = parser.parse_args()

    convert_pt_to_hf(
        source_ckpt=args.source,
        yolo_backbone=args.yolo_backbone,
        output_dir=args.output,
        save_safetensors=not args.no_safetensors,
    )
    print(f"Converted checkpoint from {args.source} into HF-style folder {args.output}.")


if __name__ == "__main__":
    main()
