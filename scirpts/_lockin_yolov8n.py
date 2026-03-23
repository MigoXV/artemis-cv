import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    canvas[pad_h:pad_h + nh, pad_w:pad_w + nw] = resized
    return canvas, scale, pad_w, pad_h, w, h


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


class YOLOv8CenterPoint(nn.Module):
    def __init__(self, weights: str = "yolov8n.pt"):
        super().__init__()
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "This script requires `ultralytics`. Install it with `pip install ultralytics`."
            ) from exc

        base_model = YOLO(weights).model
        self.backbone_neck = base_model.model
        self.save = list(base_model.save)
        self.names = {0: "center"}

        detect = self.backbone_neck[-1]
        channels = [branch[0].conv.in_channels for branch in detect.cv2]
        self.head = PointHead(channels)
        self.strides = tuple(int(s) for s in detect.stride.tolist())

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

    def forward(self, x: torch.Tensor):
        return self.head(self._forward_backbone_neck(x))


class PointLoss(nn.Module):
    def __init__(self, lambda_heatmap: float = 1.0, lambda_offset: float = 5.0, pos_weight: float = 30.0):
        super().__init__()
        self.lambda_heatmap = lambda_heatmap
        self.lambda_offset = lambda_offset
        self.pos_weight = pos_weight

    def forward(self, outputs, targets):
        heatmap_loss = torch.tensor(0.0, device=targets.device)
        offset_loss = torch.tensor(0.0, device=targets.device)

        for pred_heatmap, pred_offset in outputs:
            _, _, h, w = pred_heatmap.shape
            target_heatmap = torch.zeros_like(pred_heatmap)
            target_offset = torch.zeros_like(pred_offset)
            pos_mask = torch.zeros((targets.shape[0], h, w), dtype=torch.bool, device=targets.device)

            gx = targets[:, 0] * w
            gy = targets[:, 1] * h
            gi = gx.clamp(0, w - 1e-6).long()
            gj = gy.clamp(0, h - 1e-6).long()

            for batch_idx in range(targets.shape[0]):
                target_heatmap[batch_idx, 0, gj[batch_idx], gi[batch_idx]] = 1.0
                target_offset[batch_idx, 0, gj[batch_idx], gi[batch_idx]] = gx[batch_idx] - gi[batch_idx].float()
                target_offset[batch_idx, 1, gj[batch_idx], gi[batch_idx]] = gy[batch_idx] - gj[batch_idx].float()
                pos_mask[batch_idx, gj[batch_idx], gi[batch_idx]] = True

            heatmap_loss = heatmap_loss + F.binary_cross_entropy_with_logits(
                pred_heatmap,
                target_heatmap,
                pos_weight=torch.full((1,), self.pos_weight, device=targets.device),
            )

            pred_offset_sigmoid = pred_offset.sigmoid().permute(0, 2, 3, 1)
            target_offset_perm = target_offset.permute(0, 2, 3, 1)
            if pos_mask.any():
                offset_loss = offset_loss + F.smooth_l1_loss(
                    pred_offset_sigmoid[pos_mask],
                    target_offset_perm[pos_mask],
                    reduction="mean",
                )

        total = self.lambda_heatmap * heatmap_loss + self.lambda_offset * offset_loss
        stats = {
            "loss": float(total.item()),
            "heatmap_loss": float(heatmap_loss.item()),
            "offset_loss": float(offset_loss.item()),
        }
        return total, stats


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


__all__ = ["YOLOv8CenterPoint", "PointLoss", "decode_single_point", "letterbox"]
