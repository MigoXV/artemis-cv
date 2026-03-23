import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# PyTorch 2.6+ 默认 weights_only=True，patch 以兼容旧版 ultralytics checkpoint
_orig_torch_load = torch.load
torch.load = lambda *a, **kw: _orig_torch_load(*a, **{**kw, "weights_only": False})

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scirpts._lockin_yolov8n import YOLOv8CenterPoint, decode_single_point, letterbox

MODEL_PATH = ROOT / 'model-bin' / 'point-yolov8n-01' / 'best.pt'
PRETRAINED_PATH = ROOT / 'model-bin' / 'point-yolov8n-01' / 'yolov8n.pt'
SCREENSHOTS_DIR = ROOT / 'data-bin' / 'screenshots'
OUTPUT_DIR = ROOT / 'data-bin' / 'screenshot-outputs'
IMGSZ = 640
DEVICE = torch.device('cpu')


def draw_point(image, nx, ny, score):
    """在图像上绘制归一化坐标 (nx, ny) 对应的点。"""
    h, w = image.shape[:2]
    cx, cy = int(nx * w), int(ny * h)
    vis = image.copy()
    cv2.circle(vis, (cx, cy), 6, (0, 0, 255), -1)
    cv2.circle(vis, (cx, cy), 8, (255, 255, 255), 2)
    cv2.putText(vis, f'{score:.3f}', (cx + 10, cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return vis


if __name__ == '__main__':
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 构建模型结构（需要 yolov8n.pt 初始化 backbone/neck 架构），然后加载训练权重
    model = YOLOv8CenterPoint(weights=str(PRETRAINED_PATH)).to(DEVICE)
    ckpt = _orig_torch_load(str(MODEL_PATH), map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f'模型加载完成  epoch={ckpt["epoch"]}  '
          f'val_px={ckpt["metrics"]["mean_pixel_dist"]:.2f}')

    image_paths = sorted(SCREENSHOTS_DIR.glob('*.png'))
    if not image_paths:
        raise FileNotFoundError(f'未在 {SCREENSHOTS_DIR} 找到 PNG 图片')

    for img_path in image_paths:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f'警告：无法读取 {img_path}，跳过')
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        canvas, scale, pad_w, pad_h, orig_w, orig_h = letterbox(rgb, IMGSZ)

        tensor = torch.from_numpy(
            np.transpose(canvas.astype(np.float32) / 255.0, (2, 0, 1))
        ).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(tensor)
        pred_points, pred_scores = decode_single_point(outputs)

        # 归一化坐标 (letterboxed) -> 原图归一化坐标
        nx_lb = float(pred_points[0, 0])
        ny_lb = float(pred_points[0, 1])
        score = float(pred_scores[0])

        px = (nx_lb * IMGSZ - pad_w) / (orig_w * scale) * orig_w
        py = (ny_lb * IMGSZ - pad_h) / (orig_h * scale) * orig_h
        nx_orig = px / orig_w
        ny_orig = py / orig_h

        vis = draw_point(bgr, nx_orig, ny_orig, score)
        out_path = OUTPUT_DIR / img_path.name
        cv2.imwrite(str(out_path), vis)
        print(f'{img_path.name}: 点=({px:.1f}, {py:.1f}) 置信度={score:.3f}  -> {out_path}')
