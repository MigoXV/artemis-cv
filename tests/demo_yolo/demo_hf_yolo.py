from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import AutoModel

ROOT = Path(__file__).resolve().parents[2]
HF_MODEL_DIR = ROOT / 'model-bin' / 'artimes-yolov8n-260323-1629'
SCREENSHOTS_DIR = ROOT / 'data-bin' / 'screenshots'
OUTPUT_DIR = ROOT / 'data-bin' / 'screenshot-outputs'
IMGSZ = 640
DEVICE = torch.device('cpu')


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


def draw_point(image, nx, ny, score):
    h, w = image.shape[:2]
    cx, cy = int(nx * w), int(ny * h)
    vis = image.copy()
    cv2.circle(vis, (cx, cy), 6, (0, 0, 255), -1)
    cv2.circle(vis, (cx, cy), 8, (255, 255, 255), 2)
    cv2.putText(vis, f'{score:.3f}', (cx + 10, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return vis


if __name__ == '__main__':
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not HF_MODEL_DIR.exists():
        raise FileNotFoundError(f"HF model directory not found: {HF_MODEL_DIR}. Run conversion script first.")

    model = AutoModel.from_pretrained(str(HF_MODEL_DIR), trust_remote_code=True)
    model.to(DEVICE)
    model.eval()

    image_paths = sorted(SCREENSHOTS_DIR.glob('*.png'))
    if not image_paths:
        raise FileNotFoundError(f'No PNG images found in {SCREENSHOTS_DIR}')

    for img_path in image_paths:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f'Warning: failed to read {img_path}, skipping')
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        canvas, scale, pad_w, pad_h, orig_w, orig_h = letterbox(rgb, IMGSZ)

        tensor = torch.from_numpy(np.transpose(canvas.astype(np.float32) / 255.0, (2, 0, 1))).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(pixel_values=tensor)
        pred_points, pred_scores = decode_single_point(outputs)

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
        print(f'{img_path.name}: point=({px:.1f}, {py:.1f}) score={score:.3f} -> {out_path}')
