from pathlib import Path

import cv2
from tqdm import tqdm

from artemis_cv.inferencers.yolo import YoloPointInferencer

ROOT = Path(__file__).resolve().parents[2]
HF_MODEL_DIR = ROOT / 'model-bin' / 'artimes-yolov8n-260323-1629'
SCREENSHOTS_DIR = ROOT / 'data-bin' / 'screenshots'
OUTPUT_DIR = ROOT / 'data-bin' / 'screenshot-outputs'


def draw_point(image, px, py, score):
    vis = image.copy()
    cx, cy = int(px), int(py)
    cv2.circle(vis, (cx, cy), 6, (0, 0, 255), -1)
    cv2.circle(vis, (cx, cy), 8, (255, 255, 255), 2)
    cv2.putText(vis, f'{score:.3f}', (cx + 10, cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return vis


if __name__ == '__main__':
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    inferencer = YoloPointInferencer(HF_MODEL_DIR, device='cpu')

    image_paths = sorted(SCREENSHOTS_DIR.glob('*.png'))
    if not image_paths:
        raise FileNotFoundError(f'No PNG images found in {SCREENSHOTS_DIR}')

    for img_path in tqdm(image_paths, desc='Inferencing'):
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            tqdm.write(f'Warning: failed to read {img_path}, skipping')
            continue

        px, py, score = inferencer.infer(bgr)

        vis = draw_point(bgr, px, py, score)
        out_path = OUTPUT_DIR / img_path.name
        cv2.imwrite(str(out_path), vis)
        tqdm.write(f'{img_path.name}: point=({px:.1f}, {py:.1f}) score={score:.3f} -> {out_path}')
