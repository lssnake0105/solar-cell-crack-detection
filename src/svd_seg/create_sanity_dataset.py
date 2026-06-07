import os
import argparse
import numpy as np
import cv2
from pathlib import Path


def imwrite(path, img):
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.jpg', '.jpeg']:
        _, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    else:
        _, buf = cv2.imencode('.png', img)
    buf.tofile(path)


def make_sample(h=512, w=512):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = 30 + np.random.randint(0, 20)
    # add faint texture
    noise = (np.random.randn(h, w)*3).astype(np.float32)
    for c in range(3):
        img[..., c] = np.clip(img[..., c].astype(np.float32) + noise, 0, 255).astype(np.uint8)
    # draw a few thin lines as "cracks"
    mask = np.zeros((h, w), dtype=np.uint8)
    n_lines = np.random.randint(1, 4)
    for _ in range(n_lines):
        x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
        x2, y2 = np.random.randint(0, w), np.random.randint(0, h)
        thickness = np.random.randint(1, 3)
        cv2.line(mask, (x1, y1), (x2, y2), 255, thickness)
        # put weak signal on image
        cv2.line(img, (x1, y1), (x2, y2), (200, 80, 80), max(1, thickness-1))
    return img, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True, help='dataset root, will create images/masks train|val')
    ap.add_argument('--n', type=int, default=12, help='total samples to generate')
    ap.add_argument('--imgsz', type=int, default=512)
    args = ap.parse_args()

    root = Path(args.out)
    for sub in ['images/train', 'images/val', 'masks/train', 'masks/val']:
        (root / sub).mkdir(parents=True, exist_ok=True)

    n_train = max(1, int(args.n*0.8))
    n_val = max(1, args.n - n_train)

    # train
    for i in range(n_train):
        img, mask = make_sample(args.imgsz, args.imgsz)
        name = f'sanity_{i:03d}.png'
        imwrite(str(root / 'images/train' / name), img)
        imwrite(str(root / 'masks/train' / name), mask)
    # val
    for i in range(n_val):
        img, mask = make_sample(args.imgsz, args.imgsz)
        name = f'sanity_val_{i:03d}.png'
        imwrite(str(root / 'images/val' / name), img)
        imwrite(str(root / 'masks/val' / name), mask)

    print(f'Created sanity dataset at {root} with train={n_train}, val={n_val}')

if __name__ == '__main__':
    main()
