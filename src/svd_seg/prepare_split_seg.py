import os
import random
import argparse
import numpy as np
import cv2
from typing import List

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(folder: str) -> List[str]:
    files = []
    for root, _, names in os.walk(folder):
        for n in names:
            if os.path.splitext(n)[1].lower() in IMG_EXTS:
                files.append(os.path.join(root, n))
    return files


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def to_three_channel(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return np.stack([img, img, img], axis=2)
    if img.ndim == 3 and img.shape[2] == 1:
        return np.concatenate([img, img, img], axis=2)
    return img


def save_image(out_path: str, img3: np.ndarray):
    ok, buf = cv2.imencode(".jpg", img3)
    if not ok:
        raise RuntimeError(f"encode fail: {out_path}")
    buf.tofile(out_path)


def main():
    parser = argparse.ArgumentParser(description="Split images into segmentation dataset structure")
    parser.add_argument("--src", required=True, help="Source Trainset folder (SVD images)")
    parser.add_argument("--out", required=True, help="Output dataset root")
    parser.add_argument("--train", type=float, default=0.8, help="Train ratio (0-1)")
    args = parser.parse_args()

    src = args.src
    out = args.out
    train_ratio = max(0.0, min(1.0, args.train))

    images_train = os.path.join(out, "images", "train")
    images_val = os.path.join(out, "images", "val")
    masks_train = os.path.join(out, "masks", "train")
    masks_val = os.path.join(out, "masks", "val")

    for d in [images_train, images_val, masks_train, masks_val]:
        ensure_dir(d)

    files = list_images(src)
    if not files:
        print("[ERROR] No images found in src")
        return

    random.seed(42)
    random.shuffle(files)

    split_idx = int(len(files) * train_ratio)
    train_files = files[:split_idx]
    val_files = files[split_idx:]

    def copy(paths: List[str], out_images: str):
        for p in paths:
            data = np.fromfile(p, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"[SKIP] read fail: {p}")
                continue
            img3 = to_three_channel(img)
            base = os.path.splitext(os.path.basename(p))[0]
            out_path = os.path.join(out_images, base + ".jpg")
            try:
                save_image(out_path, img3)
            except Exception as e:
                print(f"[ERROR] {e}")

    copy(train_files, images_train)
    copy(val_files, images_val)

    print(f"[DONE] Train: {len(train_files)}, Val: {len(val_files)} -> {out}")


if __name__ == "__main__":
    main()
