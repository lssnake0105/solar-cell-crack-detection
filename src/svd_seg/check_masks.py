import os
import argparse
import cv2
import numpy as np
from pathlib import Path


def imread_gray(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def list_images(img_dir):
    return [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg','.png','.jpeg'))]


def check_subset(root, subset):
    img_dir = os.path.join(root, 'images', subset)
    mask_dir = os.path.join(root, 'masks', subset)
    files = list_images(img_dir)
    total = len(files)
    mask_exists = 0
    non_empty = 0
    per_file = []
    for name in files:
        img_path = os.path.join(img_dir, name)
        mask_path = os.path.join(mask_dir, os.path.splitext(name)[0] + '.png')
        exists = os.path.exists(mask_path)
        nz = 0
        if exists:
            m = imread_gray(mask_path)
            if m is not None:
                nz = int((m > 127).sum())
                if nz > 0:
                    non_empty += 1
                mask_exists += 1
            else:
                exists = False
        per_file.append((subset, name, bool(exists), nz))
    return {
        'subset': subset,
        'total_images': total,
        'mask_exists': mask_exists,
        'non_empty_masks': non_empty,
        'per_file': per_file,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='dataset root (svd_seg/dataset)')
    ap.add_argument('--out', default=None, help='optional CSV to write per-file stats')
    args = ap.parse_args()

    stats_train = check_subset(args.root, 'train')
    stats_val = check_subset(args.root, 'val')

    def print_stats(s):
        print(f"[{s['subset']}] images={s['total_images']} masks={s['mask_exists']} non_empty={s['non_empty_masks']}")
    print_stats(stats_train)
    print_stats(stats_val)

    # Show a few sample lines
    print('Samples (subset,name,mask_exists,nonzero_pixels):')
    for row in (stats_train['per_file'][:10] + stats_val['per_file'][:10]):
        print(row)

    if args.out:
        import csv
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['subset','name','mask_exists','nonzero_pixels'])
            for s in [stats_train, stats_val]:
                for row in s['per_file']:
                    w.writerow(row)
        print('Wrote CSV:', args.out)

if __name__ == '__main__':
    main()
