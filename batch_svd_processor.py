import os
import sys
import math
import argparse
from typing import Tuple, Optional, List
from datetime import datetime
import json

import numpy as np
import cv2


# --- Utilities for robust image IO (Chinese path safe) ---
def imread_gray(path: str) -> Optional[np.ndarray]:
    if not os.path.exists(path):
        return None
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    return img


def imwrite_safe(path: str, img: np.ndarray, ext: Optional[str] = None) -> bool:
    try:
        if ext is None:
            ext = os.path.splitext(path)[1].lower() or ".png"
        encode_ext = ext
        if encode_ext not in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
            encode_ext = ".png"
        ok, buf = cv2.imencode(encode_ext, img)
        if not ok:
            return False
        buf.tofile(path)
        return True
    except Exception:
        return False


# --- Extracted: Hough rectify (from GUI) ---
def hough_rectify_grid(input_data: np.ndarray, canny_low=50, canny_high=150, hough_thresh_factor=0.13, blur_margin=100):
    if input_data is None:
        return None, 0.0, None

    src = input_data
    h, w = src.shape[:2]

    # Fade-to-black border to suppress edge artifacts for SVD
    if blur_margin > 0:
        hm, wm = src.shape[:2]
        margin = min(blur_margin, hm // 2, wm // 2)
        if margin > 0:
            mask = np.ones((hm, wm), dtype=np.float32)
            grad = 0.5 * (1 - np.cos(np.linspace(0, np.pi, margin, dtype=np.float32)))
            mask[0:margin, :] *= grad[:, np.newaxis]
            mask[hm - margin:hm, :] *= grad[::-1, np.newaxis]
            mask[:, 0:margin] *= grad[np.newaxis, :]
            mask[:, wm - margin:wm] *= grad[np.newaxis, ::-1]
            src = (src.astype(np.float32) * mask).astype(np.uint8)

    src_for_detect = src.copy()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_enhanced = clahe.apply(src_for_detect)
    blurred = cv2.GaussianBlur(img_enhanced, (5, 5), 0)

    edges = cv2.Canny(blurred, canny_low, canny_high, apertureSize=3)
    threshold = int(min(w, h) * hough_thresh_factor)
    if threshold <= 0:
        threshold = 1
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold)
    if lines is None:
        return src, 0.0, None

    angles = []
    for line in lines:
        rho, theta = line[0]
        angle = math.degrees(theta)
        norm_angle = angle % 90
        if norm_angle > 45:
            norm_angle -= 90
        angles.append(norm_angle)
    if not angles:
        return src, 0.0, None

    detected_angle = sorted(angles)[len(angles) // 2]

    (h, w) = src.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, detected_angle, 1.0)
    abs_cos = abs(M[0, 0])
    abs_sin = abs(M[0, 1])
    bound_w = int(h * abs_sin + w * abs_cos)
    bound_h = int(h * abs_cos + w * abs_sin)
    M[0, 2] += bound_w / 2 - center[0]
    M[1, 2] += bound_h / 2 - center[1]
    result = cv2.warpAffine(src, M, (bound_w, bound_h), flags=cv2.INTER_LANCZOS4)
    return result, float(detected_angle), M


# --- Extracted: SVD reconstruction ---
def perform_svd_reconstruction(gray_matrix: np.ndarray, k: int = 10) -> Optional[np.ndarray]:
    if gray_matrix is None:
        return None
    U, S, Vt = np.linalg.svd(gray_matrix, full_matrices=False)
    S_new = S.copy()
    if k < len(S_new):
        S_new[:k] = 0
    else:
        S_new[:] = 0
    reconstructed = np.dot(U * S_new, Vt)
    return reconstructed


# --- Helpers extracted from GUI class ---
def get_kernel(kernel_type: str, size: int):
    if size < 1:
        size = 1
    if size % 2 == 0:
        size += 1
    kmap = {
        "RECT": cv2.MORPH_RECT,
        "ELLIPSE": cv2.MORPH_ELLIPSE,
        "CROSS": cv2.MORPH_CROSS,
    }
    return cv2.getStructuringElement(kmap.get(kernel_type, cv2.MORPH_RECT), (size, size))


def apply_mask_blur_suppress(
    img_gray: np.ndarray,
    threshold: int = 240,
    threshold_type: str = "BINARY",
    open_kernel_type: str = "RECT",
    open_kernel_size: int = 61,
    dilate_kernel_type: str = "RECT",
    dilate_kernel_size: int = 101,
    dilate_iterations: int = 1,
    blur_kernel: int = 101,
    edge_smooth: int = 51,
) -> np.ndarray:
    if img_gray is None:
        return None
    src = img_gray
    th_type = cv2.THRESH_BINARY if threshold_type == "BINARY" else cv2.THRESH_BINARY_INV
    _, binary = cv2.threshold(src, int(threshold), 255, th_type)
    open_kernel = get_kernel(open_kernel_type, int(open_kernel_size))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel)
    dilate_kernel = get_kernel(dilate_kernel_type, int(dilate_kernel_size))
    dilated = cv2.dilate(opened, dilate_kernel, iterations=max(1, int(dilate_iterations)))
    if blur_kernel % 2 == 0:
        blur_kernel += 1
    blurred = cv2.GaussianBlur(src.astype(np.float32), (int(blur_kernel), int(blur_kernel)), 0)
    if edge_smooth % 2 == 0:
        edge_smooth += 1
    mask_f = (dilated.astype(np.float32) / 255.0)
    smooth = cv2.GaussianBlur(mask_f, (int(edge_smooth), int(edge_smooth)), 0)
    smooth = cv2.GaussianBlur(smooth, (int(edge_smooth), int(edge_smooth)), 0)
    smooth = np.power(np.clip(smooth, 0.0, 1.0), 0.7)
    src_f = src.astype(np.float32)
    result = src_f * (1.0 - smooth) + blurred * smooth
    return np.clip(result, 0, 255).astype(np.uint8)


def estimate_background_downsample(img: np.ndarray, scale: float = 0.25, method: str = "gaussian", kernel: int = 41) -> np.ndarray:
    if img is None:
        return None
    if img.ndim != 2:
        raise ValueError("下采样背景估计仅支持灰度图")
    scale = min(max(scale, 0.01), 0.99)
    h, w = img.shape[:2]
    small_w = max(1, int(w * scale))
    small_h = max(1, int(h * scale))
    small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)
    if kernel % 2 == 0:
        kernel += 1
    method = method.lower()
    if method == "gaussian":
        blurred = cv2.GaussianBlur(small, (kernel, kernel), 0)
    elif method == "median":
        blurred = cv2.medianBlur(small, kernel)
    elif method == "opening":
        k_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
        blurred = cv2.morphologyEx(small, cv2.MORPH_OPEN, k_elem)
    else:
        raise ValueError("method 必须是 gaussian/median/opening 之一")
    background = cv2.resize(blurred, (w, h), interpolation=cv2.INTER_CUBIC)
    return background


def normalize_divide(img: np.ndarray, background: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    img_f = img.astype(np.float32)
    bg_f = background.astype(np.float32)
    mean_val = float(np.mean(img_f))
    result = img_f / (bg_f + eps) * mean_val
    return np.clip(result, 0, 255).astype(np.uint8)


def normalize_subtract(img: np.ndarray, background: np.ndarray, offset: float = 10.0) -> np.ndarray:
    img_f = img.astype(np.float32)
    bg_f = background.astype(np.float32)
    result = img_f - bg_f + float(offset)
    return np.clip(result, 0, 255).astype(np.uint8)


def preprocess_step1(
    img_gray: np.ndarray,
    # Pre-Opening
    pre_open_kernel_type: str = "RECT",
    pre_open_kernel_size: int = 5,
    pre_open_iterations: int = 1,
    # Mask-Blur-Suppress defaults
    mbs_threshold: int = 240,
    mbs_threshold_type: str = "BINARY",
    mbs_open_kernel_type: str = "RECT",
    mbs_open_kernel_size: int = 61,
    mbs_dilate_kernel_type: str = "RECT",
    mbs_dilate_kernel_size: int = 101,
    mbs_dilate_iterations: int = 1,
    mbs_blur_kernel: int = 101,
    mbs_edge_smooth: int = 51,
    # DownsampleBG params
    ds_scale: float = 0.25,
    ds_method: str = "gaussian",
    ds_kernel: int = 41,
    ds_offset: float = 10.0,
    ds_norm_mode: str = "divide",
    ds_invert: bool = False,
) -> np.ndarray:
    if img_gray is None:
        return None
    # Pre-opening
    ksize = int(pre_open_kernel_size)
    if ksize % 2 == 0:
        ksize += 1
    pre_kernel = get_kernel(pre_open_kernel_type, ksize)
    pre_iter = max(1, int(pre_open_iterations))
    pre_open_out = cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, pre_kernel, iterations=pre_iter)
    # Mask-blur-suppress
    mbs_out = apply_mask_blur_suppress(
        pre_open_out,
        threshold=int(mbs_threshold),
        threshold_type=mbs_threshold_type,
        open_kernel_type=mbs_open_kernel_type,
        open_kernel_size=int(mbs_open_kernel_size),
        dilate_kernel_type=mbs_dilate_kernel_type,
        dilate_kernel_size=int(mbs_dilate_kernel_size),
        dilate_iterations=int(mbs_dilate_iterations),
        blur_kernel=int(mbs_blur_kernel),
        edge_smooth=int(mbs_edge_smooth),
    )
    # Downsample background normalization
    bg = estimate_background_downsample(
        mbs_out,
        scale=float(ds_scale),
        method=ds_method,
        kernel=int(ds_kernel),
    )
    if ds_norm_mode == "subtract":
        norm = normalize_subtract(mbs_out, bg, offset=float(ds_offset))
    else:
        norm = normalize_divide(mbs_out, bg)
    if ds_invert:
        norm = 255 - norm
    return norm


# --- Classification (lightweight, consistent defaults) ---
# 分类逻辑已移除：仅保存 SVD 结果并按每次运行分组


def process_image_to_svd(
    img_gray: np.ndarray,
    svd_k: int = 10,
    fast: bool = False,
    cfg: Optional[dict] = None,
) -> Optional[np.ndarray]:
    # Step 1: preprocess using DownsampleBG pipeline; fast mode uses lighter kernels
    if fast:
        pre_kwargs = dict(
            pre_open_kernel_type=(cfg.get("pre_open_kernel_type") if cfg else "RECT"),
            pre_open_kernel_size=3,
            pre_open_iterations=(int(cfg.get("pre_open_iterations", 1)) if cfg else 1),
            mbs_threshold=int(cfg.get("mbs_threshold", 240)) if cfg else 240,
            mbs_threshold_type=(cfg.get("mbs_threshold_type", "BINARY") if cfg else "BINARY"),
            mbs_open_kernel_type=(cfg.get("mbs_open_kernel_type", "RECT") if cfg else "RECT"),
            mbs_open_kernel_size=31,
            mbs_dilate_kernel_type=(cfg.get("mbs_dilate_kernel_type", "RECT") if cfg else "RECT"),
            mbs_dilate_kernel_size=61,
            mbs_dilate_iterations=int(cfg.get("mbs_dilate_iterations", 1)) if cfg else 1,
            mbs_blur_kernel=31,
            mbs_edge_smooth=21,
            ds_scale=float(cfg.get("ds_scale", 0.25)) if cfg else 0.25,
            ds_method=(cfg.get("ds_method", "gaussian") if cfg else "gaussian"),
            ds_kernel=31,
            ds_offset=float(cfg.get("ds_offset", 10.0)) if cfg else 10.0,
            ds_norm_mode=(cfg.get("ds_norm_mode", "divide") if cfg else "divide"),
            ds_invert=bool(cfg.get("ds_invert", False)) if cfg else False,
        )
        step1 = preprocess_step1(img_gray, **pre_kwargs)
    else:
        pre_kwargs = dict(
            pre_open_kernel_type=(cfg.get("pre_open_kernel_type") if cfg else "RECT"),
            pre_open_kernel_size=int(cfg.get("pre_open_kernel_size", 5)) if cfg else 5,
            pre_open_iterations=int(cfg.get("pre_open_iterations", 1)) if cfg else 1,
            mbs_threshold=int(cfg.get("mbs_threshold", 240)) if cfg else 240,
            mbs_threshold_type=(cfg.get("mbs_threshold_type", "BINARY") if cfg else "BINARY"),
            mbs_open_kernel_type=(cfg.get("mbs_open_kernel_type", "RECT") if cfg else "RECT"),
            mbs_open_kernel_size=int(cfg.get("mbs_open_kernel_size", 61)) if cfg else 61,
            mbs_dilate_kernel_type=(cfg.get("mbs_dilate_kernel_type", "RECT") if cfg else "RECT"),
            mbs_dilate_kernel_size=int(cfg.get("mbs_dilate_kernel_size", 101)) if cfg else 101,
            mbs_dilate_iterations=int(cfg.get("mbs_dilate_iterations", 1)) if cfg else 1,
            mbs_blur_kernel=int(cfg.get("mbs_blur_kernel", 101)) if cfg else 101,
            mbs_edge_smooth=int(cfg.get("mbs_edge_smooth", 51)) if cfg else 51,
            ds_scale=float(cfg.get("ds_scale", 0.25)) if cfg else 0.25,
            ds_method=(cfg.get("ds_method", "gaussian") if cfg else "gaussian"),
            ds_kernel=int(cfg.get("ds_kernel", 41)) if cfg else 41,
            ds_offset=float(cfg.get("ds_offset", 10.0)) if cfg else 10.0,
            ds_norm_mode=(cfg.get("ds_norm_mode", "divide") if cfg else "divide"),
            ds_invert=bool(cfg.get("ds_invert", False)) if cfg else False,
        )
        step1 = preprocess_step1(img_gray, **pre_kwargs)
    # Step 2: rectify using Hough (use cfg if provided)
    canny_low = int(cfg.get("hr_canny_low", 50)) if cfg else 50
    canny_high = int(cfg.get("hr_canny_high", 150)) if cfg else 150
    hough_factor = float(cfg.get("hr_hough_factor", 0.13)) if cfg else 0.13
    blur_margin = int(cfg.get("hr_blur_margin", 100)) if cfg else 100
    rectified, _, _ = hough_rectify_grid(
        step1 if step1 is not None else img_gray,
        canny_low=canny_low,
        canny_high=canny_high,
        hough_thresh_factor=hough_factor,
        blur_margin=blur_margin,
    )
    if rectified is None:
        return None
    # Step 3: SVD
    inp = rectified
    # No further downsampling for SVD: factor fixed to 1.0
    recon = perform_svd_reconstruction(inp.astype(float), int(svd_k))
    if recon is None:
        return None
    norm = cv2.normalize(recon, None, 0, 255, cv2.NORM_MINMAX)
    return norm.astype(np.uint8)


def list_images(folder: str) -> List[str]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    files = []
    for root, _, names in os.walk(folder):
        for n in names:
            if os.path.splitext(n)[1].lower() in exts:
                files.append(os.path.join(root, n))
    return files


def main():
    parser = argparse.ArgumentParser(description="Batch process images up to SVD using shared GUI defaults.")
    parser.add_argument("input_folder", nargs="?", help="Folder containing images to process")
    parser.add_argument("--output", default="results_svd", help="Output root folder for SVD results")
    parser.add_argument("--svd-k", type=int, default=None, help="K components to remove in SVD (default from GUI config)")
    parser.add_argument("--config", default="crack_detection_config.json", help="GUI config JSON path to load defaults")
    parser.add_argument("--fast", action="store_true", help="Use faster preprocessing kernels for batch runs")
    args = parser.parse_args()

    in_dir = args.input_folder
    if not in_dir:
        # Try Tk file dialog to pick a folder if not provided
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk(); root.withdraw()
            in_dir = filedialog.askdirectory(title="选择待处理图片文件夹")
        except Exception:
            pass

    if not in_dir:
        print("[ERROR] 未指定输入文件夹")
        sys.exit(2)
    if not os.path.isdir(in_dir):
        print(f"[ERROR] 输入路径不是文件夹: {in_dir}")
        sys.exit(2)

    out_root = args.output
    # 每次运行建立分组目录：<输出根>/<输入文件夹名>/<时间戳>
    in_base = os.path.basename(os.path.normpath(in_dir))
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_run = os.path.join(out_root, in_base, run_tag)
    os.makedirs(out_run, exist_ok=True)

    # Load GUI config if present
    cfg = None
    if args.config and os.path.exists(args.config):
        try:
            with open(args.config, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception as e:
            print(f"[WARN] 配置加载失败: {e}")

    # Resolve SVD k from CLI or config
    svd_k_val = args.svd_k if args.svd_k is not None else (int(cfg.get("svd_k", 10)) if cfg else 10)

    files = list_images(in_dir)
    if not files:
        print("[WARN] 输入文件夹未找到图片")
        sys.exit(0)

    print(f"[INFO] 将处理 {len(files)} 张图片，输出至 {out_run}")
    total = 0

    for path in files:
        total += 1
        img = imread_gray(path)
        if img is None:
            print(f"[SKIP] 读取失败: {path}")
            continue
        # 全局统一降采样到 0.25 倍
        h, w = img.shape[:2]
        img = cv2.resize(img, (max(1, int(w * 0.25)), max(1, int(h * 0.25))), interpolation=cv2.INTER_AREA)
        svd_img = process_image_to_svd(img, svd_k=svd_k_val, fast=args.fast, cfg=cfg)
        if svd_img is None:
            print(f"[SKIP] 处理失败: {path}")
            continue

        # 镜像原目录结构保存到当前运行目录
        rel_dir = os.path.relpath(os.path.dirname(path), in_dir)
        rel_dir = "" if rel_dir == "." else rel_dir
        out_dir = os.path.join(out_run, rel_dir)
        os.makedirs(out_dir, exist_ok=True)

        base = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(out_dir, f"{base}_svd.png")
        ok = imwrite_safe(out_path, svd_img, ".png")
        if not ok:
            print(f"[ERROR] 保存失败: {out_path}")
        else:
            print(f"[OK] 保存: {out_path}")

    print(f"[DONE] 完成: 共处理 {total} 张图片；输出根: {out_run}")


if __name__ == "__main__":
    main()
