import os
import argparse
import numpy as np
import cv2
import torch
from tqdm import tqdm

from train_unet import UNet


def overlay_mask(img, mask_prob, alpha=0.5):
    h, w = img.shape[:2]
    mask_prob = cv2.resize(mask_prob, (w, h))
    mask_color = np.zeros((h, w, 3), dtype=np.uint8)
    mask_color[..., 2] = (mask_prob * 255).astype(np.uint8)  # red channel
    overlay = cv2.addWeighted(img, 1.0, mask_color, alpha, 0)
    return overlay


def main():
    parser = argparse.ArgumentParser(description='Infer UNet on an arbitrary image folder and save overlays')
    parser.add_argument('--input', required=True, help='input folder containing images (jpg/png)')
    parser.add_argument('--weights', default=os.path.join(os.path.dirname(__file__), 'runs', 'unet', 'best.pt'))
    parser.add_argument('--img', type=int, default=512, help='resize inference side length')
    parser.add_argument('--out', default=os.path.join(os.path.dirname(__file__), 'overlays', 'external'))
    parser.add_argument('--alpha', type=float, default=0.5, help='overlay alpha for red mask')
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        raise FileNotFoundError(f'input folder not found: {args.input}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=3, out_channels=1).to(device)
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f'weights not found: {args.weights}')
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    os.makedirs(args.out, exist_ok=True)
    files = [f for f in os.listdir(args.input) if f.lower().endswith(('.jpg','.png'))]
    if not files:
        print('[WARN] No images found in input folder.')

    for name in tqdm(files, desc='Infer'):
        path = os.path.join(args.input, name)
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            print(f'[SKIP] failed to read: {name}')
            continue
        img_resized = cv2.resize(img, (args.img, args.img))
        inp = img_resized.astype(np.float32)/255.0
        inp = torch.from_numpy(inp).permute(2,0,1).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(inp)
            prob = torch.sigmoid(pred)[0,0].cpu().numpy()
        overlay = overlay_mask(img, prob, alpha=args.alpha)
        out_path = os.path.join(args.out, os.path.splitext(name)[0] + '_overlay.png')
        _, buf = cv2.imencode('.png', overlay)
        buf.tofile(out_path)

    print(f'[DONE] Overlays saved to: {args.out}')


if __name__ == '__main__':
    main()
