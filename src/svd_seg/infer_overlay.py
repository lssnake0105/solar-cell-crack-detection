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
    parser = argparse.ArgumentParser(description='Infer UNet and save overlays')
    parser.add_argument('--data', required=True, help='dataset root')
    parser.add_argument('--weights', default=os.path.join(os.path.dirname(__file__), 'runs', 'unet', 'best.pt'))
    parser.add_argument('--img', type=int, default=512)
    parser.add_argument('--subset', default='val', choices=['train','val'])
    parser.add_argument('--out', default=os.path.join(os.path.dirname(__file__), 'overlays'))
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=3, out_channels=1).to(device)
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f'weights not found: {args.weights}')
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    img_dir = os.path.join(args.data, 'images', args.subset)
    out_dir = os.path.join(args.out, args.subset)
    os.makedirs(out_dir, exist_ok=True)
    files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg','.png'))]

    for name in tqdm(files, desc='Infer'):
        path = os.path.join(img_dir, name)
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        img_resized = cv2.resize(img, (args.img, args.img))
        inp = img_resized.astype(np.float32)/255.0
        inp = torch.from_numpy(inp).permute(2,0,1).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(inp)
            prob = torch.sigmoid(pred)[0,0].cpu().numpy()
        overlay = overlay_mask(img, prob, alpha=0.5)
        out_path = os.path.join(out_dir, os.path.splitext(name)[0] + '_overlay.png')
        _, buf = cv2.imencode('.png', overlay)
        buf.tofile(out_path)

if __name__ == '__main__':
    main()
