import os
import argparse
import numpy as np
import cv2
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        # Downsampling
        for f in features:
            self.downs.append(self.double_conv(in_channels, f))
            in_channels = f
        self.pool = nn.MaxPool2d(2, 2)
        # Bottleneck
        self.bottleneck = self.double_conv(features[-1], features[-1]*2)
        # Upsampling
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f*2 if f!=features[-1] else features[-1]*2, f, kernel_size=2, stride=2))
            self.ups.append(self.double_conv(f*2, f))
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def double_conv(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        skip = []
        for down in self.downs:
            x = down(x)
            skip.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            s = skip[-(i//2+1)]
            if x.shape[-2:] != s.shape[-2:]:
                x = torch.nn.functional.interpolate(x, size=s.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([s, x], dim=1)
            x = self.ups[i+1](x)
        return self.final_conv(x)

class SegDataset(Dataset):
    def __init__(self, root, subset='train', size=512, crop=0, augment=False):
        self.img_dir = os.path.join(root, 'images', subset)
        self.mask_dir = os.path.join(root, 'masks', subset)
        self.size = size
        self.crop = int(crop) if crop else 0
        self.augment = augment
        self.files = [f for f in os.listdir(self.img_dir) if f.lower().endswith(('.jpg','.png'))]
        # 预计算是否为缺陷样本（用于了解分布）
        self.is_defect = []
        for f in self.files:
            mp = os.path.join(self.mask_dir, os.path.splitext(f)[0] + '.png')
            m = None
            if os.path.exists(mp):
                data = np.fromfile(mp, dtype=np.uint8)
                if len(data) > 0:
                    m = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
            self.is_defect.append(bool(m is not None and np.any(m > 127)))

    def __len__(self):
        return len(self.files)

    def _random_crop(self, img: np.ndarray, mask: np.ndarray, size: int) -> Tuple[np.ndarray, np.ndarray]:
        h, w = img.shape[:2]
        if h < size or w < size:
            pad_h = max(0, size - h)
            pad_w = max(0, size - w)
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
            mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
            h, w = img.shape[:2]
        y = np.random.randint(0, h - size + 1)
        x = np.random.randint(0, w - size + 1)
        return img[y:y+size, x:x+size], mask[y:y+size, x:x+size]

    def _augment(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # 随机水平/垂直翻转
        if np.random.rand() < 0.5:
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)
        if np.random.rand() < 0.3:
            img = cv2.flip(img, 0)
            mask = cv2.flip(mask, 0)
        # 随机 90 度旋转
        k = np.random.choice([0,1,2,3])
        if k:
            img = np.rot90(img, k).copy()
            mask = np.rot90(mask, k).copy()
        # 小角度旋转 + 尺度扰动（保持掩膜对齐）
        if np.random.rand() < 0.5:
            angle = np.random.uniform(-12, 12)
            scale = np.random.uniform(0.9, 1.1)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        # 亮度/对比度微调
        if np.random.rand() < 0.5:
            alpha = 0.9 + np.random.rand()*0.2  # [0.9,1.1]
            beta = np.random.randint(-10, 10)
            img = np.clip(img.astype(np.float32)*alpha + beta, 0, 255).astype(np.uint8)
        # 轻度高斯噪声
        if np.random.rand() < 0.3:
            noise = np.random.randn(*img.shape) * 3.0
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        # 轻度模糊
        if np.random.rand() < 0.3:
            ksize = np.random.choice([0,3,5])
            if ksize:
                img = cv2.GaussianBlur(img, (ksize, ksize), 0)
        return img, mask

    def __getitem__(self, idx):
        name = self.files[idx]
        img_path = os.path.join(self.img_dir, name)
        mask_path = os.path.join(self.mask_dir, os.path.splitext(name)[0] + '.png')
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        mask = cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
        # 先缩放到 size，或使用随机裁剪
        if self.crop and self.augment:
            # 对训练集：随机裁剪 + 增强
            img = cv2.resize(img, (max(self.size, self.crop), max(self.size, self.crop)))
            mask = cv2.resize(mask, (max(self.size, self.crop), max(self.size, self.crop)), interpolation=cv2.INTER_NEAREST)
            img, mask = self._random_crop(img, mask, self.crop)
            img, mask = self._augment(img, mask)
            if self.crop != self.size:
                img = cv2.resize(img, (self.size, self.size))
                mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        else:
            # 验证/无增强：直接缩放到 size
            img = cv2.resize(img, (self.size, self.size))
            mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        img = img.astype(np.float32)/255.0
        mask = (mask>127).astype(np.float32)
        img = torch.from_numpy(img).permute(2,0,1)
        mask = torch.from_numpy(mask).unsqueeze(0)
        return img, mask


def dice_loss(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)
    num = 2 * (pred*target).sum() + eps
    den = pred.sum() + target.sum() + eps
    return 1 - num/den

def focal_loss(pred, target, gamma=2.0, alpha=0.25):
    # 二值 focal loss with logits
    bce = nn.functional.binary_cross_entropy_with_logits(pred, target, reduction='none')
    p = torch.sigmoid(pred)
    p_t = p*target + (1-p)*(1-target)
    alpha_t = alpha*target + (1-alpha)*(1-target)
    loss = alpha_t * (1-p_t)**gamma * bce
    return loss.mean()


def main():
    parser = argparse.ArgumentParser(description='Train UNet for SVD crack segmentation (small-data friendly)')
    parser.add_argument('--data', required=True, help='dataset root')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--img', type=int, default=512)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--crop', type=int, default=256, help='random crop size for training (0 to disable)')
    parser.add_argument('--augment', action='store_true', help='enable data augmentation (train only)')
    parser.add_argument('--pos_weight', type=float, default=3.0, help='BCE pos_weight to balance crack pixels')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Focal loss gamma (<=0 to disable)')
    parser.add_argument('--early_patience', type=int, default=10, help='early stopping patience on val loss')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=3, out_channels=1).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    pos_weight = torch.tensor([args.pos_weight], device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_ds = SegDataset(args.data, 'train', size=args.img, crop=args.crop, augment=args.augment)
    val_ds = SegDataset(args.data, 'val', size=args.img, crop=0, augment=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    def mask_stats(ds: SegDataset):
        has_mask = 0
        non_empty = 0
        for f in ds.files:
            mp = os.path.join(ds.mask_dir, os.path.splitext(f)[0] + '.png')
            if not os.path.exists(mp):
                continue
            data = np.fromfile(mp, dtype=np.uint8)
            if len(data) == 0:
                continue
            m = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            has_mask += 1
            if np.any(m > 127):
                non_empty += 1
        return has_mask, non_empty

    tr_mask, tr_nonempty = mask_stats(train_ds)
    va_mask, va_nonempty = mask_stats(val_ds)
    if tr_mask == 0:
        print('[ERROR] No mask files found in train set. Please annotate and save masks first.')
        return
    if tr_nonempty == 0:
        print('[ERROR] All train masks are empty. Please ensure defect samples have non-empty masks.')
        return
    if va_mask == 0:
        print('[WARN] No mask files found in val set; using train stats only.')

    best_val = 1e9
    out_dir = os.path.join(os.path.dirname(__file__), 'runs', 'unet')
    os.makedirs(out_dir, exist_ok=True)
    no_improve = 0
    for epoch in range(1, args.epochs+1):
        model.train()
        tbar = tqdm(train_dl, desc=f'Epoch {epoch}/{args.epochs}')
        total_loss = 0.0
        for img, mask in tbar:
            img, mask = img.to(device), mask.to(device)
            pred = model(img)
            loss = bce(pred, mask) + dice_loss(pred, mask)
            if args.focal_gamma and args.focal_gamma > 0:
                loss = loss + focal_loss(pred, mask, gamma=args.focal_gamma, alpha=0.25)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += float(loss.item())
            tbar.set_postfix({'loss': f'{loss.item():.4f}'})
        # val
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for img, mask in val_dl:
                img, mask = img.to(device), mask.to(device)
                pred = model(img)
                loss = bce(pred, mask) + dice_loss(pred, mask)
                val_loss += float(loss.item())
        avg_val = val_loss / max(1,len(val_dl))
        print(f'[VAL] loss={avg_val:.4f}')
        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), os.path.join(out_dir, 'best.pt'))
            print('[SAVE] best.pt updated')
            no_improve = 0
        else:
            no_improve += 1
            if args.early_patience and no_improve >= args.early_patience:
                print(f'[EARLY STOP] no improvement for {no_improve} epochs, stop.')
                break

    torch.save(model.state_dict(), os.path.join(out_dir, 'last.pt'))
    print('[DONE] Training complete')

if __name__ == '__main__':
    main()
