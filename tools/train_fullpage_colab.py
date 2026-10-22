#!/usr/bin/env python3
"""
Colab 全頁考卷手寫分離模型訓練腳本 (Full-Page Exam Remover Training on Colab)

核心特性：
  1. 100% 全頁直出，絕無縮放拉伸：
     採用動態 Batch-Level Padding 邊緣對齊（補白邊至 32 倍數），保留真實長寬比。
  2. 預訓練骨幹加速收斂：ResNet34-UNet (segmentation_models_pytorch)。
  3. 數值穩定複合損失：BCE + Soft Dice Loss (強制 float32 累加)。
  4. 支援 AMP 混合精度與梯度裁剪，完美適配 Colab T4 (16GB)。
"""

import os
import sys
import glob
import time
import argparse
import random
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -------------------------------------------------------------
# 1. 全頁考卷資料集 (零縮放 + 動態批次補白)
# -------------------------------------------------------------
class FullPageExamDataset(Dataset):
    def __init__(self, data_root="dataset_fullpage", is_train=True, split_ratio=0.9):
        self.img_paths = sorted(glob.glob(os.path.join(data_root, "images", "*.jpg")))
        self.mask_paths = sorted(glob.glob(os.path.join(data_root, "masks", "*.png")))
        self.tgt_paths = sorted(glob.glob(os.path.join(data_root, "targets", "*.jpg")))

        assert len(self.img_paths) > 0, f"未在 {data_root} 找到任何訓練圖片！"
        assert len(self.img_paths) == len(self.mask_paths), "影像與遮罩數量不一致！"

        # 固定種子劃分訓練/驗證集
        total = len(self.img_paths)
        indices = list(range(total))
        random.seed(42)
        random.shuffle(indices)

        split_idx = int(total * split_ratio)
        if is_train:
            selected_idx = indices[:split_idx]
        else:
            selected_idx = indices[split_idx:]

        self.img_paths = [self.img_paths[i] for i in selected_idx]
        self.mask_paths = [self.mask_paths[i] for i in selected_idx]
        self.tgt_paths = [self.tgt_paths[i] for i in selected_idx]
        self.is_train = is_train

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # 讀取原生圖片 (絕不 resize)
        im_bgr = cv2.imread(self.img_paths[idx])
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)

        # 輕微隨機色彩/亮度增強
        if self.is_train and random.random() < 0.4:
            alpha = random.uniform(0.95, 1.05)
            beta = random.uniform(-6, 6)
            im_bgr = np.clip(im_bgr.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        norm_img = (im_rgb - self.mean) / self.std

        # 轉 Tensor: (C, H, W)
        tensor_img = torch.from_numpy(norm_img).permute(2, 0, 1).float()
        tensor_mask = torch.from_numpy((mask > 128).astype(np.float32)).unsqueeze(0)

        return tensor_img, tensor_mask


def pad_to_multiple(x, pad_val, target_h, target_w):
    """將單張 Tensor 補白至目標長寬"""
    _, h, w = x.shape
    dh = target_h - h
    dw = target_w - w
    if dh > 0 or dw > 0:
        return F.pad(x, (0, dw, 0, dh), mode="constant", value=pad_val)
    return x


def fullpage_collate_fn(batch):
    """
    動態 Batch 補齊：
    找出當前 Batch 內圖片的最大長寬，並向上取整到 32 的倍數，
    確保零拉伸、零縮放，同時能順暢組合成 (B, C, H, W) Tensor！
    """
    max_h = max(item[0].shape[1] for item in batch)
    max_w = max(item[0].shape[2] for item in batch)

    pad_h = ((max_h + 31) // 32) * 32
    pad_w = ((max_w + 31) // 32) * 32

    batch_imgs = []
    batch_masks = []
    for img, mask in batch:
        # 輸入圖補 ImageNet 標準化後的背景白
        batch_imgs.append(pad_to_multiple(img, 0.0, pad_h, pad_w))
        # 遮罩補 0 (背景非手寫)
        batch_masks.append(pad_to_multiple(mask, 0.0, pad_h, pad_w))

    return torch.stack(batch_imgs, dim=0), torch.stack(batch_masks, dim=0)


# -------------------------------------------------------------
# 2. 複合損失函數 (數值穩定版 BCE + Soft Dice)
# -------------------------------------------------------------
class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        logits = logits.float()
        targets = targets.float()

        bce_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        dice_loss = 1.0 - dice

        return 0.5 * bce_loss + 0.5 * dice_loss


# -------------------------------------------------------------
# 3. 訓練主流程
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Colab 全頁考卷手寫分離模型訓練")
    parser.add_argument("--data_dir", type=str, default="dataset_fullpage", help="全頁資料集路徑")
    parser.add_argument("--epochs", type=int, default=20, help="訓練輪數 (預設 20)")
    parser.add_argument("--batch_size", type=int, default=2, help="批次大小 (預設 2, 適配 T4 16GB)")
    parser.add_argument("--lr", type=float, default=3e-4, help="學習率 (預設 3e-4)")
    parser.add_argument("--workers", type=int, default=2, help="資料載入線程")
    parser.add_argument("--save_path", type=str, default="weights/best_exam_remover.pth", help="模型儲存路徑")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Colab 全頁考卷手寫分離訓練啟動 ===")
    print(f"運算裝置: {device} | 批次大小: {args.batch_size} | 訓練輪數: {args.epochs}")

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    # 1. 載入資料集
    train_ds = FullPageExamDataset(data_root=args.data_dir, is_train=True)
    val_ds = FullPageExamDataset(data_root=args.data_dir, is_train=False)
    print(f"資料集規模: 訓練集 {len(train_ds)} 頁，驗證集 {len(val_ds)} 頁")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, collate_fn=fullpage_collate_fn,
        pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, collate_fn=fullpage_collate_fn,
        pin_memory=(device.type == "cuda")
    )

    # 2. 構建 ResNet34-UNet
    import segmentation_models_pytorch as smp
    print("[Model] 載入 ResNet34-UNet (預訓練 ImageNet 編碼器)...")
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None
    ).to(device)

    criterion = DiceBCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_loss = float("inf")

    # 3. 訓練迴圈
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        train_count = 0

        for imgs, masks in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                preds = model(imgs)
                loss = criterion(preds, masks)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * imgs.size(0)
            train_count += imgs.size(0)

        scheduler.step()
        train_loss /= max(1, train_count)

        # 驗證
        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    preds = model(imgs)
                    loss = criterion(preds, masks)
                val_loss += loss.item() * imgs.size(0)
                val_count += imgs.size(0)

        val_loss /= max(1, val_count)
        elapsed = time.time() - t0

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({elapsed:.1f}s) - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.save_path)
            print(f"  ★ 儲存最佳模型 (Val Loss: {best_val_loss:.4f}) -> {args.save_path}")

    print(f"\n🎉 訓練全數完成！最佳權重已儲存於: {args.save_path}")


if __name__ == "__main__":
    main()
