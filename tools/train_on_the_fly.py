#!/usr/bin/env python3
"""
試卷手寫擦除在線合成訓練腳本 (Train Exam Remover On-the-fly)

特點：
  1. 零靜態硬碟空間佔用：即時邊訓練邊合成，無限多樣性。
  2. 雙重模型支援：
     - 首選：segmentation_models_pytorch (ResNet34-UNet，預訓練骨幹收斂極快)
     - 備援：內建輕量 PyTorch 雙殘差 UNet (具備標準 Kaiming 初始化)
  3. 數值絕對穩定：純精確 float32 計算，徹底杜絕半精度溢出與 NaN。
  4. 自動儲存最佳權重至 weights/best_exam_remover.pth
"""

import os
import sys
import time
import argparse
import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.getcwd())

from dataset_on_the_fly import OnTheFlyExamDataset


# -------------------------------------------------------------
# 備援純 PyTorch UNet 架構 (標準 Kaiming 初始化)
# -------------------------------------------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class FallbackUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv1 = DoubleConv(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv3 = DoubleConv(128, 64)
        
        self.up4 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv4 = DoubleConv(64, 32)
        
        self.outc = nn.Conv2d(32, out_channels, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5)
        x = self.conv1(torch.cat([x, x4], dim=1))
        
        x = self.up2(x)
        x = self.conv2(torch.cat([x, x3], dim=1))
        
        x = self.up3(x)
        x = self.conv3(torch.cat([x, x2], dim=1))
        
        x = self.up4(x)
        x = self.conv4(torch.cat([x, x1], dim=1))
        
        return self.outc(x)


# -------------------------------------------------------------
# 複合損失函數 (BCE + Soft Dice Loss)
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


def build_model(device):
    try:
        import segmentation_models_pytorch as smp
        print("[Model] 偵測到 segmentation_models_pytorch，載入 ResNet34-UNet 預訓練骨幹...")
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation=None
        )
    except Exception as e:
        print(f"[Model] 採用輕量內建 FallbackUNet 架構 (備援模式)...")
        model = FallbackUNet(in_channels=3, out_channels=1)
        
    return model.to(device)


def main():
    parser = argparse.ArgumentParser(description="在線合成試卷手寫去除模型微調")
    parser.add_argument("--epochs", type=int, default=25, help="訓練輪數 (預設 25)")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小 (預設 4)")
    parser.add_argument("--lr", type=float, default=3e-4, help="學習率 (預設 3e-4)")
    parser.add_argument("--steps_per_epoch", type=int, default=200, help="每輪訓練樣本數 (預設 200)")
    parser.add_argument("--val_steps", type=int, default=40, help="每輪驗證樣本數 (預設 40)")
    parser.add_argument("--dpi", type=int, default=300, help="考卷合成 DPI (原生高解析 300)")
    parser.add_argument("--patch_size", type=int, default=1024, help="1:1 原生像素 Patch 邊長 (預設 1024)")
    parser.add_argument("--workers", type=int, default=2, help="資料加載執行緒 (預設 2)")
    parser.add_argument("--use_amp", action="store_true", help="是否啟用 AMP 半精度加速 (建議在預訓練模型使用)")
    parser.add_argument("--save_path", type=str, default="weights/best_exam_remover.pth", help="模型權重儲存路徑")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== 考卷手寫消除在線訓練啟動 ===")
    print(f"運算裝置: {device}")
    print(f"原生 DPI: {args.dpi} | Patch 尺寸: {args.patch_size}x{args.patch_size} | 批次大小: {args.batch_size} | 輪數: {args.epochs}")

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    # 1. 在線資料集
    patch_size = (args.patch_size, args.patch_size)
    train_dataset = OnTheFlyExamDataset(epoch_size=args.steps_per_epoch, dpi=args.dpi, patch_size=patch_size, is_train=True)
    val_dataset = OnTheFlyExamDataset(epoch_size=args.val_steps, dpi=args.dpi, patch_size=patch_size, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=(device.type == "cuda"))

    # 2. 模型與優化器
    model = build_model(device)
    criterion = DiceBCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    use_amp = args.use_amp and (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_loss = float("inf")

    # 3. 訓練迴圈
    for epoch in range(1, args.epochs + 1):
        epoch_t0 = time.time()
        model.train()
        train_loss = 0.0
        train_batches = 0

        for imgs, masks, _ in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad()
            if use_amp:
                with torch.amp.autocast("cuda"):
                    preds = model(imgs)
                    loss = criterion(preds, masks)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                preds = model(imgs)
                loss = criterion(preds, masks)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            train_batches += imgs.size(0)

        scheduler.step()
        train_loss /= max(1, train_batches)

        # 驗證
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for imgs, masks, _ in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                if use_amp:
                    with torch.amp.autocast("cuda"):
                        preds = model(imgs)
                        loss = criterion(preds, masks)
                else:
                    preds = model(imgs)
                    loss = criterion(preds, masks)
                val_loss += loss.item() * imgs.size(0)
                val_batches += imgs.size(0)

        val_loss /= max(1, val_batches)
        elapsed = time.time() - epoch_t0

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({elapsed:.1f}s) - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.save_path)
            print(f"  ★ 儲存最佳模型 (Val Loss: {best_val_loss:.4f}) -> {args.save_path}")

    print(f"\n🎉 訓練全數完成！最佳權重已儲存於: {args.save_path}")


if __name__ == "__main__":
    main()
