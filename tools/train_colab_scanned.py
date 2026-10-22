#!/usr/bin/env python3
"""
真實掃描試卷在線合成訓練腳本 (Real-Scanned Exam Training on Colab)
使用 106 頁真實列印掃描試卷作為底圖，徹底杜絕 Domain Shift！
"""

import os
import sys
import time
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset_scanned_onthefly import ScannedFullPageDataset, scanned_collate_fn


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


def main():
    parser = argparse.ArgumentParser(description="真實掃描試卷在線合成訓練")
    parser.add_argument("--epochs", type=int, default=15, help="訓練輪數 (預設 15)")
    parser.add_argument("--steps_per_epoch", type=int, default=100, help="每輪步數 (15x100 = 1500 頁)")
    parser.add_argument("--val_steps", type=int, default=20, help="每輪驗證步數")
    parser.add_argument("--batch_size", type=int, default=2, help="批次大小")
    parser.add_argument("--lr", type=float, default=3e-4, help="學習率")
    parser.add_argument("--workers", type=int, default=2, help="資料載入線程")
    parser.add_argument("--save_path", type=str, default="weights/best_exam_remover.pth", help="模型儲存路徑")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== 真實掃描試卷在線合成深度訓練啟動 ===")
    print(f"運算裝置: {device} | 批次大小: {args.batch_size} | 總訓練步數: {args.epochs * args.steps_per_epoch}")

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    # 1. 建立真實掃描資料集
    train_ds = ScannedFullPageDataset(
        scanned_root="scanned_pairs",
        assets_root="assets",
        catalog_path="assets/scanned_catalog.json",
        total_steps=args.steps_per_epoch * args.batch_size,
        is_train=True
    )
    val_ds = ScannedFullPageDataset(
        scanned_root="scanned_pairs",
        assets_root="assets",
        catalog_path="assets/scanned_catalog.json",
        total_steps=args.val_steps * args.batch_size,
        is_train=False
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, collate_fn=scanned_collate_fn,
        pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, collate_fn=scanned_collate_fn,
        pin_memory=(device.type == "cuda")
    )

    # 2. ResNet34-UNet
    import segmentation_models_pytorch as smp
    print("[Model] 載入 ResNet34-UNet 預訓練骨幹...")
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

    print(f"\n🎉 真實掃描試卷訓練全數完成！最佳權重已儲存於: {args.save_path}")


if __name__ == "__main__":
    main()
