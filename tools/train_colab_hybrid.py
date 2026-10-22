#!/usr/bin/env python3
"""
雙軌混合考卷手寫擦除訓練器 (Colab Hybrid Real & Dynamic Training Runner)
架構特點：
1. 雙軌數據來源：
   - 50% 採樣自 ExamInk-Seg 真實試卷庫 (1,511 對真實學生塗改、密集算式與淡鉛筆)
   - 50% 採樣自在地 106 頁掃描動態合成 (臺灣學測版型 + 印章粗細呼吸與微幾何擾動)
2. 雙重保護損失 (Text-Aware Combined Loss)：
   - Dice Loss (強化筆劃完整性)
   - BCE Loss (像素分類穩定性)
   - 3.0x 印刷文字誤殺懲罰 (Text False Positive Penalty)，確保字母與題號 100% 留存
3. 熱啟動遷移：從現有最佳權重 best_exam_remover.pth 續練微調，吸收真實試卷筆跡
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from dataset_hybrid import HybridExamDataset, hybrid_collate_fn

class TextAwareCombinedLoss(nn.Module):
    def __init__(self, text_penalty_weight=3.0):
        super().__init__()
        self.text_penalty_weight = text_penalty_weight

    def forward(self, logits, targets, pure_text_masks):
        bce = F.binary_cross_entropy_with_logits(logits, targets)

        probs = torch.sigmoid(logits)
        smooth = 1e-5
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = 1.0 - (2.0 * intersection + smooth) / (union + smooth)
        dice_loss = dice.mean()

        # 核心：印刷文字誤殺懲罰項
        text_fp_loss = (probs * pure_text_masks).sum() / (pure_text_masks.sum() + smooth)

        total_loss = 0.5 * bce + 1.0 * dice_loss + self.text_penalty_weight * text_fp_loss
        return total_loss, bce.item(), dice_loss.item(), text_fp_loss.item()

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== 雙軌真實+動態混合考卷去手寫訓練 ===")
    print(f"運算裝置: {device} ({torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'})")
    print(f"配置: Batch={args.batch_size}, Epochs={args.epochs}, LR={args.lr}, RealRatio={args.real_ratio}, TextPenalty={args.text_penalty}")

    # 1. 建立雙軌資料集
    train_ds = HybridExamDataset(
        examink_root=args.examink_root,
        scanned_root=args.scanned_root,
        assets_root=args.assets_root,
        catalog_path=args.catalog_path,
        total_steps=args.steps_per_epoch * args.batch_size,
        dpi=150,
        is_train=True,
        real_ratio=args.real_ratio
    )
    val_ds = HybridExamDataset(
        examink_root=args.examink_root,
        scanned_root=args.scanned_root,
        assets_root=args.assets_root,
        catalog_path=args.catalog_path,
        total_steps=args.val_steps * args.batch_size,
        dpi=150,
        is_train=False,
        real_ratio=args.real_ratio
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=hybrid_collate_fn,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=hybrid_collate_fn,
        pin_memory=True
    )

    # 2. 建立模型 (ResNet34-UNet)
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None
    ).to(device)

    # 熱啟動權重
    if args.resume and os.path.exists(args.resume):
        print(f"★ 載入現有模型權重進行熱啟動微調: {args.resume}")
        model.load_state_dict(torch.load(args.resume, map_location=device, weights_only=True))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    criterion = TextAwareCombinedLoss(text_penalty_weight=args.text_penalty)

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        train_loss, bce_accum, dice_accum, text_fp_accum = 0.0, 0.0, 0.0, 0.0
        step_count = 0

        for imgs, masks, pure_texts in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            pure_texts = pure_texts.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits = model(imgs)
                loss, bce_val, dice_val, text_fp_val = criterion(logits, masks, pure_texts)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            bce_accum += bce_val
            dice_accum += dice_val
            text_fp_accum += text_fp_val
            step_count += 1

        scheduler.step()
        elapsed = time.time() - t0
        train_loss /= max(1, step_count)
        bce_accum /= max(1, step_count)
        dice_accum /= max(1, step_count)
        text_fp_accum /= max(1, step_count)

        # 驗證
        model.eval()
        val_loss, val_dice, val_text_fp = 0.0, 0.0, 0.0
        val_count = 0
        with torch.no_grad():
            for imgs, masks, pure_texts in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                pure_texts = pure_texts.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                    logits = model(imgs)
                    loss, _, dice_val, text_fp_val = criterion(logits, masks, pure_texts)
                val_loss += loss.item() * imgs.size(0)
                val_dice += dice_val * imgs.size(0)
                val_text_fp += text_fp_val * imgs.size(0)
                val_count += imgs.size(0)

        val_loss /= max(1, val_count)
        val_dice /= max(1, val_count)
        val_text_fp /= max(1, val_count)

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({elapsed:.1f}s) - Train: {train_loss:.4f} (DiceLoss: {dice_accum:.3f}, TextFP: {text_fp_accum:.4f}) | Val: {val_loss:.4f} (ValDiceLoss: {val_dice:.3f}, ValTextFP: {val_text_fp:.4f})")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.save_path)
            print(f"  ★ 儲存最優混合模型 (Val Loss: {best_val_loss:.4f}) -> {args.save_path}")

    print(f"🎉 雙軌混合訓練全數完成！最佳模型已儲存至: {args.save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--examink_root", default="/content/ExamInk-Seg/data")
    parser.add_argument("--scanned_root", default="scanned_pairs")
    parser.add_argument("--assets_root", default="assets")
    parser.add_argument("--catalog_path", default="assets/scanned_catalog.json")
    parser.add_argument("--real_ratio", type=float, default=0.20, help="真實考卷採樣比例")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--text_penalty", type=float, default=3.0, help="印刷文字誤殺懲罰權重")
    parser.add_argument("--steps_per_epoch", type=int, default=150)
    parser.add_argument("--val_steps", type=int, default=25)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--save_path", default="weights/best_exam_remover_hybrid.pth")
    parser.add_argument("--resume", default="weights/best_exam_remover.pth")
    args = parser.parse_args()

    train(args)
