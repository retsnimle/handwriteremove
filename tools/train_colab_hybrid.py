#!/usr/bin/env python3
"""
三軌混合考卷手寫擦除訓練器 (Colab Tri-Track Hybrid & Decoupled Loss Runner)
架構特點：
1. 三軌混合數據來源：
   - 60% 臺灣學測在地掃描全動態包覆合成 (突破空間隔離，1.8~3.5x 大字騎乘覆蓋)
   - 20% ExamInk-Seg 真實試卷庫 (1,511 對真實學生塗改、密集算式與淡鉛筆)
   - 20% 雙尺度喚醒飛輪偽標籤 (專攻 1.0x 失效之超大字與弧線難例)
2. 解耦損失函數 (DecoupledExamLoss)：
   - 5px 形態學安全緩衝區 (消滅重疊處的交界恐懼梯度)
   - 15px 跨字手寫 4.0x 正向召回激勵 (解決機率坍塌至 0.0001 的根因)
3. 熱啟動遷移：從現有權重續練微調 8 個 Epochs
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

# 加入專案路徑
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from losses.decoupled_exam_loss import DecoupledExamLoss
try:
    from tools.dataset_hybrid import HybridExamDataset, hybrid_collate_fn
except ModuleNotFoundError:
    from dataset_hybrid import HybridExamDataset, hybrid_collate_fn

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== 三軌真實+包覆合成+偽標籤混合考卷去手寫訓練 ===")
    print(f"運算裝置: {device} ({torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'})")
    print(f"配置: Batch={args.batch_size}, Epochs={args.epochs}, LR={args.lr}, RealRatio={args.real_ratio}, PseudoRatio={args.pseudo_ratio}")

    # 1. 建立三軌資料集
    train_ds = HybridExamDataset(
        examink_root=args.examink_root,
        scanned_root=args.scanned_root,
        assets_root=args.assets_root,
        catalog_path=args.catalog_path,
        pseudo_root=args.pseudo_root,
        total_steps=args.steps_per_epoch * args.batch_size,
        dpi=150,
        is_train=True,
        real_ratio=args.real_ratio,
        pseudo_ratio=args.pseudo_ratio
    )
    val_ds = HybridExamDataset(
        examink_root=args.examink_root,
        scanned_root=args.scanned_root,
        assets_root=args.assets_root,
        catalog_path=args.catalog_path,
        pseudo_root=args.pseudo_root,
        total_steps=args.val_steps * args.batch_size,
        dpi=150,
        is_train=False,
        real_ratio=args.real_ratio,
        pseudo_ratio=args.pseudo_ratio
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=hybrid_collate_fn,
        pin_memory=True if device == "cuda" else False
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=hybrid_collate_fn,
        pin_memory=True if device == "cuda" else False
    )

    # 2. 構建 ResNet34-UNet
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None
    ).to(device)

    # 熱啟動載入
    if args.resume and os.path.exists(args.resume):
        print(f"[Model] 正在熱啟動權重: {args.resume}")
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state, strict=False)
        print("[Model] ★ 歷史最優權重載入完畢！開始微調。")
    else:
        print("[Model] ⚠️ 未指定或未找到 resume 權重，將從 ImageNet 初始特徵開始訓練。")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    # 使用解耦損失函數
    criterion = DecoupledExamLoss(
        bce_weight=0.5,
        dice_weight=1.0,
        text_fp_weight=args.text_penalty,
        overlap_recall_boost=4.0,
        buffer_kernel=5,
        proximity_kernel=15
    )

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
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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

    print(f"🎉 三軌解耦訓練全數完成！最佳模型已儲存至: {args.save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--examink_root", default="/content/ExamInk-Seg/data")
    parser.add_argument("--scanned_root", default="scanned_pairs")
    parser.add_argument("--assets_root", default="assets")
    parser.add_argument("--catalog_path", default="assets/scanned_catalog.json")
    parser.add_argument("--pseudo_root", default="output/pseudo_hard_dataset")
    parser.add_argument("--real_ratio", type=float, default=0.20, help="真實 ExamInk 採樣比例")
    parser.add_argument("--pseudo_ratio", type=float, default=0.08, help="飛輪偽標籤採樣比例 (防過擬合設為 0.08)")
    parser.add_argument("--epochs", type=int, default=8)
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
