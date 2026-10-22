#!/usr/bin/env python3
"""
Colab 遠端獨立訓練腳本 (由 colab-cli 直接於遠端 VM 執行)
"""

import os
import sys
import subprocess

def run_cmd(cmd):
    print(f">> [Remote Shell] {cmd}", flush=True)
    subprocess.check_call(cmd, shell=True)

print("=" * 60, flush=True)
print("🎓 考卷手寫分離模型遠端訓練管線啟動", flush=True)
print("=" * 60, flush=True)

# 1. 安裝必要依賴
run_cmd("pip install -q segmentation-models-pytorch albumentations opencv-python-headless")

# 2. 解壓縮資料集
if not os.path.exists("dataset/images"):
    print("[Remote] 正在解壓縮 exam_dataset.zip...", flush=True)
    run_cmd("unzip -q -o exam_dataset.zip -d dataset")

import glob
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

images = sorted(glob.glob("dataset/images/*.png"))
masks = sorted(glob.glob("dataset/masks/*.png"))
print(f"[Remote] 成功載入 {len(images)} 組資料！", flush=True)

class ExamDataset(Dataset):
    def __init__(self, img_paths, mask_paths, transform=None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.img_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        mask = (mask > 128).float().unsqueeze(0)
        return image, mask

# 解析度與資料增強
TARGET_H, TARGET_W = 1024, 768

train_transform = A.Compose([
    A.Resize(TARGET_H, TARGET_W),
    A.RandomBrightnessContrast(p=0.4),
    A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.2, hue=0.05, p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(TARGET_H, TARGET_W),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

split_idx = int(len(images) * 0.9)
train_dataset = ExamDataset(images[:split_idx], masks[:split_idx], transform=train_transform)
val_dataset = ExamDataset(images[split_idx:], masks[split_idx:], transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Remote] 運算裝置: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU'})", flush=True)

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    activation=None
).to(device)

criterion_bce = nn.BCEWithLogitsLoss()
criterion_dice = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

def total_loss(pred, target):
    return 0.5 * criterion_bce(pred, target) + 0.5 * criterion_dice(pred, target)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-6)
scaler = torch.cuda.amp.GradScaler()

NUM_EPOCHS = 25
best_val_loss = float("inf")
os.makedirs("weights", exist_ok=True)

print("=" * 60, flush=True)
print(f"[Remote] 開始訓練 (共 {NUM_EPOCHS} 個 Epochs)...", flush=True)
print("=" * 60, flush=True)

t_start = time.time()

for epoch in range(NUM_EPOCHS):
    t_epoch = time.time()
    model.train()
    train_loss = 0.0
    for imgs, msks in train_loader:
        imgs = imgs.to(device, non_blocking=True)
        msks = msks.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(imgs)
            loss = total_loss(outputs, msks)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item() * imgs.size(0)
        
    scheduler.step()
    train_loss /= len(train_dataset)
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for imgs, msks in val_loader:
            imgs = imgs.to(device, non_blocking=True)
            msks = msks.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                loss = total_loss(outputs, msks)
            val_loss += loss.item() * imgs.size(0)
            
    val_loss /= len(val_dataset)
    
    epoch_time = time.time() - t_epoch
    is_best = val_loss < best_val_loss
    if is_best:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "weights/best_exam_remover.pth")
        
    mark = " ★ 最佳" if is_best else ""
    print(f"Epoch [{epoch+1:02d}/{NUM_EPOCHS}] ({epoch_time:.1f}s) - Train: {train_loss:.4f} | Val: {val_loss:.4f}{mark}", flush=True)

total_minutes = (time.time() - t_start) / 60.0
print("=" * 60, flush=True)
print(f"🎉 訓練全數完成！總耗時: {total_minutes:.1f} 分鐘", flush=True)
print(f"最佳權重已儲存至 weights/best_exam_remover.pth (最低驗證損失: {best_val_loss:.4f})", flush=True)
print("=" * 60, flush=True)
