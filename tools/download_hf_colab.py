#!/usr/bin/env python3
"""
在 Colab 遠端下載 ynyg/ExamInk-Seg 並驗證目錄結構
"""
import os
import glob
from huggingface_hub import snapshot_download

print("[Colab] 正在從 Hugging Face 下載 ynyg/ExamInk-Seg 數據集...")
local_dir = "/content/ExamInk-Seg"
snapshot_download(
    repo_id="ynyg/ExamInk-Seg",
    repo_type="dataset",
    local_dir=local_dir,
    max_workers=8
)
print("[Colab] ★ 下載完成！正在檢查文件結構...")

train_source = glob.glob(f"{local_dir}/data/train/source/*.jpg") + glob.glob(f"{local_dir}/data/train/source/*.png")
train_mask = glob.glob(f"{local_dir}/data/train/mask/*.jpg") + glob.glob(f"{local_dir}/data/train/mask/*.png")
train_target = glob.glob(f"{local_dir}/data/train/target/*.jpg") + glob.glob(f"{local_dir}/data/train/target/*.png")

val_source = glob.glob(f"{local_dir}/data/val/source/*.jpg") + glob.glob(f"{local_dir}/data/val/source/*.png")
val_mask = glob.glob(f"{local_dir}/data/val/mask/*.jpg") + glob.glob(f"{local_dir}/data/val/mask/*.png")

print(f"Train samples: source={len(train_source)}, mask={len(train_mask)}, target={len(train_target)}")
print(f"Val samples:   source={len(val_source)}, mask={len(val_mask)}")
