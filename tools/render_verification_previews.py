#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, r"G:\CODE\handwriteremove")
sys.path.insert(0, r"G:\CODE\handwriteremove\tools")
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset_dynamic_scanned import FullDynamicScannedDataset, full_dynamic_collate_fn

ARTIFACT_DIR = r"C:\Users\retsn\.gemini\antigravity\brain\f687af4b-740f-482b-a1f5-e84a253db0a3\qa"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

dataset = FullDynamicScannedDataset(
    scanned_root=r"G:\CODE\handwriteremove\scanned_pairs",
    assets_root=r"G:\CODE\handwriteremove\assets",
    catalog_path=r"G:\CODE\handwriteremove\assets\scanned_catalog.json",
    anchors_path=r"G:\CODE\handwriteremove\assets\exam_anchors_catalog.json",
    total_steps=100,
    dpi=150,
    is_train=True
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0,
    collate_fn=full_dynamic_collate_fn
)

it = iter(loader)
next(it)
next(it)
next(it)
batch = next(it)
imgs, masks, pure_texts = batch

vis_list = []
for i in range(imgs.size(0)):
    im = imgs[i].permute(1, 2, 0).numpy()
    im = np.clip((im * std + mean) * 255.0, 0, 255).astype(np.uint8)
    im_bgr = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

    m = (masks[i, 0].numpy() > 0.5).astype(np.uint8) * 255
    txt = (pure_texts[i, 0].numpy() > 0.5).astype(np.uint8) * 255

    vis_mask = im_bgr.copy()
    vis_mask[m > 0] = [0, 0, 255]

    vis_protect = im_bgr.copy()
    vis_protect[txt > 0] = [0, 255, 0]

    trio = np.hstack([
        cv2.resize(im_bgr, (600, 600)),
        cv2.resize(vis_mask, (600, 600)),
        cv2.resize(vis_protect, (600, 600))
    ])
    vis_list.append(trio)

grid = np.vstack(vis_list)
out_path = os.path.join(ARTIFACT_DIR, "true_training_batch_with_qnums_2.jpg")
cv2.imwrite(out_path, grid, [cv2.IMWRITE_JPEG_QUALITY, 94])
print(f"Exported true_training_batch_with_qnums_2.jpg successfully!")
