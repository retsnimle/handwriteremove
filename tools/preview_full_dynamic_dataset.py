#!/usr/bin/env python3
"""
生成在真實考卷上「蓋上動態印章」的貼合成果預覽 (QA Inspection)
"""
import os
import cv2
import numpy as np
import torch
from dataset_dynamic_scanned import FullDynamicScannedDataset, full_dynamic_collate_fn

def main():
    ds = FullDynamicScannedDataset(
        scanned_root="scanned_pairs",
        assets_root="assets",
        catalog_path="assets/scanned_catalog.json",
        total_steps=10,
        dpi=150,
        is_train=True
    )

    qa_dir = r"C:\Users\retsn\.gemini\antigravity\brain\f687af4b-740f-482b-a1f5-e84a253db0a3\qa"
    os.makedirs(qa_dir, exist_ok=True)

    samples = [ds[i] for i in range(4)]
    imgs, masks, pure_texts = full_dynamic_collate_fn(samples)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    vis_list = []
    for i in range(imgs.size(0)):
        im = imgs[i].permute(1, 2, 0).numpy()
        im = np.clip((im * std + mean) * 255.0, 0, 255).astype(np.uint8)
        im_bgr = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

        m = (masks[i, 0].numpy() > 0.5).astype(np.uint8) * 255
        txt = (pure_texts[i, 0].numpy() > 0.5).astype(np.uint8) * 255

        vis_mask = im_bgr.copy()
        vis_mask[m > 0] = [0, 0, 255] # 手寫筆跡 (紅)

        vis_protect = im_bgr.copy()
        vis_protect[txt > 0] = [0, 255, 0] # 純印刷字保護區 (綠)

        trio = np.hstack([
            cv2.resize(im_bgr, (400, 400)),
            cv2.resize(vis_mask, (400, 400)),
            cv2.resize(vis_protect, (400, 400))
        ])
        vis_list.append(trio)

    grid = np.vstack(vis_list)
    out_path = os.path.join(qa_dir, "full_dynamic_synth_preview.jpg")
    cv2.imwrite(out_path, grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"雙端動態全頁合成預覽已儲存: {out_path}")

if __name__ == "__main__":
    main()
