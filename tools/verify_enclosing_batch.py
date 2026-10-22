import os
import random
import cv2
import numpy as np
import torch

from dataset_dynamic_scanned import FullDynamicScannedDataset

def verify():
    ds = FullDynamicScannedDataset(
        scanned_root="scanned_pairs",
        assets_root="assets",
        catalog_path="assets/scanned_catalog.json",
        anchors_path="assets/exam_anchors_catalog.json",
        total_steps=100,
        dpi=150,
        is_train=True
    )

    MEAN = np.array([0.485, 0.456, 0.406])
    STD = np.array([0.229, 0.224, 0.225])

    os.makedirs("output/qa", exist_ok=True)
    
    # 產出 3 個不同的真實 batch sample
    samples_saved = 0
    for i in range(15):
        tensor_img, tensor_mask, tensor_pt = ds[i]
        
        im_norm = tensor_img.permute(1, 2, 0).numpy()
        im_rgb = (im_norm * STD + MEAN) * 255.0
        im_bgr = np.clip(im_rgb[:, :, [2, 1, 0]], 0, 255).astype(np.uint8)
        mask = tensor_mask[0].numpy() > 0.5
        pt = tensor_pt[0].numpy() > 0.5

        # 檢查手寫像素量是否充沛 (大字生成)
        if mask.sum() > 1500:
            vis_m = im_bgr.copy()
            vis_m[mask] = [0, 0, 255] # 紅色高亮手寫標籤
            overlay_m = cv2.addWeighted(im_bgr, 0.65, vis_m, 0.35, 0)
            
            vis_pt = im_bgr.copy()
            vis_pt[pt] = [255, 200, 0] # 青藍色高亮受保護印刷字
            overlay_pt = cv2.addWeighted(im_bgr, 0.70, vis_pt, 0.30, 0)
            
            combo = np.hstack([im_bgr, overlay_m, overlay_pt])
            out_file = f"output/qa/stage2_enclosing_batch_{samples_saved+1}.jpg"
            cv2.imwrite(out_file, combo)
            print(f"Saved: {out_file} (mask pixels: {mask.sum()})")
            samples_saved += 1
            if samples_saved >= 3:
                break

if __name__ == "__main__":
    verify()
