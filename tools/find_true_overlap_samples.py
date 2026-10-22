import os
import random
import cv2
import numpy as np
import torch

from dataset_hybrid import HybridExamDataset

def find_overlaps():
    ds = HybridExamDataset(
        examink_root="assets",
        scanned_root="scanned_pairs",
        assets_root="assets",
        catalog_path="assets/exam_anchors_catalog.json",
        dpi=150,
        is_train=True,
        real_ratio=0.0
    )

    MEAN = np.array([0.485, 0.456, 0.406])
    STD = np.array([0.229, 0.224, 0.225])

    os.makedirs("output/qa", exist_ok=True)
    found = 0
    for i in range(100):
        tensor_img, tensor_mask, tensor_pt = ds[i]
        im_norm = tensor_img.permute(1, 2, 0).numpy()
        im_rgb = (im_norm * STD + MEAN) * 255.0
        im_bgr = np.clip(im_rgb[:, :, [2, 1, 0]], 0, 255).astype(np.uint8)
        mask = tensor_mask[0].numpy() > 0.5
        pt = tensor_pt[0].numpy() > 0.5

        # 找左側邊界區域 (x: 0~350) 有題號且被手寫 mask 重疊的樣本
        # 題號/正文處通常灰度 < 100
        gray = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY)
        left_overlap = (gray[:, :350] < 100) & mask[:, :350]

        if left_overlap.sum() > 400: # 確實在題號/題幹左側有大面積重疊
            # 切出左側重疊的局部 (y: 隨重疊位置動態定位, x: 0~450)
            ys, xs = np.where(left_overlap)
            ymin = max(0, ys.min() - 40)
            ymax = min(768, ys.max() + 40)
            
            crop_bgr = im_bgr[ymin:ymax, 0:450]
            crop_mask = mask[ymin:ymax, 0:450]
            crop_pt = pt[ymin:ymax, 0:450]
            
            vis_m = crop_bgr.copy()
            vis_m[crop_mask] = [0, 0, 255]
            overlay_m = cv2.addWeighted(crop_bgr, 0.65, vis_m, 0.35, 0)
            
            vis_pt = crop_bgr.copy()
            vis_pt[crop_pt] = [255, 200, 0]
            overlay_pt = cv2.addWeighted(crop_bgr, 0.70, vis_pt, 0.30, 0)
            
            combo = np.hstack([crop_bgr, overlay_m, overlay_pt])
            out_path = f"output/qa/verified_real_overlap_{found+1}.jpg"
            cv2.imwrite(out_path, combo)
            print(f"Overlap sample #{found+1} saved to {out_path}: overlap_px={left_overlap.sum()}, y_range=[{ymin}, {ymax}]")
            found += 1
            if found >= 2:
                break

if __name__ == "__main__":
    find_overlaps()
