import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset_hybrid import HybridExamDataset, hybrid_collate_fn

def export_true_batch_preview():
    # 建立與訓練一模一樣的真實 DataLoader
    ds = HybridExamDataset(
        examink_root="assets",  # 即使本地無完整 examink，hybrid 會以 scanned 為主
        scanned_root="scanned_pairs",
        assets_root="assets",
        catalog_path="assets/exam_anchors_catalog.json",
        total_steps=20,
        dpi=150,
        is_train=True,
        real_ratio=0.0  # 本地生成全數檢驗在地考卷與大字蓋印
    )
    
    loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=hybrid_collate_fn)
    
    MEAN = np.array([0.485, 0.456, 0.406])
    STD = np.array([0.229, 0.224, 0.225])
    
    os.makedirs("output/qa", exist_ok=True)
    
    saved_count = 0
    for imgs, masks, pure_texts in loader:
        B = imgs.shape[0]
        rows = []
        for b in range(B):
            # 反歸一化影像 (RGB -> BGR)
            im_norm = imgs[b].permute(1, 2, 0).cpu().numpy()
            im_rgb = (im_norm * STD + MEAN) * 255.0
            im_bgr = np.clip(im_rgb[:, :, [2, 1, 0]], 0, 255).astype(np.uint8)
            
            # 手寫 mask (綠色高亮半透明疊加)
            m = masks[b, 0].cpu().numpy() > 0.5
            vis_overlay = im_bgr.copy()
            vis_overlay[m] = [0, 0, 255] # 紅色表示手寫真值標籤
            overlay = cv2.addWeighted(im_bgr, 0.65, vis_overlay, 0.35, 0)
            
            # 純印刷文字保護 mask (青色高亮)
            pt = pure_texts[b, 0].cpu().numpy() > 0.5
            vis_pt = im_bgr.copy()
            vis_pt[pt] = [255, 200, 0] # 青藍色表示受保護的印刷字體
            overlay_pt = cv2.addWeighted(im_bgr, 0.70, vis_pt, 0.30, 0)
            
            row = np.hstack([im_bgr, overlay, overlay_pt])
            rows.append(row)
        
        full_vis = np.vstack(rows)
        save_path = f"output/qa/true_batch_giant_chars_{saved_count+1}.jpg"
        cv2.imwrite(save_path, full_vis)
        print(f"Saved: {save_path}")
        saved_count += 1
        if saved_count >= 2:
            break

if __name__ == "__main__":
    export_true_batch_preview()
