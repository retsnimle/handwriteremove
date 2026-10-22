#!/usr/bin/env python3
"""
新舊模型橫向對比測試 (Old Dynamic vs Hybrid Real+Dynamic)
對比兩大模型在同一測試區域下的表現：
- Model A: 舊版動態模型 (weights/best_exam_remover.pth)
- Model B: 雙軌混合新模型 (weights/best_exam_remover_hybrid.pth)
在門檻 0.50 下進行視覺並排對比
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import fitz
import segmentation_models_pytorch as smp

def run_compare():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 載入兩套模型
    old_model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1).to(dev)
    old_model.load_state_dict(torch.load("weights/best_exam_remover.pth", map_location=dev, weights_only=True))
    old_model.eval()

    new_model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1).to(dev)
    new_model.load_state_dict(torch.load("weights/best_exam_remover_hybrid.pth", map_location=dev, weights_only=True))
    new_model.eval()

    MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

    qa_dir = "output/qa"
    os.makedirs(qa_dir, exist_ok=True)

    # 1. 測試 20251020202938305.pdf Page 1 (手寫塗鴉與複雜算式)
    doc = fitz.open("20251020202938305.pdf")
    pix = doc[0].get_pixmap(dpi=150)
    raw_bgr = cv2.imdecode(np.frombuffer(pix.tobytes("png"), np.uint8), cv2.IMREAD_COLOR)
    doc.close()

    H, W, _ = raw_bgr.shape
    t_raw = torch.from_numpy(raw_bgr).to(dev)
    t_rgb = t_raw[..., [2, 1, 0]].permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
    t_norm = (t_rgb - MEAN) / STD

    pad_h = ((H + 31) // 32) * 32 - H
    pad_w = ((W + 31) // 32) * 32 - W
    t_pad = F.pad(t_norm, (0, pad_w, 0, pad_h), mode='constant', value=0.0) if (pad_h > 0 or pad_w > 0) else t_norm

    with torch.no_grad():
        p_old = torch.sigmoid(old_model(t_pad))[0, 0, :H, :W].cpu().numpy()
        p_new = torch.sigmoid(new_model(t_pad))[0, 0, :H, :W].cpu().numpy()

    # 取算式與塗鴉密集區
    y1, y2, x1, x2 = int(H * 0.20), int(H * 0.20) + 400, int(W * 0.15), int(W * 0.15) + 650
    crop_orig = raw_bgr[y1:y2, x1:x2].copy()
    cv2.putText(crop_orig, "Original Exam", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # 舊模型推論 (Th=0.50)
    m_old = p_old > 0.50
    cln_old = raw_bgr.copy()
    cln_old[m_old] = [252, 252, 252]
    crop_old = cln_old[y1:y2, x1:x2].copy()
    cv2.putText(crop_old, "Old Model (~60% Recall)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # 新雙軌模型推論 (Th=0.45)
    m_new = p_new > 0.45
    cln_new = raw_bgr.copy()
    cln_new[m_new] = [252, 252, 252]
    crop_new = cln_new[y1:y2, x1:x2].copy()
    cv2.putText(crop_new, "Hybrid Model (Real+Dyn)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 180, 0), 2)

    compare_strip = np.hstack([crop_orig, crop_old, crop_new])
    out_path = os.path.join(qa_dir, "compare_old_vs_hybrid_p1.png")
    cv2.imwrite(out_path, compare_strip)
    print(f"★ 新舊模型對比圖已儲存: {out_path}")

if __name__ == "__main__":
    run_compare()
