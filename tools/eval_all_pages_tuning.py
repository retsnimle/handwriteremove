#!/usr/bin/env python3
"""
全頁面參數對比評估 (All Pages Parameter Comparison)
對比兩份真實考卷在三種配置下的表現：
- Config A (Baseline): Threshold 0.45, Dilate 3x3, MinArea 0
- Config B (Balanced): Threshold 0.65, No Dilate, MinArea 10
- Config C (Conservative): Threshold 0.75, No Dilate, MinArea 12
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import fitz

def load_remover_model(weights_path="weights/best_exam_remover.pth", device="cuda"):
    import segmentation_models_pytorch as smp
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None
    )
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model

def filter_small_components(binary_mask, min_area=10):
    if min_area <= 0:
        return binary_mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    clean_mask = np.zeros_like(binary_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean_mask[labels == i] = 255
    return clean_mask

def predict_probs(model, img_bgr, device="cuda"):
    H, W = img_bgr.shape[:2]
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    im_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    norm_img = (im_rgb - mean) / std
    tensor = torch.from_numpy(norm_img).permute(2, 0, 1).unsqueeze(0).float()
    pad_h = ((H + 31) // 32) * 32
    pad_w = ((W + 31) // 32) * 32
    dh, dw = pad_h - H, pad_w - W
    if dh > 0 or dw > 0:
        tensor = F.pad(tensor, (0, dw, 0, dh), mode="constant", value=0.0)
    tensor = tensor.to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)[0, 0]
    if dh > 0 or dw > 0:
        probs = probs[:H, :W]
    return probs.cpu().numpy()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_remover_model(device=device)

    target_pdfs = [
        ("20251020202938305.pdf", "橫向 A3 跨頁雙欄 (4 頁)"),
        ("20251025101835564.pdf", "直式 B4 考卷 (8 頁)")
    ]

    print(f"\n{'='*95}")
    print(f"{'PDF 檔案與頁碼':<24} | {'核心文字像素':<10} | {'基準誤殺 (0.45+Dil)':<18} | {'調優誤殺 (0.65+Filter)':<20} | {'誤殺降幅 (%)':<10}")
    print(f"{'='*95}")

    for pdf_name, desc in target_pdfs:
        doc = fitz.open(pdf_name)
        for pi in range(len(doc)):
            page = doc[pi]
            mat = fitz.Matrix(150 / 72.0, 150 / 72.0)
            pix = page.get_pixmap(matrix=mat)
            raw_bgr = cv2.imdecode(np.frombuffer(pix.tobytes("png"), np.uint8), cv2.IMREAD_COLOR)
            
            gray = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY)
            core_text = (gray < 80)
            total_text = np.sum(core_text)

            probs = predict_probs(model, raw_bgr, device=device)

            # 基準線 (0.45 + Dilate)
            m_base = (probs > 0.45).astype(np.uint8) * 255
            m_base = cv2.dilate(m_base, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
            kill_base = np.sum((m_base > 0) & core_text)

            # 調優版 (0.65 + No Dilate + MinArea 10)
            m_tuned = (probs > 0.65).astype(np.uint8) * 255
            m_tuned = filter_small_components(m_tuned, min_area=10)
            kill_tuned = np.sum((m_tuned > 0) & core_text)

            # 保守極致保護版 (0.75 + No Dilate + MinArea 12)
            m_ultra = (probs > 0.75).astype(np.uint8) * 255
            m_ultra = filter_small_components(m_ultra, min_area=12)
            kill_ultra = np.sum((m_ultra > 0) & core_text)

            drop_rate = ((kill_base - kill_tuned) / max(kill_base, 1)) * 100.0
            
            page_id = f"{pdf_name[:12]}.. P{pi+1:02d}"
            base_str = f"{kill_base:5d} ({100-kill_base/total_text*100:5.2f}%)"
            tuned_str = f"{kill_tuned:5d} ({100-kill_tuned/total_text*100:5.2f}%)"
            print(f"{page_id:<24} | {total_text:<10,d} | {base_str:<18} | {tuned_str:<20} | -{drop_rate:5.1f}%")

        doc.close()

if __name__ == "__main__":
    main()
