#!/usr/bin/env python3
"""
信心度門檻與後處理微調實驗腳本 (Threshold & Post-processing Fine-tuner)
針對真實考卷 20251025101835564.pdf (直式 B4，印刷密集、手寫豐富) 進行系統性參數掃描。
評估指標：
1. 印刷文字保護率 (Text Preservation Rate, gray < 80 區域中未被誤擦除的比例)
2. 誤殺像素數 (False Positive Pixels on text)
3. 總擦除比例 (Total Erased Ratio)
4. 連通域面積過濾效果 (Min Area Filter: 移除 1~8px 孤立雜點)
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

def get_page_image(pdf_path, page_idx, dpi=150):
    doc = fitz.open(pdf_path)
    page = doc[page_idx]
    mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pix = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("png")
    doc.close()
    return cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

def predict_raw_probs(model, img_bgr, device="cuda"):
    H, W = img_bgr.shape[:2]
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    im_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    norm_img = (im_rgb - mean) / std
    tensor = torch.from_numpy(norm_img).permute(2, 0, 1).unsqueeze(0).float()

    pad_h = ((H + 31) // 32) * 32
    pad_w = ((W + 31) // 32) * 32
    dh = pad_h - H
    dw = pad_w - W
    if dh > 0 or dw > 0:
        tensor = F.pad(tensor, (0, dw, 0, dh), mode="constant", value=0.0)

    tensor = tensor.to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)[0, 0]

    if dh > 0 or dw > 0:
        probs = probs[:H, :W]

    return probs.cpu().numpy()

def filter_small_components(binary_mask, min_area=8):
    """過濾小於 min_area 的孤立假陽性微噪點"""
    if min_area <= 0:
        return binary_mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    clean_mask = np.zeros_like(binary_mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            clean_mask[labels == i] = 255
    return clean_mask

def evaluate():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...")
    model = load_remover_model(device=device)

    pdf_path = "20251025101835564.pdf"
    
    # 測試 Page 1 (第一頁，手寫標記 + 大量文字) 與 Page 2 (第二頁，先前最容易誤殺的密集題目頁)
    for page_idx in [0, 1]:
        img = get_page_image(pdf_path, page_idx)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 定義核心深黑印刷文字區 (gray < 80) 與次深文字區 (gray < 120)
        core_text = (gray < 80)
        total_core_text_pixels = np.sum(core_text)

        probs = predict_raw_probs(model, img, device=device)

        print(f"\n=======================================================")
        print(f"📊 正在評估: {pdf_path} Page {page_idx+1}")
        print(f"   總核心文字像素 (gray < 80): {total_core_text_pixels:,}")
        print(f"=======================================================")
        print(f"{'門檻 (Thresh)':<14} | {'膨脹 (Dilate)':<12} | {'最小面積 (MinArea)':<16} | {'誤殺像素數':<10} | {'文字保留率 (%)':<14} | {'全頁擦除率 (%)':<12}")
        print("-" * 88)

        # 掃描組合
        configs = [
            # 原版基準線
            (0.45, True, 0),
            # 單純提高信心度門檻 (0.50, 0.60, 0.65, 0.70, 0.75, 0.80) 仍有膨脹
            (0.55, True, 0),
            (0.65, True, 0),
            (0.75, True, 0),
            # 移除粗暴膨脹 (No Dilate)
            (0.45, False, 0),
            (0.55, False, 0),
            (0.65, False, 0),
            (0.70, False, 0),
            (0.75, False, 0),
            (0.80, False, 0),
            # 結合去除微噪點 (No Dilate + MinArea 8~15)
            (0.60, False, 8),
            (0.65, False, 8),
            (0.65, False, 12),
            (0.70, False, 8),
            (0.70, False, 12),
            (0.75, False, 8),
        ]

        for th, use_dilate, min_area in configs:
            mask = (probs > th).astype(np.uint8) * 255
            if min_area > 0:
                mask = filter_small_components(mask, min_area=min_area)
            if use_dilate:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                mask = cv2.dilate(mask, kernel, iterations=1)
            
            # 計算誤殺
            killed_text = np.sum((mask > 0) & core_text)
            preservation_rate = (1.0 - (killed_text / total_core_text_pixels)) * 100.0
            total_erased_rate = (np.sum(mask > 0) / mask.size) * 100.0

            dilate_str = "3x3 橢圓" if use_dilate else "無 (None)"
            print(f"{th:<14.2f} | {dilate_str:<12} | {min_area:<16} | {killed_text:<10} | {preservation_rate:<14.2f}% | {total_erased_rate:<12.2f}%")

if __name__ == "__main__":
    evaluate()
