#!/usr/bin/env python3
"""
雙軌混合新模型成效驗證腳本 (Hybrid Real+Dynamic Model Validation)
測試最新訓練出的 weights/best_exam_remover_hybrid.pth：
1. 針對 20251025101835564.pdf (直式) 第 1 頁：檢查 (A)(B) DNA 英文與手寫筆跡
2. 針對 20251020202938305.pdf (橫式 A3) 第 1 頁：檢查學生紅筆/藍筆/淡鉛筆擦除率
3. 生成局部特寫對照切片至 output/qa/
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import fitz
import segmentation_models_pytorch as smp

def test_model():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== 雙軌混合模型成效驗證 (Device: {dev}) ===")
    
    weights_path = "weights/best_exam_remover_hybrid.pth"
    if not os.path.exists(weights_path):
        print(f"找不到權重檔: {weights_path}")
        return

    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
    model.load_state_dict(torch.load(weights_path, map_location=dev, weights_only=True))
    model.to(dev).eval()

    MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

    qa_dir = "output/qa"
    os.makedirs(qa_dir, exist_ok=True)

    # ---------------------------------------------------------
    # 測試 1: 20251025101835564.pdf 第 1 頁 (檢查英文選項保留 + 筆跡擦除)
    # ---------------------------------------------------------
    pdf1 = "20251025101835564.pdf"
    if os.path.exists(pdf1):
        doc = fitz.open(pdf1)
        pix = doc[0].get_pixmap(dpi=150)
        raw_bgr1 = cv2.imdecode(np.frombuffer(pix.tobytes("png"), np.uint8), cv2.IMREAD_COLOR)
        doc.close()

        H, W, _ = raw_bgr1.shape
        t_raw = torch.from_numpy(raw_bgr1).to(dev)
        t_rgb = t_raw[..., [2, 1, 0]].permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
        t_norm = (t_rgb - MEAN) / STD

        pad_h = ((H + 31) // 32) * 32 - H
        pad_w = ((W + 31) // 32) * 32 - W
        t_pad = F.pad(t_norm, (0, pad_w, 0, pad_h), mode='constant', value=0.0) if (pad_h > 0 or pad_w > 0) else t_norm

        with torch.no_grad():
            logits = model(t_pad)
            probs = torch.sigmoid(logits)[0, 0, :H, :W].cpu().numpy()

        # 核心文字區
        gray = cv2.cvtColor(raw_bgr1, cv2.COLOR_BGR2GRAY)
        core_text = gray < 85

        print(f"\n[PDF 1: {pdf1} Page 1] 尺寸: {W}x{H}")
        for th in [0.35, 0.45, 0.55, 0.65]:
            m = probs > th
            kill = (core_text & m).sum()
            kill_rate = (kill / core_text.sum()) * 100
            print(f"  門檻 {th:.2f}: 核心印刷字誤殺={kill} ({kill_rate:.3f}%) | 擦除像素佔比={m.mean()*100:.2f}%")

        # 裁切英文選項區
        y1, y2, x1, x2 = int(H * 0.35), int(H * 0.35) + 360, 80, 720
        raw_crop = raw_bgr1[y1:y2, x1:x2].copy()
        cv2.putText(raw_crop, "Original", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cards = [raw_crop]
        for th in [0.40, 0.50, 0.60]:
            m = (probs > th).astype(np.uint8)
            cln = raw_bgr1.copy()
            cln[m > 0] = [252, 252, 252]
            c = cln[y1:y2, x1:x2].copy()
            cv2.putText(c, f"Hybrid Th={th:.2f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cards.append(c)

        crop_out1 = os.path.join(qa_dir, "hybrid_model_p1_dna_crop.png")
        cv2.imwrite(crop_out1, np.hstack(cards))
        print(f"  ★ 已產出英文選項區對比: {crop_out1}")

    # ---------------------------------------------------------
    # 測試 2: 20251020202938305.pdf 第 1 頁 (橫式 A3 學生算式與手寫塗鴉)
    # ---------------------------------------------------------
    pdf2 = "20251020202938305.pdf"
    if os.path.exists(pdf2):
        doc = fitz.open(pdf2)
        pix = doc[0].get_pixmap(dpi=150)
        raw_bgr2 = cv2.imdecode(np.frombuffer(pix.tobytes("png"), np.uint8), cv2.IMREAD_COLOR)
        doc.close()

        H, W, _ = raw_bgr2.shape
        t_raw = torch.from_numpy(raw_bgr2).to(dev)
        t_rgb = t_raw[..., [2, 1, 0]].permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
        t_norm = (t_rgb - MEAN) / STD

        pad_h = ((H + 31) // 32) * 32 - H
        pad_w = ((W + 31) // 32) * 32 - W
        t_pad = F.pad(t_norm, (0, pad_w, 0, pad_h), mode='constant', value=0.0) if (pad_h > 0 or pad_w > 0) else t_norm

        with torch.no_grad():
            logits = model(t_pad)
            probs = torch.sigmoid(logits)[0, 0, :H, :W].cpu().numpy()

        gray = cv2.cvtColor(raw_bgr2, cv2.COLOR_BGR2GRAY)
        core_text = gray < 85

        print(f"\n[PDF 2: {pdf2} Page 1] 尺寸: {W}x{H}")
        for th in [0.35, 0.45, 0.55, 0.65]:
            m = probs > th
            kill = (core_text & m).sum()
            kill_rate = (kill / core_text.sum()) * 100
            print(f"  門檻 {th:.2f}: 核心印刷字誤殺={kill} ({kill_rate:.3f}%) | 擦除像素佔比={m.mean()*100:.2f}%")

        # 裁切中央題目與學生手寫計算區 (取大區塊 500x700)
        y1, y2, x1, x2 = int(H * 0.20), int(H * 0.20) + 400, int(W * 0.15), int(W * 0.15) + 600
        raw_crop = raw_bgr2[y1:y2, x1:x2].copy()
        cv2.putText(raw_crop, "Original", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cards = [raw_crop]
        for th in [0.40, 0.50, 0.60]:
            m = (probs > th).astype(np.uint8)
            cln = raw_bgr2.copy()
            cln[m > 0] = [252, 252, 252]
            c = cln[y1:y2, x1:x2].copy()
            cv2.putText(c, f"Hybrid Th={th:.2f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cards.append(c)

        crop_out2 = os.path.join(qa_dir, "hybrid_model_p1_handwriting_crop.png")
        cv2.imwrite(crop_out2, np.hstack(cards))
        print(f"  ★ 已產出手寫筆跡區對比: {crop_out2}")

if __name__ == "__main__":
    test_model()
