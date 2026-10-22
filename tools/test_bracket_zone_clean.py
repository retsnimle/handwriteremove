#!/usr/bin/env python3
"""
作答區幾何定錨與強效去手寫驗證 (Answer Zone Anchor & Aggressive Erase Test)
利用 OCR 抓到的每題正文開頭 (右括號 ) 坐標，反向推導作答框坐標 [x_start - 120, x_start]，
在作答框內部施加「高靈敏度手寫消除」，同時保護印刷數字與括號！
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import fitz
import segmentation_models_pytorch as smp
from rapidocr_onnxruntime import RapidOCR

dev = "cuda" if torch.cuda.is_available() else "cpu"

# 1. 載入模型
model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1).to(dev)
model.load_state_dict(torch.load("weights/best_exam_remover_hybrid.pth", map_location=dev, weights_only=True))
model.eval()

MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

doc = fitz.open("20251020202938305.pdf")
pix = doc[0].get_pixmap(dpi=150)
raw_bgr = cv2.imdecode(np.frombuffer(pix.tobytes("png"), np.uint8), cv2.IMREAD_COLOR)
doc.close()

H, W, _ = raw_bgr.shape

# 2. U-Net 取得整頁的手寫預測機率圖
t_raw = torch.from_numpy(raw_bgr).to(dev)
t_rgb = t_raw[..., [2, 1, 0]].permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
t_norm = (t_rgb - MEAN) / STD

pad_h = ((H + 31) // 32) * 32 - H
pad_w = ((W + 31) // 32) * 32 - W
t_pad = F.pad(t_norm, (0, pad_w, 0, pad_h), mode='constant', value=0.0) if (pad_h > 0 or pad_w > 0) else t_norm

with torch.no_grad():
    probs = torch.sigmoid(model(t_pad))[0, 0, :H, :W].cpu().numpy()

# 3. RapidOCR 快速偵測每題正文開頭 (定位作答錨點)
ocr = RapidOCR()
ocr_results, _ = ocr(raw_bgr)

# 建立分區門檻圖 (預設全圖一般門檻 0.50，作答區超敏感門檻 0.15)
adaptive_threshold_map = np.ones((H, W), dtype=np.float32) * 0.50

anchors = []
for item in ocr_results:
    bbox, text, score = item
    pts = np.array(bbox, dtype=np.int32)
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    
    # 若文字以 '）' 或 ')' 開頭，或者包含題號開頭
    if text.startswith("）") or text.startswith(")") or "）如圖" in text or "）有關" in text or "5.(" in text:
        # 作答框在正文起點的左側約 110~130px
        box_x1 = max(0, x_min - 120)
        box_x2 = x_min + 15
        box_y1 = max(0, y_min - 10)
        box_y2 = min(H, y_max + 10)
        anchors.append((box_x1, box_y1, box_x2, box_y2, text[:10]))
        # 將作答區門檻直接降到 0.15 (激進擦除手寫答案)
        adaptive_threshold_map[box_y1:box_y2, box_x1:box_x2] = 0.15

print(f"★ 成功定位 {len(anchors)} 個題目作答框錨點！")

# 4. 執行自適應分區推論
final_mask = probs > adaptive_threshold_map

# 5. 比較題 1 與題 2 區域
cleaned_img = raw_bgr.copy()
cleaned_img[final_mask] = [252, 252, 252]

# 裁切題 1 與題 2 區域特寫
crop_orig = raw_bgr[300:850, 100:900].copy()
crop_cleaned = cleaned_img[300:850, 100:900].copy()

cv2.putText(crop_orig, "Original Exam", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
cv2.putText(crop_cleaned, "Zone Adaptive Cleaned", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 180, 0), 2)

comp = np.hstack([crop_orig, crop_cleaned])
out_p = "output/qa/zone_adaptive_q1_q2_compare.png"
cv2.imwrite(out_p, comp)
print(f"★ 成果特寫對照已儲存: {out_p}")
