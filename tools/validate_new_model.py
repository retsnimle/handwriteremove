#!/usr/bin/env python3
"""
全新雙端全動態抗過擬合模型效果檢驗 (New Dynamic Model Validation)
針對先前誤殺最嚴重的 20251025101835564.pdf Page 1 (原圖、原生無銳化推論、輸出特寫)，
檢驗在完全「不加任何銳化、不手動校正」的原生狀態下，文字是否能自然被保護，手寫是否能乾淨抹除！
"""
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import fitz
import segmentation_models_pytorch as smp

dev = "cuda"
model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
model.load_state_dict(torch.load("weights/best_exam_remover.pth", map_location=dev, weights_only=True))
model.to(dev).eval()

MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

doc = fitz.open("20251025101835564.pdf")
page = doc[0]
pix = page.get_pixmap(dpi=150)
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
    logits = model(t_pad)
    probs = torch.sigmoid(logits)[0, 0, :H, :W]

probs_np = probs.cpu().numpy()

# 測試不同門檻 (0.45, 0.55, 0.65) 在原生完全不銳化狀態下的表現
qa_dir = "output/qa"
os.makedirs(qa_dir, exist_ok=True)

# 裁切先前最敏感的題 3 英文選項區 (A)(B) DNA
y1, y2, x1, x2 = int(H * 0.35), int(H * 0.35) + 360, 80, 720
raw_crop = raw_bgr[y1:y2, x1:x2].copy()
cv2.putText(raw_crop, "Original", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

cards = [raw_crop]
for th in [0.45, 0.55, 0.65]:
    m = (probs_np > th).astype(np.uint8)
    cln = raw_bgr.copy()
    cln[m > 0] = [252, 252, 252]
    c = cln[y1:y2, x1:x2].copy()
    cv2.putText(c, f"Thresh={th:.2f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cards.append(c)

vis_compare = np.hstack(cards)
out_path = os.path.join(qa_dir, "new_dynamic_model_p1_crop_test.png")
cv2.imwrite(out_path, vis_compare)
print(f"原生推論成果切片已儲存: {out_path}")

# 計算全頁核心印刷字誤殺率與總擦除率
gray = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY)
core = gray < 80
for th in [0.45, 0.55, 0.65]:
    m = probs_np > th
    kill = (core & m).sum()
    print(f"門檻 {th:.2f}: 核心文字誤殺={kill} ({kill/core.sum()*100:.2f}%) | 全頁擦除率={m.mean()*100:.2f}%")
