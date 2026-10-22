#!/usr/bin/env python3
"""
分析 Page 1 題目選項區的預測概率熱力圖
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

doc = fitz.open("20251025101835564.pdf")
page = doc[0]
mat = fitz.Matrix(150 / 72.0, 150 / 72.0)
pix = page.get_pixmap(matrix=mat)
raw_bgr = cv2.imdecode(np.frombuffer(pix.tobytes("png"), np.uint8), cv2.IMREAD_COLOR)
doc.close()

H, W = raw_bgr.shape[:2]
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
im_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
norm_img = (im_rgb - mean) / std
tensor = torch.from_numpy(norm_img).permute(2, 0, 1).unsqueeze(0).float()
pad_h = ((H + 31) // 32) * 32
pad_w = ((W + 31) // 32) * 32
dh, dw = pad_h - H, pad_w - W
if dh > 0 or dw > 0:
    tensor = F.pad(tensor, (0, dw, 0, dh), mode="constant", value=0.0)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_remover_model(device=device)
tensor = tensor.to(device)

with torch.no_grad():
    logits = model(tensor)
    probs = torch.sigmoid(logits)[0, 0]
if dh > 0 or dw > 0:
    probs = probs[:H, :W]

probs_np = probs.cpu().numpy()

# 裁切中段選項區
y1, y2, x1, x2 = int(H * 0.4), int(H * 0.4) + 400, 100, 700
crop_prob = probs_np[y1:y2, x1:x2]
crop_raw = raw_bgr[y1:y2, x1:x2]

# 印出文字區域的概率分佈
gray = cv2.cvtColor(crop_raw, cv2.COLOR_BGR2GRAY)
text_mask = (gray < 80)
text_probs = crop_prob[text_mask]

print(f"Crop Text Pixels: {np.sum(text_mask)}")
print(f"Text Probs: min={text_probs.min():.4f}, median={np.median(text_probs):.4f}, mean={text_probs.mean():.4f}, max={text_probs.max():.4f}")
print(f"Percentiles:")
for p in [10, 25, 50, 75, 90, 95, 99]:
    print(f"  P{p}: {np.percentile(text_probs, p):.4f}")

# 生成熱力圖
heatmap = cv2.applyColorMap((crop_prob * 255).astype(np.uint8), cv2.COLORMAP_JET)
overlay = cv2.addWeighted(crop_raw, 0.6, heatmap, 0.4, 0)

qa_dir = r"C:\Users\retsn\.gemini\antigravity\brain\f687af4b-740f-482b-a1f5-e84a253db0a3\qa"
cv2.imwrite(os.path.join(qa_dir, "p01_heatmap_crop.png"), overlay)
cv2.imwrite(os.path.join(qa_dir, "p01_prob_gray.png"), (crop_prob * 255).astype(np.uint8))
print("Saved p01 heatmap crops.")
