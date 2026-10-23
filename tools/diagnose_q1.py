import cv2
import numpy as np
import fitz
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp

device = "cuda" if torch.cuda.is_available() else "cpu"
model = smp.Unet(encoder_name="resnet34", in_channels=3, classes=1)
model.load_state_dict(torch.load("weights/best_exam_remover_hybrid.pth", map_location=device))
model.to(device).eval()

MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

def infer(bgr):
    H, W, _ = bgr.shape
    t = torch.from_numpy(bgr).to(device)[..., [2, 1, 0]].permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
    t = (t - MEAN) / STD
    pad_h = ((H + 31) // 32) * 32 - H
    pad_w = ((W + 31) // 32) * 32 - W
    t_pad = F.pad(t, (0, pad_w, 0, pad_h), mode='constant', value=0.0)
    with torch.no_grad():
        logits = model(t_pad)
        probs = torch.sigmoid(logits)[0, 0, :H, :W].cpu().numpy()
    return probs

doc = fitz.open("20251020202938305.pdf")
pix = doc[0].get_pixmap(dpi=150)
bgr = cv2.imdecode(np.frombuffer(pix.tobytes("png"), np.uint8), cv2.IMREAD_COLOR)
doc.close()

H, W, _ = bgr.shape
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

p_10 = infer(bgr)
bgr_half = cv2.resize(bgr, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
p_half = infer(bgr_half)
p_05_raw = cv2.resize(p_half, (W, H), interpolation=cv2.INTER_LINEAR)

# 觀察第 1 題題號 "1." (y: 370~410, x: 25~50)
q1_gray = gray[370:410, 25:50]
q1_p10 = p_10[370:410, 25:50]
q1_p05 = p_05_raw[370:410, 25:50]

print("=== 題號 '1.' 核心像素診斷 ===")
dark_mask = q1_gray < 120
print(f"深色文字像素數: {np.sum(dark_mask)}")
if np.sum(dark_mask) > 0:
    print(f"p_10 on text: min={q1_p10[dark_mask].min():.4f}, mean={q1_p10[dark_mask].mean():.4f}, max={q1_p10[dark_mask].max():.4f}")
    print(f"p_05 on text: min={q1_p05[dark_mask].min():.4f}, mean={q1_p05[dark_mask].mean():.4f}, max={q1_p05[dark_mask].max():.4f}")

# 診斷整個大 D 筆劃 (y: 360~660, x: 20~300)
crop_p10 = p_10[360:660, 20:300]
crop_p05 = p_05_raw[360:660, 20:300]
print("=== 大 D 區域機率 ===")
print(f"p_10 全區 max={crop_p10.max():.4f}, mean={crop_p10.mean():.4f}")
print(f"p_05 全區 max={crop_p05.max():.4f}, mean={crop_p05.mean():.4f}")

# 儲存可視化
vis_p10 = (p_10[360:660, 20:300] * 255).astype(np.uint8)
vis_p05 = (p_05_raw[360:660, 20:300] * 255).astype(np.uint8)
crop_orig = bgr[360:660, 20:300]
diag_comp = np.hstack([crop_orig, cv2.cvtColor(vis_p10, cv2.COLOR_GRAY2BGR), cv2.cvtColor(vis_p05, cv2.COLOR_GRAY2BGR)])
cv2.imwrite("output/qa/diag_p10_p05.png", diag_comp)
print("Saved output/qa/diag_p10_p05.png")
