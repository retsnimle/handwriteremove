import cv2
import numpy as np
import fitz
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp

device = "cuda" if torch.cuda.is_available() else "cpu"
model = smp.Unet(encoder_name="resnet34", in_channels=3, classes=1)
model.load_state_dict(torch.load("weights/best_exam_remover_hybrid.pth", map_location=device, weights_only=False))
model.to(device).eval()

lama_model = torch.jit.load("weights/big-lama.pt", map_location=device).eval()

MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

def infer_scale(bgr):
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

p_10 = infer_scale(bgr)
bgr_half = cv2.resize(bgr, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
p_half = infer_scale(bgr_half)
p_05_raw = cv2.resize(p_half, (W, H), interpolation=cv2.INTER_LINEAR)

# 1. 修正後的印刷文字護盾：
# 高解析度原生尺度 p_10 < 0.20 且灰度深 (gray < 110) 的像素，100% 屬於印刷文字！
printed_skeleton = (gray < 110) & (p_10 < 0.20)
k_3x3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
printed_shield = (cv2.dilate(printed_skeleton.astype(np.uint8), k_3x3) > 0) & (gray < 210) & (p_10 < 0.35)

# 2. 融合機率：在護盾保護區外，雙尺度取最大值
p_fused = np.maximum(p_10, p_05_raw)
p_fused[printed_shield] = 0.0

# 3. 獲取擦除遮罩
erase_mask = ((p_fused > 0.35) & (~printed_shield)).astype(np.uint8) * 255
# 輕微膨脹 1px 消除邊緣泛白
k_3x3_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
erase_mask = cv2.dilate(erase_mask, k_3x3_rect)
erase_mask[printed_shield] = 0

# 4. 局部 LaMa 重繪 Question 1 區域 (y: 360~660, x: 20~300)
crop_rgb = cv2.cvtColor(bgr[360:660, 20:300], cv2.COLOR_BGR2RGB)
crop_mask = erase_mask[360:660, 20:300]

# 調用 LaMa
t_img = torch.from_numpy(crop_rgb).to(device).permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
t_mask = torch.from_numpy(crop_mask).to(device).unsqueeze(0).unsqueeze(0).float().div_(255.0)

ch, cw = crop_rgb.shape[:2]
pad_h = ((ch + 31) // 32) * 32 - ch
pad_w = ((cw + 31) // 32) * 32 - cw

t_img_pad = F.pad(t_img, (0, pad_w, 0, pad_h), mode="reflect")
t_mask_pad = F.pad(t_mask, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

with torch.no_grad():
    out_pad = lama_model(t_img_pad, t_mask_pad)

crop_lama = out_pad[0, :, :ch, :cw].permute(1, 2, 0).clamp(0, 1).mul(255).byte().cpu().numpy()
crop_lama_bgr = cv2.cvtColor(crop_lama, cv2.COLOR_RGB2BGR)

# 塗白對照
crop_white = bgr[360:660, 20:300].copy()
crop_white[crop_mask > 0] = 255

crop_orig = bgr[360:660, 20:300]
crop_mask_3c = cv2.cvtColor(crop_mask, cv2.COLOR_GRAY2BGR)

# 四聯切片: [原圖 (大D), 擦除遮罩 (保護 1.), 塗白對照, LaMa 重繪]
res_comp = np.hstack([crop_orig, crop_mask_3c, crop_white, crop_lama_bgr])
out_p = "output/qa/perfect_q1_lama.png"
cv2.imwrite(out_p, res_comp)
print("★ 已產出完美題號保護與 LaMa 重繪圖:", out_p)
