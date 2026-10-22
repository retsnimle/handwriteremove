#!/usr/bin/env python3
"""
診斷：為何印刷英文字母 (A)(B)、DNA 被判為手寫？
實驗：
  E1 色彩：被誤判的印刷像素 vs 未誤判的印刷像素，BGR 均值與飽和度比較
  E2 去色：把輸入轉灰階(三通道複製)再推論，誤殺是否消失
  E3 訓練底稿：在 scanned_pairs 的空白掃描頁上推論，看模型在「訓練見過的英文字母」上是否也誤判
"""
import glob
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import fitz
import segmentation_models_pytorch as smp

QA = r"C:\Users\retsn\.gemini\antigravity\brain\f687af4b-740f-482b-a1f5-e84a253db0a3\qa"
dev = "cuda"
model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
model.load_state_dict(torch.load("weights/best_exam_remover.pth", map_location=dev, weights_only=True))
model.to(dev).eval()
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)


def render(pdf, i, dpi=150):
    d = fitz.open(pdf)
    pix = d[i].get_pixmap(dpi=dpi)
    im = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
    d.close()
    return cv2.cvtColor(im, cv2.COLOR_RGB2BGR)


@torch.no_grad()
def probs(bgr):
    t = torch.from_numpy(bgr).to(dev)
    H, W, _ = t.shape
    x = t[..., [2, 1, 0]].permute(2, 0, 1).unsqueeze(0).float() / 255.0
    x = (x - MEAN) / STD
    x = F.pad(x, (0, (32 - W % 32) % 32, 0, (32 - H % 32) % 32))
    return torch.sigmoid(model(x))[0, 0, :H, :W].cpu().numpy()


def heat(bgr, p):
    hm = cv2.applyColorMap((p * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(bgr, 0.55, hm, 0.45, 0)


# ---------------- E1 / E2 ----------------
img = render("20251025101835564.pdf", 0)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
core = gray < 80
p_color = probs(img)
fp = core & (p_color > 0.65)
ok = core & (p_color < 0.1)
print("=== E1 色彩 (20251025101835564 P1) ===")
for name, m in [("誤判印刷像素", fp), ("正常印刷像素", ok)]:
    b, g, r = [img[:, :, c][m].mean() for c in range(3)]
    print(f"  {name}: n={m.sum():6d}  BGR=({b:5.1f},{g:5.1f},{r:5.1f})  S={hsv[:, :, 1][m].mean():5.1f}  gray={gray[m].mean():5.1f}")

g3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
p_gray = probs(g3)
print("=== E2 去色輸入 ===")
print(f"  彩色輸入 誤殺核心文字: {int((core & (p_color > 0.65)).sum())}")
print(f"  灰階輸入 誤殺核心文字: {int((core & (p_gray > 0.65)).sum())}")
H, W = gray.shape
y1, x1 = int(H * 0.4), 100
crop = (slice(y1, y1 + 400), slice(x1, x1 + 700))
cv2.imwrite(os.path.join(QA, "diag_e2_color_vs_gray.png"),
            np.hstack([heat(img[crop], p_color[crop]), heat(g3[crop], p_gray[crop])]))

# 放大看被誤判字母的像素結構 (4x 最近鄰)
zy, zx = y1 + 40, x1 + 0
z = img[zy:zy + 60, zx:zx + 260]
cv2.imwrite(os.path.join(QA, "diag_zoom_test_latin.png"), cv2.resize(z, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST))

# ---------------- E3 ----------------
print("=== E3 訓練底稿 (空白掃描) 上的誤判 ===")
worst = None
for pdf in sorted(glob.glob("scanned_pairs/*.pdf")):
    d = fitz.open(pdf)
    n = d.page_count
    d.close()
    for i in range(0, n, max(1, n // 4)):
        im = render(pdf, i)
        p = probs(im)
        gr = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        c = gr < 80
        k = int((c & (p > 0.65)).sum())
        print(f"  {os.path.basename(pdf)[:22]:<22} p{i + 1:02d}  核心文字={int(c.sum()):6d}  誤判={k:5d} ({k / max(1, c.sum()) * 100:.2f}%)  全頁>0.65={(p > 0.65).mean() * 100:.2f}%")
        if worst is None or k > worst[0]:
            worst = (k, im, p, pdf, i)
k, im, p, pdf, i = worst
ys, xs = np.where((p > 0.65) & (cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) < 80))
if len(ys):
    cy, cx = int(np.median(ys)), int(np.median(xs))
    sl = (slice(max(0, cy - 200), cy + 200), slice(max(0, cx - 350), cx + 350))
    cv2.imwrite(os.path.join(QA, "diag_e3_train_scan_worst.png"), heat(im[sl], p[sl]))
    print(f"  最嚴重: {os.path.basename(pdf)} p{i + 1}")
