#!/usr/bin/env python3
"""銳化能壓低印刷誤殺，但會不會也讓手寫漏抓？以紅/藍筆(高飽和)像素作為手寫代理指標。"""
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


def render(pdf, i):
    d = fitz.open(pdf)
    pix = d[i].get_pixmap(dpi=150)
    im = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3].copy()
    d.close()
    return cv2.cvtColor(im, cv2.COLOR_RGB2BGR)


@torch.no_grad()
def prob(bgr):
    t = torch.from_numpy(np.ascontiguousarray(bgr)).to(dev)
    H, W, _ = t.shape
    x = t[..., [2, 1, 0]].permute(2, 0, 1).unsqueeze(0).float() / 255.0
    x = F.pad((x - MEAN) / STD, (0, (32 - W % 32) % 32, 0, (32 - H % 32) % 32))
    return torch.sigmoid(model(x))[0, 0, :H, :W].cpu().numpy()


def sharpen(im, k):
    bl = cv2.GaussianBlur(im, (0, 0), 1.0)
    return cv2.addWeighted(im, 1 + k, bl, -k, 0)


for pdf, i in [("20251025101835564.pdf", 0), ("20251025101835564.pdf", 6), ("20251020202938305.pdf", 0)]:
    im = render(pdf, i)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    core = gray < 80
    ink = (hsv[:, :, 1] > 90) & (hsv[:, :, 2] < 230)  # 彩色筆跡
    print(f"{pdf[:14]} p{i + 1}  (印刷 {core.sum()} px / 彩色筆跡 {ink.sum()} px)")
    for k in [0.0, 0.3, 0.6]:
        p = prob(sharpen(im, k) if k else im) > 0.65
        print(f"   銳化 k={k:.1f}: 印刷誤殺 {(core & p).sum() / core.sum() * 100:5.2f}%   彩色筆跡抓到 {(ink & p).sum() / max(1, ink.sum()) * 100:5.2f}%   全頁擦除 {p.mean() * 100:4.2f}%")
        if pdf.startswith("20251025") and i == 0:
            H = im.shape[0]
            sl = (slice(int(H * 0.4), int(H * 0.4) + 400), slice(100, 700))
            out = im.copy()
            out[p] = 252
            cv2.imwrite(os.path.join(QA, f"diag_sharpen_k{int(k * 10)}.png"), out[sl])
