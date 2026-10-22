#!/usr/bin/env python3
"""驗證：真實考卷是否微幅歪斜？反向旋轉校正後，誤殺是否下降？"""
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


def render(pdf, i):
    d = fitz.open(pdf)
    pix = d[i].get_pixmap(dpi=150)
    im = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3].copy()
    d.close()
    return cv2.cvtColor(im, cv2.COLOR_RGB2BGR)


@torch.no_grad()
def fp(bgr):
    t = torch.from_numpy(bgr).to(dev)
    H, W, _ = t.shape
    x = t[..., [2, 1, 0]].permute(2, 0, 1).unsqueeze(0).float() / 255.0
    x = F.pad((x - MEAN) / STD, (0, (32 - W % 32) % 32, 0, (32 - H % 32) % 32))
    p = torch.sigmoid(model(x))[0, 0, :H, :W].cpu().numpy()
    core = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) < 80
    return (core & (p > 0.65)).sum() / max(1, core.sum()) * 100


def rot(im, a):
    h, w = im.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
    return cv2.warpAffine(im, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))


def skew_angle(bgr):
    """投影輪廓法：旋轉後水平投影變異數最大者 = 文字行最水平。"""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    b = (g < 128).astype(np.float32)
    b = cv2.resize(b, None, fx=0.5, fy=0.5)
    best = None
    for a in np.arange(-2.0, 2.01, 0.05):
        h, w = b.shape
        M = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
        prof = cv2.warpAffine(b, M, (w, h)).sum(1)
        v = prof.var()
        if best is None or v > best[0]:
            best = (v, a)
    return best[1]


for pdf, pages in [("20251025101835564.pdf", [0, 2, 6]), ("20251020202938305.pdf", [0, 2])]:
    for i in pages:
        im = render(pdf, i)
        a = skew_angle(im)
        print(f"{pdf[:14]} p{i + 1}: 估計歪斜 {a:+.2f}°  原樣誤殺 {fp(im):5.2f}%  校正後誤殺 {fp(rot(im, a)):5.2f}%")
