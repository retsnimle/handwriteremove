#!/usr/bin/env python3
"""
隔離「重新取樣質地」：幾何完全不變，只改像素質地。
  R1 放大2倍再縮回 (INTER_LINEAR)：形狀不變，只有內插造成的邊緣柔化
  R2 輕微高斯模糊 σ=0.6
  R3 銳化
在訓練底稿 (原本誤判 0%) 與真實考卷上各測一次。
"""
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
def fp(bgr, core):
    t = torch.from_numpy(np.ascontiguousarray(bgr)).to(dev)
    H, W, _ = t.shape
    x = t[..., [2, 1, 0]].permute(2, 0, 1).unsqueeze(0).float() / 255.0
    x = F.pad((x - MEAN) / STD, (0, (32 - W % 32) % 32, 0, (32 - H % 32) % 32))
    p = torch.sigmoid(model(x))[0, 0, :H, :W].cpu().numpy()
    return (core & (p > 0.65)).sum() / max(1, core.sum()) * 100


def updown(im):
    h, w = im.shape[:2]
    return cv2.resize(cv2.resize(im, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR), (w, h), interpolation=cv2.INTER_LINEAR)


def sharpen(im):
    bl = cv2.GaussianBlur(im, (0, 0), 1.0)
    return cv2.addWeighted(im, 1.6, bl, -0.6, 0)


print(f"{'頁面':<26}{'原樣':>8}{'內插柔化':>10}{'模糊0.6':>10}{'銳化':>8}")
for pdf, i in [("scanned_pairs/22-114學測自然試題定稿 掃描.pdf", 5), ("scanned_pairs/3561749-題目卷 掃描.pdf", 1),
               ("20251025101835564.pdf", 0), ("20251025101835564.pdf", 6), ("20251020202938305.pdf", 0)]:
    im = render(pdf, i)
    core = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) < 80   # 以原圖定義印刷像素，避免模糊改變分母
    r = [fp(im, core), fp(updown(im), core), fp(cv2.GaussianBlur(im, (0, 0), 0.6), core), fp(sharpen(im), core)]
    name = pdf.split("/")[-1][:14] + f" p{i + 1}"
    print(f"{name:<26}{r[0]:7.2f}%{r[1]:9.2f}%{r[2]:9.2f}%{r[3]:7.2f}%")
