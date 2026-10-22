#!/usr/bin/env python3
"""
記憶化檢驗：模型在訓練底稿上誤判=0。若只是把同一頁稍微縮放/旋轉就出現大量誤判，
代表模型背下了那 106 頁的長相，而不是學會「印刷 vs 手寫」。
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


def render(pdf, i, dpi=150):
    d = fitz.open(pdf)
    pix = d[i].get_pixmap(dpi=dpi)
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
    return cv2.warpAffine(im, M, (w, h), borderValue=(255, 255, 255))


pages = [("scanned_pairs/22-114學測自然試題定稿 掃描.pdf", 5), ("scanned_pairs/3561749-題目卷 掃描.pdf", 1),
         ("scanned_pairs/20261021202624663.pdf", 34)]
print(f"{'頁面':<28}{'原樣':>8}{'DPI165':>8}{'DPI135':>8}{'旋轉1.5°':>10}{'JPEG q60':>10}")
for pdf, i in pages:
    base = render(pdf, i)
    _, enc = cv2.imencode(".jpg", base, [cv2.IMWRITE_JPEG_QUALITY, 60])
    r = [fp(base), fp(render(pdf, i, 165)), fp(render(pdf, i, 135)), fp(rot(base, 1.5)),
         fp(cv2.imdecode(enc, cv2.IMREAD_COLOR))]
    name = pdf.split("/")[-1][:14] + f" p{i + 1}"
    print(f"{name:<28}" + "".join(f"{v:7.2f}%" for v in r[:3]) + f"{r[3]:9.2f}%{r[4]:9.2f}%")
