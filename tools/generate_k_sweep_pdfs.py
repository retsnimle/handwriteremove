#!/usr/bin/env python3
"""
生成不同 sharpen_k 值 (0.0, 0.15, 0.30, 0.45, 0.60) 的 PDF 與特寫對照圖供使用者決定。
測試對象：20251025101835564.pdf (直式 B4，印刷密集、手寫豐富)
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

def create_gaussian_kernel(kernel_size=5, sigma=1.0, channels=3, device="cuda"):
    x = torch.arange(kernel_size, dtype=torch.float32, device=device) - (kernel_size - 1) / 2.0
    gauss_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    gauss_1d = gauss_1d / gauss_1d.sum()
    gauss_2d = (gauss_1d[:, None] * gauss_1d[None, :]).view(1, 1, kernel_size, kernel_size)
    return gauss_2d.repeat(channels, 1, 1, 1)

K_GAUSS = create_gaussian_kernel(kernel_size=5, sigma=1.0, channels=3, device=dev)

def clean_page(raw_bgr, k_val, threshold=0.65):
    t_raw = torch.from_numpy(raw_bgr).to(dev, non_blocking=True)
    H, W, _ = t_raw.shape
    t_rgb = t_raw[..., [2, 1, 0]].permute(2, 0, 1).unsqueeze(0).float().div_(255.0)

    if k_val > 0:
        blurred = F.conv2d(t_rgb, K_GAUSS, padding=2, groups=3)
        t_sharp = torch.clamp((1.0 + k_val) * t_rgb - k_val * blurred, 0.0, 1.0)
    else:
        t_sharp = t_rgb

    t_norm = (t_sharp - MEAN) / STD
    pad_h = ((H + 31) // 32) * 32 - H
    pad_w = ((W + 31) // 32) * 32 - W
    t_pad = F.pad(t_norm, (0, pad_w, 0, pad_h), mode='constant', value=0.0) if (pad_h > 0 or pad_w > 0) else t_norm

    with torch.no_grad():
        logits = model(t_pad)
        probs = torch.sigmoid(logits)[:, 0:1, :H, :W]

    mask = (probs > threshold).float()
    eroded = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
    mask = F.max_pool2d(eroded, kernel_size=3, stride=1, padding=1)

    mask_2d = (mask[0, 0] > 0.5).unsqueeze(-1)
    paper_color = torch.tensor([252, 252, 252], dtype=torch.uint8, device=dev)
    cleaned_gpu = torch.where(mask_2d, paper_color, t_raw)

    return cleaned_gpu.cpu().numpy(), (mask[0, 0] > 0.5).byte().cpu().numpy()

def main():
    pdf_path = "20251025101835564.pdf"
    k_list = [0.0, 0.15, 0.30, 0.45, 0.60]
    out_dir = "output_cleaned/k_sweep"
    qa_dir = r"C:\Users\retsn\.gemini\antigravity\brain\f687af4b-740f-482b-a1f5-e84a253db0a3\qa"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(qa_dir, exist_ok=True)

    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    print(f"開始為 {pdf_path} 生成 5 組不同 K 值的完整 PDF...")

    # 先渲染所有頁面點陣圖
    pages_bgr = []
    for i in range(total_pages):
        page = doc[i]
        pix = page.get_pixmap(matrix=fitz.Matrix(150 / 72.0, 150 / 72.0))
        img = cv2.imdecode(np.frombuffer(pix.tobytes("png"), np.uint8), cv2.IMREAD_COLOR)
        pages_bgr.append(img)

    # 針對每個 K 值生成獨立 PDF
    for k in k_list:
        k_str = f"k_{int(k*100):02d}"
        pdf_out_path = os.path.join(out_dir, f"20251025101835564_{k_str}.pdf")
        out_doc = fitz.open()

        total_erased = 0
        total_pixels = 0

        for i in range(total_pages):
            raw = pages_bgr[i]
            cleaned, mask = clean_page(raw, k)
            total_erased += np.sum(mask > 0)
            total_pixels += mask.size

            _, buf = cv2.imencode(".jpg", cleaned, [cv2.IMWRITE_JPEG_QUALITY, 88])
            p = out_doc.new_page(width=doc[i].rect.width, height=doc[i].rect.height)
            p.insert_image(doc[i].rect, stream=buf.tobytes())

        out_doc.save(pdf_out_path, deflate=True, garbage=3)
        out_doc.close()
        ratio = (total_erased / total_pixels) * 100.0
        print(f"★ 已完成 K={k:.2f} ➔ {pdf_out_path} (平均擦除率: {ratio:.2f}%)")

    # 製作關鍵對照長條圖 (第一頁題3與下方手寫算式區、題6手繪幾何區)
    p1 = pages_bgr[0]
    H, W = p1.shape[:2]
    
    # 裁切兩個代表性區域：
    # 區塊1: 題3 (包含 (A)(B) DNA 印刷字 + 旁邊筆記)
    crop1_box = (int(H * 0.35), int(H * 0.35) + 360, 80, 720)
    # 區塊2: 題6、題7 (包含手繪立體幾何圖形、紅色筆劃與算式)
    crop2_box = (int(H * 0.65), int(H * 0.65) + 400, 80, 720)

    for box_idx, (y1, y2, x1, x2), box_name in [(1, crop1_box, "crop_q3_dna"), (2, crop2_box, "crop_q6_geometry")]:
        slices = []
        # 原圖
        raw_crop = p1[y1:y2, x1:x2].copy()
        cv2.putText(raw_crop, "Original", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        slices.append(raw_crop)

        for k in k_list:
            cln, _ = clean_page(p1, k)
            c_crop = cln[y1:y2, x1:x2].copy()
            cv2.putText(c_crop, f"K={k:.2f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            slices.append(c_crop)

        combined = np.hstack(slices)
        cv2.imwrite(os.path.join(qa_dir, f"k_sweep_compare_{box_name}.png"), combined)
        print(f"對照切片已儲存: k_sweep_compare_{box_name}.png")

    doc.close()
    print("全部不同 K 值的 PDF 與切片對照圖已全數生成完畢！")

if __name__ == "__main__":
    main()
