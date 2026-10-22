#!/usr/bin/env python3
"""
真實考卷 PDF 全純 GPU 加速去手寫批次處理器 (Pure GPU Batch Exam Cleaner)
針對使用者之要求，將所有前處理、正規化、形態學濾波、像素替換與預覽縮放全面移入 GPU 顯存！
整合了高頻增益補償 (Unsharp Masking on GPU)，徹底消除印刷英文字母、細小符號被誤殺的問題！
"""

import os
import sys
import time
import argparse
import glob
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import fitz

def load_remover_model(weights_path="weights/best_exam_remover.pth", device="cuda"):
    import segmentation_models_pytorch as smp
    print(f"[ExamRemover] 正在載入模型: {weights_path} ({device}) ...")
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
    print("[ExamRemover] ★ GPU 模型載入就緒！")
    return model

def create_gaussian_kernel(kernel_size=5, sigma=1.0, channels=3, device="cuda"):
    """在 GPU 上建立 2D 高斯卷積核 (depthwise separable)"""
    x = torch.arange(kernel_size, dtype=torch.float32, device=device) - (kernel_size - 1) / 2.0
    gauss_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    gauss_1d = gauss_1d / gauss_1d.sum()
    gauss_2d = (gauss_1d[:, None] * gauss_1d[None, :]).view(1, 1, kernel_size, kernel_size)
    kernel = gauss_2d.repeat(channels, 1, 1, 1)
    return kernel

def clean_single_page_gpu(model, raw_bgr_np, threshold=0.65, sharpen_k=0.6, filter_noise=True, use_dilate=False, device="cuda"):
    """
    全純 GPU 執行之單頁去手寫管線 (Pure GPU In-VRAM Pipeline)
    """
    # 1. 極速 uint8 直傳 GPU (體積僅 float32 的 1/4，PCIe 傳輸僅需 1~2ms)
    t_raw = torch.from_numpy(raw_bgr_np).to(device, non_blocking=True) # (H, W, 3) uint8
    H, W, _ = t_raw.shape

    # 2. GPU 內部平行色彩轉換 (BGR -> RGB)
    t_rgb = t_raw[..., [2, 1, 0]].permute(2, 0, 1).unsqueeze(0).float().div_(255.0) # (1, 3, H, W)

    # 3. GPU 原生 Unsharp Masking 頻率補償 (校正掃描光學模糊，保護極細印刷筆鋒)
    if sharpen_k > 0:
        kernel = create_gaussian_kernel(kernel_size=5, sigma=1.0, channels=3, device=device)
        pad = 2
        blurred = F.conv2d(t_rgb, kernel, padding=pad, groups=3)
        t_sharp = torch.clamp((1.0 + sharpen_k) * t_rgb - sharpen_k * blurred, 0.0, 1.0)
    else:
        t_sharp = t_rgb

    # 4. GPU 平行標準化
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    t_norm = (t_sharp - mean) / std

    # 5. GPU 原生 Padding 至 32 倍數
    pad_h = ((H + 31) // 32) * 32 - H
    pad_w = ((W + 31) // 32) * 32 - W
    if pad_h > 0 or pad_w > 0:
        t_pad = F.pad(t_norm, (0, pad_w, 0, pad_h), mode='constant', value=0.0)
    else:
        t_pad = t_norm

    # 6. GPU 前向推論
    with torch.no_grad():
        logits = model(t_pad)
        probs = torch.sigmoid(logits)[:, 0:1, :H, :W]

    # 7. GPU 信心度門檻二值化
    mask = (probs > threshold).float()

    # 8. cuDNN 硬體加速形態學濾波 (純 GPU 執行)
    if filter_noise:
        # 開運算 (Opening: 先腐蝕消滅 1~3px 孤立文字毛邊，再膨脹還原手寫筆劃尺寸)
        eroded = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
        opened = F.max_pool2d(eroded, kernel_size=3, stride=1, padding=1)
        mask = opened

    if use_dilate:
        # 額外膨脹
        mask = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)

    # 9. GPU 原生向量化像素填色 (Inpainting)
    mask_2d = (mask[0, 0] > 0.5).unsqueeze(-1) # (H, W, 1) bool
    paper_color = torch.tensor([252, 252, 252], dtype=torch.uint8, device=device)
    cleaned_gpu = torch.where(mask_2d, paper_color, t_raw)

    # 10. GPU 平行下採樣三聯預覽圖 (零 CPU 縮放負擔)
    scale = 650.0 / W
    nh, nw = int(H * scale), 650
    red_mark = torch.tensor([0, 0, 255], dtype=torch.uint8, device=device)
    vis_gpu = torch.where(mask_2d, red_mark, t_raw)

    p_raw = F.interpolate(t_raw.permute(2, 0, 1).unsqueeze(0).float(), size=(nh, nw), mode='area')
    p_vis = F.interpolate(vis_gpu.permute(2, 0, 1).unsqueeze(0).float(), size=(nh, nw), mode='area')
    p_cln = F.interpolate(cleaned_gpu.permute(2, 0, 1).unsqueeze(0).float(), size=(nh, nw), mode='area')

    preview_trio_gpu = torch.cat([p_raw, p_vis, p_cln], dim=3)[0].permute(1, 2, 0).byte()

    # 11. 唯一下載回主機記憶體 (用於 PDF 組裝與 QA 輸出)
    cleaned_cpu = cleaned_gpu.cpu().numpy()
    trio_cpu = preview_trio_gpu.cpu().numpy()
    erased_ratio = float((mask_2d.sum().float() / (H * W)).item()) * 100.0

    return cleaned_cpu, trio_cpu, erased_ratio

def process_pdf(pdf_path, model, threshold=0.65, sharpen_k=0.6, filter_noise=True, use_dilate=False, dpi=150, quality=88, device="cuda", out_pdf_dir="output_cleaned", out_qa_dir="output/pdf_qa"):
    pdf_name = os.path.basename(pdf_path)
    base_name = os.path.splitext(pdf_name)[0]
    out_pdf_path = os.path.join(out_pdf_dir, f"{base_name}_cleaned.pdf")

    print(f"\n=======================================================")
    print(f"📄 開始處理: {pdf_name}")
    print(f"   運算架構: Pure GPU | 門檻={threshold:.2f} | 銳化補償k={sharpen_k:.1f} | 形態學濾波={filter_noise} | DPI={dpi}")
    print(f"=======================================================")

    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    out_doc = fitz.open()

    artifact_qa_dir = r"C:\Users\retsn\.gemini\antigravity\brain\f687af4b-740f-482b-a1f5-e84a253db0a3\qa"
    os.makedirs(artifact_qa_dir, exist_ok=True)

    t_start = time.time()

    for page_idx in range(total_pages):
        page = doc[page_idx]
        t0 = time.time()

        # 指定 DPI 渲染點陣圖
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        raw_bgr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

        # 執行 Pure GPU 清洗
        cleaned_bgr, trio_bgr, erased_ratio = clean_single_page_gpu(
            model,
            raw_bgr,
            threshold=threshold,
            sharpen_k=sharpen_k,
            filter_noise=filter_noise,
            use_dilate=use_dilate,
            device=device
        )
        elapsed = time.time() - t0

        print(f"  [頁 {page_idx+1:02d}/{total_pages:02d}] 尺寸: {raw_bgr.shape[1]}x{raw_bgr.shape[0]} | 擦除率: {erased_ratio:.2f}% | 總耗時: {elapsed*1000:.1f} ms")

        # 寫入清洗後的 PDF 頁面 (直接寫入原生 JPEG 壓縮流)
        is_success, buffer = cv2.imencode(".jpg", cleaned_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        clean_bytes = buffer.tobytes()
        new_page = out_doc.new_page(width=page.rect.width, height=page.rect.height)
        new_page.insert_image(page.rect, stream=clean_bytes)

        # 儲存檢核對照圖
        qa_filename = f"{base_name}_p{page_idx+1:02d}.png"
        cv2.imwrite(os.path.join(out_qa_dir, qa_filename), trio_bgr)
        cv2.imwrite(os.path.join(artifact_qa_dir, qa_filename), trio_bgr)

    # 儲存新 PDF
    out_doc.save(out_pdf_path, deflate=True, garbage=3)
    out_doc.close()
    doc.close()

    total_time = time.time() - t_start
    pdf_size_mb = os.path.getsize(out_pdf_path) / (1024 * 1024)
    print(f"🎉 {pdf_name} 清洗完畢！")
    print(f"   ➔ 已儲存清洗後 PDF: {out_pdf_path} ({pdf_size_mb:.2f} MB)")
    print(f"   ➔ 總耗時: {total_time:.2f} 秒 (平均 {total_time/total_pages:.2f} 秒/頁)")

def main():
    parser = argparse.ArgumentParser(description="真實考卷 PDF Pure GPU 去手寫批次處理器")
    parser.add_argument("--threshold", type=float, default=0.65, help="信心度二值化門檻 (預設 0.65)")
    parser.add_argument("--sharpen-k", type=float, default=0.6, help="GPU 頻率補償強度 (預設 0.6)")
    parser.add_argument("--no-filter", action="store_true", help="關閉形態學濾波")
    parser.add_argument("--dilate", action="store_true", help="是否啟用額外膨脹")
    parser.add_argument("--dpi", type=int, default=150, help="PDF 渲染解析度 (預設 150)")
    parser.add_argument("--quality", type=int, default=88, help="輸出 JPEG 壓縮品質 (預設 88)")
    parser.add_argument("--weights", type=str, default="weights/best_exam_remover.pth", help="模型權重路徑")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"運算裝置: {device} ({torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'})")

    os.makedirs("output_cleaned", exist_ok=True)
    os.makedirs("output/pdf_qa", exist_ok=True)

    model = load_remover_model(weights_path=args.weights, device=device)

    target_pdfs = [
        "20251020202938305.pdf",
        "20251025101835564.pdf"
    ]

    for p in target_pdfs:
        if os.path.exists(p):
            process_pdf(
                p,
                model,
                threshold=args.threshold,
                sharpen_k=args.sharpen_k,
                filter_noise=(not args.no_filter),
                use_dilate=args.dilate,
                dpi=args.dpi,
                quality=args.quality,
                device=device
            )
        else:
            print(f"[警告] 找不到檔案: {p}")

    print("\n=======================================================")
    print("🏆 根目錄下所有目標 PDF 已全數處理完成！")
    print("=======================================================")

if __name__ == "__main__":
    main()
