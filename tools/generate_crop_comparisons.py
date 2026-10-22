#!/usr/bin/env python3
"""
檢視 Page 1 與 Page 2 的誤殺細節與視覺對比切片
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

def filter_small_components(binary_mask, min_area=8):
    if min_area <= 0:
        return binary_mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    clean_mask = np.zeros_like(binary_mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            clean_mask[labels == i] = 255
    return clean_mask

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_remover_model(device=device)

    pdf_path = "20251025101835564.pdf"
    doc = fitz.open(pdf_path)

    qa_dir = r"C:\Users\retsn\.gemini\antigravity\brain\f687af4b-740f-482b-a1f5-e84a253db0a3\qa"
    os.makedirs(qa_dir, exist_ok=True)

    for page_idx in [0, 1]:
        page = doc[page_idx]
        mat = fitz.Matrix(150 / 72.0, 150 / 72.0)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        raw_bgr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
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
        tensor = tensor.to(device)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.sigmoid(logits)[0, 0]
        if dh > 0 or dw > 0:
            probs = probs[:H, :W]
        probs_np = probs.cpu().numpy()

        # 方案 A: 原始基準 (0.45 + 3x3 Dilate)
        mask_base = (probs_np > 0.45).astype(np.uint8) * 255
        mask_base = cv2.dilate(mask_base, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        cln_base = raw_bgr.copy()
        cln_base[mask_base > 0] = [252, 252, 252]

        # 方案 B: 微調信心度 (0.65 + 無膨脹 + 微噪點過濾 min_area=8)
        mask_tuned = (probs_np > 0.65).astype(np.uint8) * 255
        mask_tuned = filter_small_components(mask_tuned, min_area=8)
        cln_tuned = raw_bgr.copy()
        cln_tuned[mask_tuned > 0] = [252, 252, 252]

        # 方案 C: 高門檻極致保護 (0.75 + 無膨脹 + min_area=12)
        mask_ultra = (probs_np > 0.75).astype(np.uint8) * 255
        mask_ultra = filter_small_components(mask_ultra, min_area=12)
        cln_ultra = raw_bgr.copy()
        cln_ultra[mask_ultra > 0] = [252, 252, 252]

        # 尋找手寫與文字重疊密度最高、或文字密集的區域進行特寫裁切 (400x500)
        # 我們找 mask_tuned 覆蓋度高的區域，或是中上方題目區
        # 產生幾處代表性局部裁切
        crops = [
            ("header_or_q1", 100, 500, 100, 700),
            ("middle_dense", int(H * 0.4), int(H * 0.4) + 400, 100, 700),
            ("bottom_notes", int(H * 0.7), int(H * 0.7) + 400, 100, 700),
        ]

        for crop_name, y1, y2, x1, x2 in crops:
            y2 = min(y2, H)
            x2 = min(x2, W)
            c_raw = raw_bgr[y1:y2, x1:x2]
            c_base = cln_base[y1:y2, x1:x2]
            c_tuned = cln_tuned[y1:y2, x1:x2]
            c_ultra = cln_ultra[y1:y2, x1:x2]

            # 加上標題
            def add_title(im, title):
                vis = im.copy()
                cv2.putText(vis, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)
                return vis

            comparison = np.hstack([
                add_title(c_raw, "Original (Raw)"),
                add_title(c_base, "Base (0.45+Dilate)"),
                add_title(c_tuned, "Tuned (0.65+NoDil)"),
                add_title(c_ultra, "Ultra (0.75+NoDil)")
            ])

            save_path = os.path.join(qa_dir, f"compare_p{page_idx+1:02d}_{crop_name}.png")
            cv2.imwrite(save_path, comparison)
            print(f"Saved: {save_path}")

    doc.close()
    print("All crop comparisons generated successfully!")

if __name__ == "__main__":
    main()
