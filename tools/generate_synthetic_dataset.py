#!/usr/bin/env python3
"""
試卷手寫筆跡自動合成引擎 (Synthetic Dataset Generator)

功能：
  1. 載入 clean_pdfs/ 的純淨考卷頁面 (353 頁)。
  2. 載入 assets/handwriting/ 下提取的 10,205 個真實手寫字母與 251 個草稿印章。
  3. 模擬真實考試情境：
     - 在作答括號區隨機蓋上手寫字母 (模擬字體衝出括號、壓到題號與邊界)。
     - 在空白區/題幹周圍蓋上演算草稿、紅筆勾叉、刪除線。
     - 施加色調、旋轉、微幅尺度擾動。
  4. 產出精確像素級成對資料：
     - images/ : 合成後的作答試卷 (輸入圖)
     - masks/  : 精確的手寫真值遮罩 (Ground Truth Mask)
     - targets/: 原始無瑕疵試卷 (訓練修復目標)
"""

import os
import sys
import glob
import random
import time
import cv2
import numpy as np
import fitz
import zipfile

def load_stamps(letter_dir, scratch_dir):
    letter_paths = glob.glob(os.path.join(letter_dir, "*.png"))
    scratch_paths = glob.glob(os.path.join(scratch_dir, "*.png"))
    print(f"[Synthesizer] 載入 {len(letter_paths)} 個字母印章，{len(scratch_paths)} 個草稿印章。")
    return letter_paths, scratch_paths

def overlay_stamp(bg_bgr, bg_mask, stamp_bgra, x, y, scale=1.0, angle=0.0, color_tint=None):
    """
    將透明印章以指定縮放、旋轉與顏色擾動貼到底圖上，並同步更新真值手寫遮罩 (Mask)
    """
    sh, sw = stamp_bgra.shape[:2]
    new_w = max(4, int(sw * scale))
    new_h = max(4, int(sh * scale))
    resized = cv2.resize(stamp_bgra, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    if abs(angle) > 1.0:
        M = cv2.getRotationMatrix2D((new_w / 2, new_h / 2), angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        bound_w = int((new_h * sin) + (new_w * cos))
        bound_h = int((new_h * cos) + (new_w * sin))
        M[0, 2] += (bound_w / 2) - new_w / 2
        M[1, 2] += (bound_h / 2) - new_h / 2
        resized = cv2.warpAffine(resized, M, (bound_w, bound_h), flags=cv2.INTER_LINEAR, borderValue=(0,0,0,0))
        new_w, new_h = bound_w, bound_h

    # 顏色擾動 (如染成微藍、微紅、或鉛筆灰)
    bgr = resized[:, :, :3].astype(float)
    if color_tint is not None:
        tb, tg, tr = color_tint
        bgr[:, :, 0] = np.clip(bgr[:, :, 0] * tb, 0, 255)
        bgr[:, :, 1] = np.clip(bgr[:, :, 1] * tg, 0, 255)
        bgr[:, :, 2] = np.clip(bgr[:, :, 2] * tr, 0, 255)
    bgr = bgr.astype(np.uint8)
    
    alpha = resized[:, :, 3].astype(float) / 255.0
    
    # 邊界裁切處理
    H, W = bg_bgr.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + new_w), min(H, y + new_h)
    
    if x2 <= x1 or y2 <= y1:
        return
        
    sx1, sy1 = x1 - x, y1 - y
    sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)
    
    sub_alpha = alpha[sy1:sy2, sx1:sx2, None]
    sub_bgr = bgr[sy1:sy2, sx1:sx2]
    
    # 正片疊底 (Multiply blending) 模擬真實墨水吸收
    bg_roi = bg_bgr[y1:y2, x1:x2].astype(float)
    blended = (bg_roi * sub_bgr) / 255.0
    bg_bgr[y1:y2, x1:x2] = np.clip(
        sub_alpha * blended + (1.0 - sub_alpha) * bg_roi, 0, 255
    ).astype(np.uint8)
    
    # 同步更新手寫真值遮罩 (手寫不透明度 > 0.15 處標為 255)
    mask_patch = (alpha[sy1:sy2, sx1:sx2] > 0.15).astype(np.uint8) * 255
    bg_mask[y1:y2, x1:x2] = np.maximum(bg_mask[y1:y2, x1:x2], mask_patch)

def synthesize_exam_page(page, clean_bgr, letter_stamps, scratch_stamps, scale_pt_to_px):
    H, W = clean_bgr.shape[:2]
    synth_img = clean_bgr.copy()
    synth_mask = np.zeros((H, W), dtype=np.uint8)
    
    # 1. 利用 PDF 向量坐標尋找括號作答區
    bracket_rects = []
    for b_char in ['(', '（', '〔', '【']:
        rects = page.search_for(b_char)
        bracket_rects.extend(rects)
        
    # 2. 在括號位置填入答案 (模擬 60% ~ 90% 的題目有作答)
    random.shuffle(bracket_rects)
    num_to_answer = int(len(bracket_rects) * random.uniform(0.6, 0.95))
    
    for r in bracket_rects[:num_to_answer]:
        stamp_path = random.choice(letter_stamps)
        stamp = cv2.imread(stamp_path, cv2.IMREAD_UNCHANGED)
        if stamp is None or stamp.shape[2] < 4:
            continue
            
        # 轉換 PDF 坐標到影像坐標
        base_x = int(r.x0 * scale_pt_to_px)
        base_y = int(r.y0 * scale_pt_to_px)
        
        # 模擬人類手寫偏移：有的剛好在括號內，有的往右衝出，有的往左壓到括號/題號
        offset_x = random.randint(-4, 18)
        offset_y = random.randint(-12, 6)
        x = base_x + offset_x
        y = base_y + offset_y
        
        scale = random.uniform(0.85, 1.35)
        angle = random.uniform(-12, 12)
        
        # 墨水色調 (黑筆 65%, 藍筆 25%, 鉛筆 10%)
        mode = random.random()
        if mode < 0.25:
            tint = (1.4, 0.9, 0.6)  # 藍筆
        elif mode < 0.35:
            tint = (1.1, 1.1, 1.1)  # 鉛筆淡灰
        else:
            tint = (1.0, 1.0, 1.0)  # 純黑/深黑
            
        overlay_stamp(synth_img, synth_mask, stamp, x, y, scale=scale, angle=angle, color_tint=tint)
        
    # 如果頁面括號極少 (如全是非選題或作圖題)，補一些隨機欄位答案
    if len(bracket_rects) < 5:
        is_two_cols = (W > 1800)
        for _ in range(random.randint(5, 15)):
            stamp_path = random.choice(letter_stamps)
            stamp = cv2.imread(stamp_path, cv2.IMREAD_UNCHANGED)
            if stamp is None or stamp.shape[2] < 4:
                continue
            if is_two_cols:
                col = random.choice(['left', 'right'])
                x = random.randint(int(W * 0.04), int(W * 0.10)) if col == 'left' else random.randint(int(W * 0.54), int(W * 0.60))
            else:
                x = random.randint(int(W * 0.03), int(W * 0.10))
            y = random.randint(int(H * 0.08), int(H * 0.92))
            overlay_stamp(synth_img, synth_mask, stamp, x, y, scale=random.uniform(0.8, 1.3), angle=random.uniform(-15, 15))

    # 3. 模擬空白處演算推導草稿 (約 2 ~ 6 塊)
    num_scratches = random.randint(2, 6)
    for _ in range(num_scratches):
        stamp_path = random.choice(scratch_stamps)
        stamp = cv2.imread(stamp_path, cv2.IMREAD_UNCHANGED)
        if stamp is None or stamp.shape[2] < 4:
            continue
            
        sh, sw = stamp.shape[:2]
        best_x = random.randint(50, max(51, W - sw - 50))
        best_y = random.randint(50, max(51, H - sh - 50))
        
        scale = random.uniform(0.75, 1.25)
        angle = random.uniform(-8, 8)
        tint = random.choice([(1.0, 1.0, 1.0), (1.35, 0.9, 0.6), (1.1, 1.1, 1.1)])
        overlay_stamp(synth_img, synth_mask, stamp, best_x, best_y, scale=scale, angle=angle, color_tint=tint)
        
    # 4. 模擬手寫圈選選項 (A/B/C/D) 或紅筆批改勾叉 (40% 機率)
    if random.random() < 0.45:
        num_marks = random.randint(1, 4)
        for _ in range(num_marks):
            mx = random.randint(50, W - 100)
            my = random.randint(50, H - 100)
            # 紅色或藍色批改標記
            is_red = random.random() < 0.7
            color = (random.randint(15, 45), random.randint(15, 55), random.randint(180, 245)) if is_red else (random.randint(180, 230), random.randint(100, 150), 30)
            thickness = random.randint(2, 4)
            if random.random() < 0.5:
                # 勾
                pts = np.array([[mx, my], [mx + 15, my + 30], [mx + 55, my - 25]], np.int32)
                cv2.polylines(synth_img, [pts], False, color, thickness, cv2.LINE_AA)
                cv2.polylines(synth_mask, [pts], False, 255, thickness + 2, cv2.LINE_AA)
            else:
                # 叉
                cv2.line(synth_img, (mx, my), (mx + 35, my + 35), color, thickness, cv2.LINE_AA)
                cv2.line(synth_img, (mx + 35, my), (mx, my + 35), color, thickness, cv2.LINE_AA)
                cv2.line(synth_mask, (mx, my), (mx + 35, my + 35), 255, thickness + 2, cv2.LINE_AA)
                cv2.line(synth_mask, (mx + 35, my), (mx, my + 35), 255, thickness + 2, cv2.LINE_AA)
                
    return synth_img, synth_mask

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=1000, help="生成總樣本數量")
    parser.add_argument("--test_only", action="store_true", help="只生成 10 張供人工檢查檢驗")
    args = parser.parse_args()
    
    num_to_gen = 10 if args.test_only else args.num_samples
    
    out_dir = "dataset/synthetic_exam"
    os.makedirs(f"{out_dir}/images", exist_ok=True)
    os.makedirs(f"{out_dir}/masks", exist_ok=True)
    os.makedirs(f"{out_dir}/targets", exist_ok=True)
    
    letters, scratches = load_stamps("assets/handwriting/letters", "assets/handwriting/scratches")
    clean_pdfs = sorted(glob.glob("clean_pdfs/*.pdf"))
    
    print(f"[Synthesizer] 準備生成 {num_to_gen} 組成對資料...")
    t0 = time.time()
    
    # 預先收集純淨考卷的所有頁面引用 (pdf_idx, page_idx)
    page_catalog = []
    for p_idx, pdf_path in enumerate(clean_pdfs):
        doc = fitz.open(pdf_path)
        for page_num in range(doc.page_count):
            page_catalog.append((pdf_path, page_num))
        doc.close()
        
    print(f"[Synthesizer] 共有 {len(page_catalog)} 頁純淨試卷底圖可供採樣。")
    
    scale_pt_to_px = 150 / 72.0
    mat = fitz.Matrix(scale_pt_to_px, scale_pt_to_px)
    
    for i in range(num_to_gen):
        pdf_path, page_num = random.choice(page_catalog)
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_num)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_bytes = pix.tobytes("png")
        nparr = np.frombuffer(img_bytes, np.uint8)
        clean_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        synth_img, synth_mask = synthesize_exam_page(page, clean_bgr, letters, scratches, scale_pt_to_px)
        doc.close()
        
        sample_id = f"sample_{i+1:05d}"
        cv2.imwrite(f"{out_dir}/images/{sample_id}.png", synth_img)
        cv2.imwrite(f"{out_dir}/masks/{sample_id}.png", synth_mask)
        cv2.imwrite(f"{out_dir}/targets/{sample_id}.png", clean_bgr)
        
        if (i + 1) % 50 == 0 or (i + 1) == num_to_gen:
            print(f"  已生成 {i+1} / {num_to_gen} 組 (耗時: {time.time() - t0:.1f}s)")
            
    print("=" * 60)
    print(f"[完成] 成功生成 {num_to_gen} 組訓練資料！儲存於: {out_dir}")
    print("=" * 60)
    
    # 自動打包為 exam_dataset.zip
    zip_path = "exam_dataset.zip"
    print(f"[打包] 正在將資料集壓縮為 {zip_path}，便於上傳 Colab...")
    t_zip0 = time.time()
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        for folder in ['images', 'masks']:
            f_paths = glob.glob(f"{out_dir}/{folder}/*.png")
            for f in f_paths:
                arcname = f"{folder}/{os.path.basename(f)}"
                z.write(f, arcname)
    zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"[打包完成] {zip_path} 大小: {zip_size_mb:.2f} MB (耗時: {time.time() - t_zip0:.1f}s)")

if __name__ == "__main__":
    main()
