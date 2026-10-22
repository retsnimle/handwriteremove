#!/usr/bin/env python3
"""
高解析度考卷 Patch 資料集本地預生成工具 (Patch Dataset Generator)

特點：
  1. 300 DPI 原生尺度合成，保持手寫筆觸與印刷字體 100% 真實像素（絕不縮小模糊）。
  2. 隨機採樣 1024x1024 原生 Patch：
     - 包含題號填答、選項圈選、題幹刪除塗改、密集演算推導。
     - 智慧權重：80% 採樣包含手寫墨水與題目的區域，20% 採樣純淨留白與圖表區。
  3. 輕量高效儲存：
     - inputs/: 1024x1024 JPG (品質 95, 每張約 120KB)
     - masks/: 1024x1024 PNG (二值遮罩, 每張約 20KB)
     - targets/: 1024x1024 JPG (品質 95, 乾淨修復目標)
  4. 多進程加速，快速產出 600 ~ 1000 組高質量訓練對。
"""

import os
import sys
import time
import argparse
import random
import cv2
import numpy as np

# 引用合成模組
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from synthetic_engine_v2 import ExamSynthesizer

def generate_patch_dataset(num_samples=600, out_dir="dataset_patches", dpi=300, patch_size=1024):
    in_dir = os.path.join(out_dir, "images")
    mask_dir = os.path.join(out_dir, "masks")
    tgt_dir = os.path.join(out_dir, "targets")
    os.makedirs(in_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(tgt_dir, exist_ok=True)

    print(f"=== 開始本地高解析度 Patch 資料集生成 ===")
    print(f"目標樣本數: {num_samples} | 解析度: {dpi} DPI | Patch 邊長: {patch_size}x{patch_size}")
    
    engine = ExamSynthesizer(dpi=dpi)
    sc = engine.scale_pt_to_px
    base_scale = (dpi / 150.0)

    t0 = time.time()
    generated = 0

    # 頁面快取，避免重複渲染 PDF
    rendered_cache = {}

    while generated < num_samples:
        # 1. 選取考卷並渲染
        item = random.choice(engine.catalog)
        key = (item["pdf"], item["page"])
        if key not in rendered_cache:
            if len(rendered_cache) > 25:
                rendered_cache.pop(next(iter(rendered_cache)))
            rendered_cache[key] = engine.render_clean_page(item["pdf"], item["page"])
            
        raw_clean = rendered_cache[key]
        target_bgr = engine.degrade_clean_print(raw_clean)
        synth_img = target_bgr.copy()
        H, W = synth_img.shape[:2]
        synth_mask = np.zeros((H, W), dtype=np.uint8)

        # 2. 注入高密度作答 (85% ~ 100%)
        brackets = item.get("brackets", [])
        if brackets and engine.cat_letters:
            num_ans = max(1, int(len(brackets) * random.uniform(0.85, 1.0)))
            for box in random.sample(brackets, min(num_ans, len(brackets))):
                fname = random.choice(engine.cat_letters)
                stamp = cv2.imread(os.path.join(engine.marks_dir, fname), cv2.IMREAD_UNCHANGED)
                if stamp is None: continue
                bx = int(box[0] * sc) + random.randint(-8, 12)
                by = int(box[1] * sc) + random.randint(-10, 10)
                engine.overlay_rgba_stamp(synth_img, synth_mask, stamp, bx, by, base_scale * random.uniform(0.9, 1.25), random.uniform(-10, 10))

        # 3. 選項圈選
        options = item.get("options", [])
        if options and engine.cat_circles:
            num_circles = random.randint(4, min(14, len(options)))
            for opt_box in random.sample(options, num_circles):
                fname = random.choice(engine.cat_circles)
                stamp = cv2.imread(os.path.join(engine.marks_dir, fname), cv2.IMREAD_UNCHANGED)
                if stamp is None: continue
                cx = int((opt_box[0] + opt_box[2]) / 2.0 * sc) - int(stamp.shape[1] * base_scale) // 2 + random.randint(-6, 6)
                cy = int((opt_box[1] + opt_box[3]) / 2.0 * sc) - int(stamp.shape[0] * base_scale) // 2 + random.randint(-6, 6)
                engine.overlay_rgba_stamp(synth_img, synth_mask, stamp, cx, cy, base_scale * random.uniform(0.9, 1.3), random.uniform(-12, 12))

        # 4. 散落各處字母/數字註記
        if engine.cat_letters and random.random() < 0.9:
            num_scattered = random.randint(6, 18)
            for _ in range(num_scattered):
                fname = random.choice(engine.cat_letters)
                stamp = cv2.imread(os.path.join(engine.marks_dir, fname), cv2.IMREAD_UNCHANGED)
                if stamp is None: continue
                rx = random.randint(int(W * 0.05), int(W * 0.92))
                ry = random.randint(int(H * 0.08), int(H * 0.92))
                engine.overlay_rgba_stamp(synth_img, synth_mask, stamp, rx, ry, base_scale * random.uniform(0.8, 1.2), random.uniform(-15, 15))

        # 5. 題幹刪除塗抹
        text_lines = item.get("text_lines", [])
        if text_lines and engine.cat_strikes and random.random() < 0.75:
            num_strikes = random.randint(1, min(6, len(text_lines)))
            for line_box in random.sample(text_lines, num_strikes):
                fname = random.choice(engine.cat_strikes)
                stamp = cv2.imread(os.path.join(engine.marks_dir, fname), cv2.IMREAD_UNCHANGED)
                if stamp is None: continue
                lx = int(line_box[0] * sc) + random.randint(0, 80)
                ly = int(line_box[1] * sc) + random.randint(-6, 6)
                engine.overlay_rgba_stamp(synth_img, synth_mask, stamp, lx, ly, base_scale * random.uniform(0.85, 1.25), random.uniform(-4, 4))

        # 6. 批改記號
        if engine.cat_marks and random.random() < 0.65:
            for _ in range(random.randint(1, 5)):
                fname = random.choice(engine.cat_marks)
                stamp = cv2.imread(os.path.join(engine.marks_dir, fname), cv2.IMREAD_UNCHANGED)
                if stamp is None: continue
                rx = random.randint(int(W * 0.04), int(W * 0.88))
                ry = random.randint(int(H * 0.06), int(H * 0.90))
                engine.overlay_rgba_stamp(synth_img, synth_mask, stamp, rx, ry, base_scale * random.uniform(0.8, 1.2), random.uniform(-15, 15))

        # 7. 豐富演算推導與草稿
        if engine.layer_paths:
            for _ in range(random.randint(4, 8)):
                l_path = random.choice(engine.layer_paths)
                layer = cv2.imread(l_path, cv2.IMREAD_UNCHANGED)
                if layer is None: continue
                lh, lw = layer.shape[:2]
                cw = random.randint(400, min(1400, lw))
                ch = random.randint(350, min(1000, lh))
                cx = random.randint(0, lw - cw)
                cy = random.randint(0, lh - ch)
                crop_stamp = layer[cy:cy+ch, cx:cx+cw]
                if np.sum(crop_stamp[:, :, 3] > 0) > 400:
                    px = random.randint(20, max(21, W - cw - 20))
                    py = random.randint(50, max(51, H - ch - 50))
                    engine.overlay_rgba_stamp(synth_img, synth_mask, crop_stamp, px, py, random.uniform(0.9, 1.25), random.uniform(-8, 8))

        # 8. 共同輕微微歪斜
        if random.random() < 0.5:
            angle = random.uniform(-0.8, 0.8)
            M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
            synth_img = cv2.warpAffine(synth_img, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=(250, 250, 250))
            synth_mask = cv2.warpAffine(synth_mask, M, (W, H), flags=cv2.INTER_NEAREST, borderValue=0)
            target_bgr = cv2.warpAffine(target_bgr, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=(250, 250, 250))

        # 9. 嚴格 1:1 原生像素 Patch 採樣 (絕無縮放、絕不允許非 1024x1024)
        ph, pw = patch_size, patch_size
        if H < ph or W < pw:
            print(f"  [警告] 頁面尺寸 {W}x{H} 小於 Patch 尺寸 {pw}x{ph}，略過此頁")
            continue

        patches_from_this_page = random.randint(3, 5)
        for _ in range(patches_from_this_page):
            if generated >= num_samples:
                break
                
            # 80% 機率採樣有手寫筆跡或作答框的有效核心區
            for _ in range(8):
                cy = random.randint(0, H - ph)
                cx = random.randint(0, W - pw)
                p_mask = synth_mask[cy:cy+ph, cx:cx+pw]
                if random.random() < 0.8 and np.any(p_mask > 0):
                    break

            p_synth = synth_img[cy:cy+ph, cx:cx+pw]
            p_mask = synth_mask[cy:cy+ph, cx:cx+pw]
            p_tgt = target_bgr[cy:cy+ph, cx:cx+pw]

            # 嚴格品質校驗：尺寸必須 100% 精確吻合，否則拒絕寫入
            assert p_synth.shape == (ph, pw, 3), f"尺寸異常: p_synth shape={p_synth.shape}"
            assert p_mask.shape == (ph, pw), f"尺寸異常: p_mask shape={p_mask.shape}"
            assert p_tgt.shape == (ph, pw, 3), f"尺寸異常: p_tgt shape={p_tgt.shape}"

            # 存檔
            idx_str = f"{generated + 1:05d}"
            cv2.imwrite(os.path.join(in_dir, f"patch_{idx_str}.jpg"), p_synth, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(os.path.join(mask_dir, f"patch_{idx_str}.png"), p_mask)
            cv2.imwrite(os.path.join(tgt_dir, f"patch_{idx_str}.jpg"), p_tgt, [cv2.IMWRITE_JPEG_QUALITY, 95])

            generated += 1
            if generated % 50 == 0 or generated == num_samples:
                elapsed = time.time() - t0
                speed = generated / max(1.0, elapsed)
                print(f"  [進度] 已精確校驗並產出 {generated}/{num_samples} 組 1024x1024 Patch ({speed:.1f} patch/s, 已耗時 {elapsed:.1f}s)")

    print(f"\n🎉 預生成完畢！共產出 {num_samples} 組 1024x1024 Patch 資料集於 {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=600, help="生成 Patch 數量")
    parser.add_argument("--dpi", type=int, default=300, help="原生考卷 DPI")
    parser.add_argument("--patch_size", type=int, default=1024, help="Patch 邊長")
    args = parser.parse_args()

    generate_patch_dataset(num_samples=args.num, dpi=args.dpi, patch_size=args.patch_size)
