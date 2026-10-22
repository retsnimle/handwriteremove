#!/usr/bin/env python3
"""
高解析度考卷全頁 (Full-Page) 訓練集生成工具 (原生尺寸無損自適應版)

核心原則：
  1. 零變形、零拉伸、零縮放 (Zero Resizing / Zero Distortion)：
     原本乾淨的 PDF 頁面在指定 DPI (150 DPI) 下渲染出多少像素 (W x H)，
     訓練集產出的 input, mask, target 就 100% 嚴格等於多少像素 (W x H)！
  2. 直式 (A4, ~1241x1754) 與橫式 (A3/答題卷, ~2481x1754) 自適應保持原比例。
  3. 保留整頁完整的宏觀排版語意 (題號、括號填答、選項圈選、題幹刪除塗改、邊界草稿)。
"""

import os
import sys
import time
import argparse
import random
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from synthetic_engine_v2 import ExamSynthesizer

def generate_fullpage_dataset(num_samples=300, out_dir="dataset_fullpage", dpi=150):
    in_dir = os.path.join(out_dir, "images")
    mask_dir = os.path.join(out_dir, "masks")
    tgt_dir = os.path.join(out_dir, "targets")
    os.makedirs(in_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(tgt_dir, exist_ok=True)

    print(f"=== 開始高解析度原生考卷訓練集生成 (零變形自適應) ===")
    print(f"目標頁數: {num_samples} | 解析度: {dpi} DPI | 模式: 100% 原生尺寸直出 (絕不縮放變形)")

    engine = ExamSynthesizer(dpi=dpi)
    sc = engine.scale_pt_to_px

    t0 = time.time()
    generated = 0

    catalog_len = len(engine.catalog)
    indices = list(range(catalog_len))
    random.shuffle(indices)

    while generated < num_samples:
        for idx in indices:
            if generated >= num_samples:
                break

            item = engine.catalog[idx]
            # 1. 直接以原生尺寸渲染純淨 PDF 頁面
            raw_clean = engine.render_clean_page(item["pdf"], item["page"])
            H, W = raw_clean.shape[:2]
            
            # 2. 印刷退化 (PSF 模糊 + 紙色 + 感測器微雜訊)
            target_bgr = engine.degrade_clean_print(raw_clean)
            synth_img = target_bgr.copy()
            synth_mask = np.zeros((H, W), dtype=np.uint8)

            # 3. 高覆蓋率填答作答框 (85% ~ 100%)
            brackets = item.get("brackets", [])
            if brackets and engine.cat_letters:
                num_ans = max(1, int(len(brackets) * random.uniform(0.85, 1.0)))
                for box in random.sample(brackets, min(num_ans, len(brackets))):
                    fname = random.choice(engine.cat_letters)
                    stamp = cv2.imread(os.path.join(engine.marks_dir, fname), cv2.IMREAD_UNCHANGED)
                    if stamp is None: continue
                    bx = int(box[0] * sc) + random.randint(-5, 8)
                    by = int(box[1] * sc) + random.randint(-6, 6)
                    engine.overlay_rgba_stamp(synth_img, synth_mask, stamp, bx, by, random.uniform(0.85, 1.25), random.uniform(-10, 10))

            # 4. 選項圈選
            options = item.get("options", [])
            if options and engine.cat_circles:
                max_circles = min(12, len(options))
                num_circles = random.randint(1, max_circles)
                for opt_box in random.sample(options, num_circles):
                    fname = random.choice(engine.cat_circles)
                    stamp = cv2.imread(os.path.join(engine.marks_dir, fname), cv2.IMREAD_UNCHANGED)
                    if stamp is None: continue
                    cx = int((opt_box[0] + opt_box[2]) / 2.0 * sc) - stamp.shape[1] // 2 + random.randint(-4, 4)
                    cy = int((opt_box[1] + opt_box[3]) / 2.0 * sc) - stamp.shape[0] // 2 + random.randint(-4, 4)
                    engine.overlay_rgba_stamp(synth_img, synth_mask, stamp, cx, cy, random.uniform(0.85, 1.3), random.uniform(-12, 12))

            # 5. 散落各處解題字母/註記 (5 ~ 15 個)
            if engine.cat_letters and random.random() < 0.9:
                num_scattered = random.randint(5, 15)
                for _ in range(num_scattered):
                    fname = random.choice(engine.cat_letters)
                    stamp = cv2.imread(os.path.join(engine.marks_dir, fname), cv2.IMREAD_UNCHANGED)
                    if stamp is None: continue
                    rx = random.randint(int(W * 0.05), int(W * 0.92))
                    ry = random.randint(int(H * 0.08), int(H * 0.92))
                    engine.overlay_rgba_stamp(synth_img, synth_mask, stamp, rx, ry, random.uniform(0.75, 1.15), random.uniform(-15, 15))

            # 6. 題幹刪除塗改
            text_lines = item.get("text_lines", [])
            if text_lines and engine.cat_strikes and random.random() < 0.75:
                num_strikes = random.randint(1, min(5, len(text_lines)))
                for line_box in random.sample(text_lines, num_strikes):
                    fname = random.choice(engine.cat_strikes)
                    stamp = cv2.imread(os.path.join(engine.marks_dir, fname), cv2.IMREAD_UNCHANGED)
                    if stamp is None: continue
                    lx = int(line_box[0] * sc) + random.randint(0, 50)
                    ly = int(line_box[1] * sc) + random.randint(-4, 4)
                    engine.overlay_rgba_stamp(synth_img, synth_mask, stamp, lx, ly, random.uniform(0.8, 1.25), random.uniform(-4, 4))

            # 7. 批改紅筆記號
            if engine.cat_marks and random.random() < 0.6:
                for _ in range(random.randint(1, 4)):
                    fname = random.choice(engine.cat_marks)
                    stamp = cv2.imread(os.path.join(engine.marks_dir, fname), cv2.IMREAD_UNCHANGED)
                    if stamp is None: continue
                    rx = random.randint(int(W * 0.04), int(W * 0.88))
                    ry = random.randint(int(H * 0.06), int(H * 0.90))
                    engine.overlay_rgba_stamp(synth_img, synth_mask, stamp, rx, ry, random.uniform(0.7, 1.1), random.uniform(-15, 15))

            # 8. 演算推導與草稿
            if engine.layer_paths:
                for _ in range(random.randint(3, 6)):
                    l_path = random.choice(engine.layer_paths)
                    layer = cv2.imread(l_path, cv2.IMREAD_UNCHANGED)
                    if layer is None: continue
                    lh, lw = layer.shape[:2]
                    cw = random.randint(min(280, lw), min(800, lw))
                    ch = random.randint(min(220, lh), min(650, lh))
                    cx = random.randint(0, lw - cw)
                    cy = random.randint(0, lh - ch)
                    crop_stamp = layer[cy:cy+ch, cx:cx+cw]
                    if np.sum(crop_stamp[:, :, 3] > 0) > 250:
                        px = random.randint(15, max(16, W - cw - 15))
                        py = random.randint(35, max(36, H - ch - 35))
                        engine.overlay_rgba_stamp(synth_img, synth_mask, crop_stamp, px, py, random.uniform(0.8, 1.15), random.uniform(-8, 8))

            # 9. 輕微共同微歪斜
            if random.random() < 0.5:
                angle = random.uniform(-0.6, 0.6)
                M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
                synth_img = cv2.warpAffine(synth_img, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=(250, 250, 250))
                synth_mask = cv2.warpAffine(synth_mask, M, (W, H), flags=cv2.INTER_NEAREST, borderValue=0)
                target_bgr = cv2.warpAffine(target_bgr, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=(250, 250, 250))

            # 關鍵保證：絕不 resize，原生大小 1:1 直出
            assert synth_img.shape == (H, W, 3)
            assert synth_mask.shape == (H, W)
            assert target_bgr.shape == (H, W, 3)

            idx_str = f"{generated + 1:05d}"
            cv2.imwrite(os.path.join(in_dir, f"page_{idx_str}.jpg"), synth_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(os.path.join(mask_dir, f"page_{idx_str}.png"), synth_mask)
            cv2.imwrite(os.path.join(tgt_dir, f"page_{idx_str}.jpg"), target_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

            generated += 1
            if generated % 25 == 0 or generated == num_samples:
                elapsed = time.time() - t0
                speed = generated / max(1.0, elapsed)
                print(f"  [進度] 已產出 {generated}/{num_samples} 頁原生考卷 (當前頁面: {W}x{H}, {speed:.2f} 頁/秒, 已耗時 {elapsed:.1f}s)")

    print(f"\n🎉 原生全頁考卷訓練集製作完畢！共產出 {num_samples} 頁於 {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=300, help="生成全頁數量")
    parser.add_argument("--dpi", type=int, default=150, help="DPI")
    args = parser.parse_args()

    generate_fullpage_dataset(num_samples=args.num, dpi=args.dpi)
