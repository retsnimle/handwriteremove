#!/usr/bin/env python3
"""
高解析度考卷全頁 (Full-Page) 多進程極速生成工具 (1500 頁無損自適應版)
利用多核心 CPU 平行渲染與正片疊底合成，全速產出 1500 頁全頁題目卷資料集。
"""

import os
import sys
import time
import argparse
import random
import glob
import cv2
import numpy as np
from multiprocessing import Pool, cpu_count

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from synthetic_engine_v2 import ExamSynthesizer

# 全域合成引擎 (各子進程獨立初始化)
_global_engine = None

def init_worker(dpi=150):
    global _global_engine
    _global_engine = ExamSynthesizer(dpi=dpi)

def generate_single_page(args):
    """子進程任務：生成單張原生全頁考卷"""
    out_idx, item_idx, in_dir, mask_dir, tgt_dir = args
    global _global_engine
    engine = _global_engine
    sc = engine.scale_pt_to_px

    item = engine.catalog[item_idx]
    raw_clean = engine.render_clean_page(item["pdf"], item["page"])
    H, W = raw_clean.shape[:2]

    target_bgr = engine.degrade_clean_print(raw_clean)
    synth_img = target_bgr.copy()
    synth_mask = np.zeros((H, W), dtype=np.uint8)

    # 1. 括號填答 (85% ~ 100%)
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

    # 2. 選項圈選
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

    # 3. 散落解題註記
    if engine.cat_letters and random.random() < 0.9:
        for _ in range(random.randint(5, 15)):
            fname = random.choice(engine.cat_letters)
            stamp = cv2.imread(os.path.join(engine.marks_dir, fname), cv2.IMREAD_UNCHANGED)
            if stamp is None: continue
            rx = random.randint(int(W * 0.05), int(W * 0.92))
            ry = random.randint(int(H * 0.08), int(H * 0.92))
            engine.overlay_rgba_stamp(synth_img, synth_mask, stamp, rx, ry, random.uniform(0.75, 1.15), random.uniform(-15, 15))

    # 4. 題幹刪除塗抹
    text_lines = item.get("text_lines", [])
    if text_lines and engine.cat_strikes and random.random() < 0.75:
        for line_box in random.sample(text_lines, random.randint(1, min(5, len(text_lines)))):
            fname = random.choice(engine.cat_strikes)
            stamp = cv2.imread(os.path.join(engine.marks_dir, fname), cv2.IMREAD_UNCHANGED)
            if stamp is None: continue
            lx = int(line_box[0] * sc) + random.randint(0, 50)
            ly = int(line_box[1] * sc) + random.randint(-4, 4)
            engine.overlay_rgba_stamp(synth_img, synth_mask, stamp, lx, ly, random.uniform(0.8, 1.25), random.uniform(-4, 4))

    # 5. 批改記號
    if engine.cat_marks and random.random() < 0.6:
        for _ in range(random.randint(1, 4)):
            fname = random.choice(engine.cat_marks)
            stamp = cv2.imread(os.path.join(engine.marks_dir, fname), cv2.IMREAD_UNCHANGED)
            if stamp is None: continue
            rx = random.randint(int(W * 0.04), int(W * 0.88))
            ry = random.randint(int(H * 0.06), int(H * 0.90))
            engine.overlay_rgba_stamp(synth_img, synth_mask, stamp, rx, ry, random.uniform(0.7, 1.1), random.uniform(-15, 15))

    # 6. 演算推導與草稿
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

    # 7. 輕微微歪斜
    if random.random() < 0.5:
        angle = random.uniform(-0.6, 0.6)
        M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
        synth_img = cv2.warpAffine(synth_img, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=(250, 250, 250))
        synth_mask = cv2.warpAffine(synth_mask, M, (W, H), flags=cv2.INTER_NEAREST, borderValue=0)
        target_bgr = cv2.warpAffine(target_bgr, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=(250, 250, 250))

    # 存檔 (保持原生尺寸)
    idx_str = f"{out_idx:05d}"
    cv2.imwrite(os.path.join(in_dir, f"page_{idx_str}.jpg"), synth_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(os.path.join(mask_dir, f"page_{idx_str}.png"), synth_mask)
    cv2.imwrite(os.path.join(tgt_dir, f"page_{idx_str}.jpg"), target_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return out_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total", type=int, default=1500, help="目標總頁數")
    parser.add_argument("--dpi", type=int, default=150, help="渲染 DPI")
    parser.add_argument("--workers", type=int, default=4, help="並行進程數")
    parser.add_argument("--out_dir", type=str, default="dataset_fullpage", help="輸出目錄")
    args = parser.parse_args()

    in_dir = os.path.join(args.out_dir, "images")
    mask_dir = os.path.join(args.out_dir, "masks")
    tgt_dir = os.path.join(args.out_dir, "targets")
    os.makedirs(in_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(tgt_dir, exist_ok=True)

    # 檢查現有已生成頁數
    existing = len(glob.glob(os.path.join(in_dir, "*.jpg")))
    print(f"=== 多進程全頁考卷生成啟動 ===")
    print(f"目標: {args.total} 頁 | 現有: {existing} 頁 | 待生成: {args.total - existing} 頁 | 進程數: {args.workers}")

    if existing >= args.total:
        print(f"🎉 現有頁數 ({existing}) 已達到或超過目標 ({args.total})，無需重新生成！")
        return

    # 載入目錄
    temp_engine = ExamSynthesizer(dpi=args.dpi)
    catalog_len = len(temp_engine.catalog)
    del temp_engine

    # 構建任務列表
    tasks = []
    for out_idx in range(existing + 1, args.total + 1):
        item_idx = random.randint(0, catalog_len - 1)
        tasks.append((out_idx, item_idx, in_dir, mask_dir, tgt_dir))

    t0 = time.time()
    completed = existing

    # 啟動進程池
    with Pool(processes=args.workers, initializer=init_worker, initargs=(args.dpi,)) as pool:
        for idx in pool.imap_unordered(generate_single_page, tasks, chunksize=5):
            completed += 1
            if completed % 50 == 0 or completed == args.total:
                elapsed = time.time() - t0
                speed = (completed - existing) / max(1.0, elapsed)
                remain_s = (args.total - completed) / max(0.1, speed)
                print(f"  [進度] 已完成 {completed}/{args.total} 頁 ({speed:.2f} 頁/秒, 已耗時 {elapsed:.1f}s, 預估剩餘 {remain_s/60:.1f} 分鐘)")

    total_time = time.time() - t0
    print(f"\n🎉 1500 頁全頁題目卷資料集全部生成完畢！共耗時 {total_time:.1f} 秒。")

if __name__ == "__main__":
    main()
