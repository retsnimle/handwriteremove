#!/usr/bin/env python3
r"""
手寫筆跡素材 300 DPI 原生極限提取器 (Handwriting Assets Extractor 300 DPI)
========================================================================
1. 原生 300 DPI 渲染原始手寫樣本 PDF，1:1 保留石墨微粒、紙張纖維與墨水毛細擴散微觀紋理。
2. 平場校正與遲滯門檻自動縮放適配 300 DPI 尺度。
3. 輸出：
   - assets/hw_layers_300dpi/*.png (RGBA 全頁草稿圖層)
   - assets/hw_marks_300dpi/*.png (RGBA 高解析度手寫記號/印章)
   - assets/hw_marks_300dpi/marks_meta.json
"""

import os
import json
import glob
import argparse
import cv2
import numpy as np
import fitz

DPI = 300
LOW = 0.07          # 遲滯門檻低端
HIGH = 0.16         # 遲滯門檻高端（種子）
EDGE_CROP = 0.02    # 每邊裁掉的比例
MIN_AREA = 48       # 去除微小雜點 (300 DPI 下約為 150 DPI 的 4 倍)
FRINGE = 0.03       # 筆緣保留門檻

ELEMENT_SHEETS = {("20261021144305805", 0), ("20261021144305805", 1), ("20261021144305805", 2)}


def render_page(page):
    pix = page.get_pixmap(dpi=DPI, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def flat_field(bgr):
    """估計紙張底色（300 DPI 下使用 61x61 閉運算抹掉筆劃 + 中值平滑）"""
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (61, 61))
    out = np.empty_like(bgr, dtype=np.float32)
    for c in range(3):
        ch = bgr[:, :, c]
        bg = cv2.morphologyEx(ch, cv2.MORPH_CLOSE, k)
        bg = cv2.medianBlur(bg, 61).astype(np.float32)
        out[:, :, c] = ch.astype(np.float32) / np.maximum(bg, 1.0)
    return np.clip(out, 0.0, 1.0)


def stroke_mask(norm):
    """遲滯門檻：保留真實 300 DPI 筆劃及其柔和筆緣"""
    depth = 1.0 - norm.min(axis=2)
    low = (depth > LOW).astype(np.uint8)
    high = depth > HIGH
    n, labels, stats, _ = cv2.connectedComponentsWithStats(low, connectivity=8)
    has_seed = np.zeros(n, dtype=bool)
    has_seed[np.unique(labels[high])] = True
    has_seed[0] = False
    keep = has_seed & (stats[:, cv2.CC_STAT_AREA] >= MIN_AREA)
    core = keep[labels]
    # 保留筆劃周圍 4px 內的淡色邊緣（300 DPI 掃描柔和筆緣）
    ring = cv2.dilate(core.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0
    return core | (ring & (depth > FRINGE)), depth


def crop_edges(arr):
    h, w = arr.shape[:2]
    dy, dx = int(h * EDGE_CROP), int(w * EDGE_CROP)
    return arr[dy : h - dy, dx : w - dx]


def to_rgba(norm, mask):
    rgb = (norm * 255).astype(np.uint8)
    rgb[~mask] = 255
    alpha = mask.astype(np.uint8) * 255
    return np.dstack([rgb, alpha])


def extract_marks(rgba, mask, stem, page_idx, out_dir):
    """元素表記號提取（300 DPI 膨脹適配）"""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    grouped = cv2.dilate(mask.astype(np.uint8), k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(grouped, connectivity=8)
    metas = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if w < 16 and h < 16:
            continue
        pad = 6
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(mask.shape[1], x + w + pad), min(mask.shape[0], y + h + pad)
        sub = rgba[y1:y2, x1:x2].copy()
        own = labels[y1:y2, x1:x2] == i
        sub[~own] = (255, 255, 255, 0)
        ink = sub[:, :, 3] > 0
        if ink.sum() < 80:
            continue
        b, g, r = [float(sub[:, :, c][ink].mean()) for c in range(3)]
        name = f"{stem}_p{page_idx + 1:02d}_m{i:04d}.png"
        cv2.imwrite(os.path.join(out_dir, name), sub)
        metas.append({
            "file": name,
            "w": int(x2 - x1),
            "h": int(y2 - y1),
            "ink_px": int(ink.sum()),
            "mean_bgr": [round(b), round(g), round(r)]
        })
    return metas


def main():
    src_dir = "handwritten_samples"
    layer_dir = "assets/hw_layers_300dpi"
    mark_dir = "assets/hw_marks_300dpi"
    for d in (layer_dir, mark_dir):
        os.makedirs(d, exist_ok=True)

    all_marks = []
    pdf_list = sorted(glob.glob(os.path.join(src_dir, "*.pdf")))
    print(f"=== [300 DPI 筆跡素材提取] 正在自 {len(pdf_list)} 份原始樣本中提取 300 DPI 原生筆跡與記號 ===", flush=True)

    for pdf in pdf_list:
        stem = os.path.splitext(os.path.basename(pdf))[0]
        doc = fitz.open(pdf)
        for pi in range(doc.page_count):
            orig = crop_edges(render_page(doc[pi]))
            norm = flat_field(orig)
            mask, depth = stroke_mask(norm)
            rgba = to_rgba(norm, mask)
            tag = f"{stem}_p{pi + 1:02d}"

            if (stem, pi) in ELEMENT_SHEETS:
                metas = extract_marks(rgba, mask, stem, pi, mark_dir)
                all_marks.extend(metas)
                print(f"  [300 DPI 元素表] {tag}: 切出 {len(metas)} 個高解析度記號", flush=True)
            else:
                out_png = os.path.join(layer_dir, f"{tag}.png")
                cv2.imwrite(out_png, rgba)
                print(f"  [300 DPI 筆跡層] {tag}: 筆跡像素 {int(mask.sum())} -> {os.path.basename(out_png)}", flush=True)
        doc.close()

    with open(os.path.join(mark_dir, "marks_meta.json"), "w", encoding="utf-8") as f:
        json.dump(all_marks, f, ensure_ascii=False, indent=2)

    total_layers = len(glob.glob(layer_dir + "/*.png"))
    print(f"★ 300 DPI 手寫素材提取完成！共取得筆跡層 {total_layers} 頁，記號 {len(all_marks)} 個！", flush=True)


if __name__ == "__main__":
    main()
