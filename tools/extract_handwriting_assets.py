#!/usr/bin/env python3
"""
手寫筆跡素材提取器 v2

原則：
  - 筆跡的每一個像素都來自使用者的掃描，不切碎、不以程式繪製任何筆劃。
  - 全部以 150 DPI 渲染，與純淨考卷的渲染解析度一致（筆劃物理尺寸一致）。

流程：
  1. 平場校正：估計紙張底色後相除，紙變純白、保留筆色。
  2. 遲滯門檻：深度 > HIGH 的像素作為種子，向外擴張到 > LOW 的相連區域。
     → 真實筆劃（含淡鉛筆的淡端）會被保留；孤立的淡色透印沒有種子而被排除。
  3. 裁邊：去除掃描器邊緣虛線。
  4. 輸出：
     - 整頁筆跡層 assets/hw_layers/*.png（RGBA，不切碎）
     - 元素表記號 assets/hw_marks/*.png（RGBA，完整單個記號）+ marks_meta.json
  5. 檢核圖 output/qa/：每頁「原掃描 | 保留的筆跡 | 被排除的像素(洋紅)」
"""

import os
import json
import glob
import argparse

import cv2
import numpy as np
import fitz

DPI = 150
LOW = 0.07          # 遲滯門檻低端（相對紙張的深度）
HIGH = 0.16         # 遲滯門檻高端（種子）
EDGE_CROP = 0.02    # 每邊裁掉的比例
MIN_AREA = 12       # 去除微小雜點（像素）
FRINGE = 0.03       # 筆緣保留門檻

# 元素表：(PDF 檔名主幹, 頁碼從 0 起算)
ELEMENT_SHEETS = {("20261021144305805", 0), ("20261021144305805", 1), ("20261021144305805", 2)}


def render_page(page):
    pix = page.get_pixmap(dpi=DPI, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def flat_field(bgr):
    """估計紙張底色（閉運算抹掉筆劃 + 中值平滑），原圖除以底色。"""
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))
    out = np.empty_like(bgr, dtype=np.float32)
    for c in range(3):
        ch = bgr[:, :, c]
        bg = cv2.morphologyEx(ch, cv2.MORPH_CLOSE, k)
        bg = cv2.medianBlur(bg, 31).astype(np.float32)
        out[:, :, c] = ch.astype(np.float32) / np.maximum(bg, 1.0)
    return np.clip(out, 0.0, 1.0)  # 1.0 = 紙白


def stroke_mask(norm):
    """遲滯門檻：深度取三通道最小值（紅、藍筆在某通道會很深）。"""
    depth = 1.0 - norm.min(axis=2)
    low = (depth > LOW).astype(np.uint8)
    high = depth > HIGH
    n, labels, stats, _ = cv2.connectedComponentsWithStats(low, connectivity=8)
    has_seed = np.zeros(n, dtype=bool)
    has_seed[np.unique(labels[high])] = True
    has_seed[0] = False
    keep = has_seed & (stats[:, cv2.CC_STAT_AREA] >= MIN_AREA)
    core = keep[labels]
    # 保留筆劃周圍 2px 內的淡色邊緣（掃描的柔和筆緣），避免筆劃變細變硬
    ring = cv2.dilate(core.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
    return core | (ring & (depth > FRINGE)), depth


def crop_edges(arr):
    h, w = arr.shape[:2]
    dy, dx = int(h * EDGE_CROP), int(w * EDGE_CROP)
    return arr[dy:h - dy, dx:w - dx]


def to_rgba(norm, mask):
    rgb = (norm * 255).astype(np.uint8)
    rgb[~mask] = 255
    alpha = mask.astype(np.uint8) * 255
    return np.dstack([rgb, alpha])


def qa_image(orig, norm, mask, depth, width=900):
    kept = (norm * 255).astype(np.uint8)
    kept[~mask] = 255
    rejected = np.full_like(orig, 255)
    rej = (~mask) & (depth > 0.04)
    rejected[rej] = (255, 0, 255)
    trio = np.hstack([orig, kept, rejected])
    s = width * 3 / trio.shape[1]
    return cv2.resize(trio, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


def qa_zoom(orig, norm, mask, depth, size=600):
    """原解析度局部放大：取筆跡最密集的區塊。"""
    h, w = mask.shape
    dens = cv2.boxFilter(mask.astype(np.float32), -1, (size, size))
    cy, cx = np.unravel_index(np.argmax(dens), dens.shape)
    y1 = int(np.clip(cy - size // 2, 0, max(0, h - size)))
    x1 = int(np.clip(cx - size // 2, 0, max(0, w - size)))
    sl = (slice(y1, y1 + size), slice(x1, x1 + size))
    kept = (norm * 255).astype(np.uint8)
    kept[~mask] = 255
    rejected = np.full_like(orig, 255)
    rejected[(~mask) & (depth > 0.04)] = (255, 0, 255)
    return np.hstack([orig[sl], kept[sl], rejected[sl]])


def extract_marks(rgba, mask, stem, page_idx, out_dir):
    """元素表：小幅膨脹後取連通區塊，以原始（未膨脹）遮罩裁出完整記號。"""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    grouped = cv2.dilate(mask.astype(np.uint8), k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(grouped, connectivity=8)
    metas = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if w < 10 and h < 10:
            continue
        pad = 3
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(mask.shape[1], x + w + pad), min(mask.shape[0], y + h + pad)
        sub = rgba[y1:y2, x1:x2].copy()
        own = labels[y1:y2, x1:x2] == i
        sub[~own] = (255, 255, 255, 0)
        ink = sub[:, :, 3] > 0
        if ink.sum() < 30:
            continue
        b, g, r = [float(sub[:, :, c][ink].mean()) for c in range(3)]
        name = f"{stem}_p{page_idx + 1:02d}_m{i:04d}.png"
        cv2.imwrite(os.path.join(out_dir, name), sub)
        metas.append({"file": name, "w": int(x2 - x1), "h": int(y2 - y1),
                      "ink_px": int(ink.sum()), "mean_bgr": [round(b), round(g), round(r)]})
    return metas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="handwritten_samples")
    args = ap.parse_args()

    layer_dir, mark_dir, qa_dir = "assets/hw_layers", "assets/hw_marks", "output/qa"
    for d in (layer_dir, mark_dir, qa_dir):
        os.makedirs(d, exist_ok=True)

    all_marks = []
    for pdf in sorted(glob.glob(os.path.join(args.src, "*.pdf"))):
        stem = os.path.splitext(os.path.basename(pdf))[0]
        doc = fitz.open(pdf)
        for pi in range(doc.page_count):
            orig = crop_edges(render_page(doc[pi]))
            norm = flat_field(orig)
            mask, depth = stroke_mask(norm)
            rgba = to_rgba(norm, mask)

            tag = f"{stem}_p{pi + 1:02d}"
            cv2.imwrite(os.path.join(qa_dir, f"{tag}.jpg"), qa_image(orig, norm, mask, depth),
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
            cv2.imwrite(os.path.join(qa_dir, f"{tag}_zoom.png"), qa_zoom(orig, norm, mask, depth))

            if (stem, pi) in ELEMENT_SHEETS:
                metas = extract_marks(rgba, mask, stem, pi, mark_dir)
                all_marks.extend(metas)
                print(f"[元素表] {tag}: 切出 {len(metas)} 個記號")
            else:
                cv2.imwrite(os.path.join(layer_dir, f"{tag}.png"), rgba)
                print(f"[筆跡層] {tag}: 筆跡像素 {int(mask.sum())}")
        doc.close()

    with open(os.path.join(mark_dir, "marks_meta.json"), "w", encoding="utf-8") as f:
        json.dump(all_marks, f, ensure_ascii=False, indent=1)
    print(f"完成：筆跡層 {len(glob.glob(layer_dir + '/*.png'))} 頁，記號 {len(all_marks)} 個")


if __name__ == "__main__":
    main()
