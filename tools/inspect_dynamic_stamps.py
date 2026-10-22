#!/usr/bin/env python3
"""
檢驗動態印章的隨機微擾動 (Dynamic Stamp Inspection)
針對真實記號 (hw_marks) 與草稿層 (hw_layers)，對同一印章進行 6 次不同擾動抽樣，
輸出「原始印章 vs 6 次動態微擾動」的特寫對照圖，供視覺檢查筆劃粗細、傾斜剪切與墨濃變化。
"""
import os
import glob
import random
import cv2
import numpy as np
from dataset_dynamic_scanned import FullDynamicScannedDataset

def main():
    ds = FullDynamicScannedDataset(
        scanned_root="scanned_pairs",
        assets_root="assets",
        catalog_path="assets/scanned_catalog.json",
        total_steps=10,
        dpi=150,
        is_train=True
    )

    qa_dir = r"C:\Users\retsn\.gemini\antigravity\brain\f687af4b-740f-482b-a1f5-e84a253db0a3\qa"
    os.makedirs(qa_dir, exist_ok=True)

    # 挑選 4 個具備代表性的印章 (1 個紅色勾記、1 個圈選、1 個手寫算式、1 個草稿碎片)
    mark_files = sorted(glob.glob("assets/hw_marks/*.png"))
    layer_files = sorted(glob.glob("assets/hw_layers/*.png"))

    selected = []
    # 找有顏色的紅筆記號
    for mf in mark_files:
        im = cv2.imread(mf, cv2.IMREAD_UNCHANGED)
        if im is not None and im.shape[0] > 30 and im.shape[1] > 30:
            b, g, r = [im[:, :, c][im[:, :, 3] > 0].mean() for c in range(3)]
            if r > b + 20 and r > g + 20: # 紅筆
                selected.append(("Red_Mark", im))
                break

    # 找黑色鉛筆/原子筆記號
    for mf in mark_files:
        im = cv2.imread(mf, cv2.IMREAD_UNCHANGED)
        if im is not None and im.shape[0] > 40 and im.shape[1] > 40:
            selected.append(("Black_Symbol", im))
            break

    # 找草稿切片
    if layer_files:
        l = cv2.imread(layer_files[0], cv2.IMREAD_UNCHANGED)
        crop = l[200:450, 200:550]
        selected.append(("Draft_Formula", crop))

    row_visualizations = []

    for name, raw_rgba in selected:
        h, w = raw_rgba.shape[:2]
        target_h, target_w = 200, 250

        def to_vis(rgba):
            # 貼在白底上方便肉眼觀察
            rgb = rgba[:, :, :3]
            alpha = (rgba[:, :, 3:4].astype(float) / 255.0)
            white = np.ones((rgba.shape[0], rgba.shape[1], 3), dtype=np.uint8) * 255
            blended = (rgb * alpha + white * (1.0 - alpha)).astype(np.uint8)
            # 等比縮放至 target 框內居中
            scale = min((target_w - 20) / blended.shape[1], (target_h - 20) / blended.shape[0])
            nw, nh = int(blended.shape[1] * scale), int(blended.shape[0] * scale)
            resized = cv2.resize(blended, (nw, nh))
            canvas = np.ones((target_h, target_w, 3), dtype=np.uint8) * 248
            y_off = (target_h - nh) // 2
            x_off = (target_w - nw) // 2
            canvas[y_off:y_off+nh, x_off:x_off+nw] = resized
            # 畫邊框
            cv2.rectangle(canvas, (0, 0), (target_w-1, target_h-1), (220, 220, 220), 1)
            return canvas

        cards = []
        raw_card = to_vis(raw_rgba)
        cv2.putText(raw_card, f"Raw: {name}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cards.append(raw_card)

        # 進行 5 次不同的動態擾動
        for k in range(5):
            jittered = ds.apply_stamp_dynamic_jitter(raw_rgba.copy())
            c = to_vis(jittered)
            cv2.putText(c, f"Jitter #{k+1}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 120, 0), 2)
            cards.append(c)

        row_visualizations.append(np.hstack(cards))

    final_grid = np.vstack(row_visualizations)
    out_path = os.path.join(qa_dir, "stamp_dynamic_inspection.png")
    cv2.imwrite(out_path, final_grid)
    print(f"動態印章視覺檢核切片已儲存: {out_path}")

if __name__ == "__main__":
    main()
