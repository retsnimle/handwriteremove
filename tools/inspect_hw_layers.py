#!/usr/bin/env python3
"""檢查訓練用筆跡層 (hw_layers) 是否混入印刷文字：把 RGBA 合成到白底，輸出縮圖總覽。"""
import glob
import os
import cv2
import numpy as np

QA = r"C:\Users\retsn\.gemini\antigravity\brain\f687af4b-740f-482b-a1f5-e84a253db0a3\qa"
paths = sorted(glob.glob("assets/hw_layers/*.png"))

def on_white(rgba):
    a = rgba[:, :, 3:4].astype(np.float32) / 255.0
    return (rgba[:, :, :3] * a + 255 * (1 - a)).astype(np.uint8)

thumbs = []
for p in paths:
    im = on_white(cv2.imread(p, cv2.IMREAD_UNCHANGED))
    t = cv2.resize(im, (300, int(im.shape[0] * 300 / im.shape[1])), interpolation=cv2.INTER_AREA)
    t = cv2.copyMakeBorder(t[:420], 0, max(0, 420 - t[:420].shape[0]), 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(t, os.path.basename(p)[-10:-4], (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    thumbs.append(t)
while len(thumbs) % 7:
    thumbs.append(np.full_like(thumbs[0], 255))
rows = [np.hstack(thumbs[i:i + 7]) for i in range(0, len(thumbs), 7)]
cv2.imwrite(os.path.join(QA, "hw_layers_overview.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])

# 全解析度：第一頁左上 900x700
full = on_white(cv2.imread(paths[0], cv2.IMREAD_UNCHANGED))
cv2.imwrite(os.path.join(QA, "hw_layer_p01_full_crop.png"), full[:700, :900])
print("done", len(paths))
