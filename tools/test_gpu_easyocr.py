#!/usr/bin/env python3
"""
EasyOCR 純 GPU 考卷題號與作答框偵測測試 (Pure GPU EasyOCR Test)
"""
import time
import torch
import cv2
import numpy as np
import fitz
import easyocr

print("=== 啟動 EasyOCR (CUDA GPU) 實測 ===")
t0 = time.time()
reader = easyocr.Reader(['en'], gpu=True)
print(f"★ EasyOCR 初始化就緒 (耗時: {time.time() - t0:.2f}s, Device: {reader.device})")

doc = fitz.open("20251020202938305.pdf")
pix = doc[0].get_pixmap(dpi=150)
img_bgr = cv2.imdecode(np.frombuffer(pix.tobytes("png"), np.uint8), cv2.IMREAD_COLOR)
doc.close()

# 監控 GPU 顯存
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

t_infer0 = time.time()
# 執行 GPU 推論 (只檢測或識別)
results = reader.readtext(img_bgr)
t_infer = time.time() - t_infer0

vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
print(f"★ 純 GPU 推論完成！耗時: {t_infer:.2f} 秒，GPU 顯存峰值: {vram_mb:.1f} MB，偵測到 {len(results)} 個文字塊。")

vis = img_bgr.copy()
matched = 0
for bbox, text, prob in results:
    pts = np.array(bbox, dtype=np.int32)
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    
    # 檢查是否包含題號或括號特徵
    if "(" in text or ")" in text or any(char.isdigit() for char in text):
        matched += 1
        cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(vis, text[:8], (x_min, max(15, y_min - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

cv2.imwrite("output/qa/gpu_easyocr_vis_p01.png", vis)
print(f"★ 視覺化標註已儲存至: output/qa/gpu_easyocr_vis_p01.png (命中 {matched} 個題號/符號候選框)")
