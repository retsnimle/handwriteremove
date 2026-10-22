#!/usr/bin/env python3
"""
測試 RapidOCR 定位臺灣考卷題號與作答括號 (Test OCR Bracket Detection)
"""
import fitz
import cv2
import numpy as np
import re
from rapidocr_onnxruntime import RapidOCR

ocr = RapidOCR()

doc = fitz.open("20251020202938305.pdf")
pix = doc[0].get_pixmap(dpi=150)
img_bgr = cv2.imdecode(np.frombuffer(pix.tobytes("png"), np.uint8), cv2.IMREAD_COLOR)
doc.close()

result, elapse_list = ocr(img_bgr)
print(f"RapidOCR 偵測到 {len(result) if result else 0} 個文字塊，耗時: {elapse_list}")

vis = img_bgr.copy()
bracket_zones = []

# 搜尋符合題號與括號模式的文字塊，例如：
# 1. (  ) 或 2. (  ) 或 ( D ) 或 1.(D) 等等
pattern = re.compile(r"(\d+[\.、]?\s*[\(（]|[\(（][A-Za-z\s]*[\)）]|[\(（]\s*[\)）])")

if result:
    for item in result:
        bbox, text, score = item
        # bbox 格式: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        pts = np.array(bbox, dtype=np.int32)
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)

        # 檢查是否包含括號或題號特徵
        if "(" in text or "（" in text or ")" in text or "）" in text or re.match(r"^\d+[\.、]", text):
            print(f"  [Match] Text: '{text}' (Conf: {score:.2f}) -> BBox: [{x_min}, {y_min}, {x_max}, {y_max}]")
            cv2.polylines(vis, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
            cv2.putText(vis, text, (x_min, max(15, y_min - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            bracket_zones.append((x_min, y_min, x_max, y_max, text))

cv2.imwrite("output/qa/ocr_bracket_detect_vis.png", vis)
print(f"視覺化結果已存至: output/qa/ocr_bracket_detect_vis.png, 總共鎖定 {len(bracket_zones)} 個潛在作答區")
