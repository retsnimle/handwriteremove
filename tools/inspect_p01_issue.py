#!/usr/bin/env python3
"""
深入診斷 Page 1 中段為何 (A)基因 A 和 DNA 會被誤判
"""
import cv2
import numpy as np
import fitz

doc = fitz.open("20251025101835564.pdf")
page = doc[0]
mat = fitz.Matrix(150 / 72.0, 150 / 72.0)
pix = page.get_pixmap(matrix=mat)
img = cv2.imdecode(np.frombuffer(pix.tobytes("png"), np.uint8), cv2.IMREAD_COLOR)
doc.close()

H, W = img.shape[:2]
y1, y2, x1, x2 = int(H * 0.4), int(H * 0.4) + 400, 100, 700
crop = img[y1:y2, x1:x2]

# 儲存高解析原圖以供分析
cv2.imwrite("C:/Users/retsn/.gemini/antigravity/brain/f687af4b-740f-482b-a1f5-e84a253db0a3/qa/p01_dense_raw_inspect.png", crop)
print("Saved raw inspect crop.")
