import cv2
import numpy as np

img = cv2.imread(r'G:\CODE\handwriteremove\assets\debug_p1_crop.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, bin_inv = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

contours, _ = cv2.findContours(bin_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
boxes = [cv2.boundingRect(c) for c in contours]
# 找 y 在 150~350 之間最左邊的
cand = [b for b in boxes if 150 <= b[1] <= 350 and b[0] < 200]
cand.sort(key=lambda b: (b[1], b[0]))
for b in cand:
    print(f"bbox: x={b[0]}, y={b[1]}, w={b[2]}, h={b[3]}")
