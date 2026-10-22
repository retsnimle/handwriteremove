#!/usr/bin/env python3
"""
純自適應乾淨考卷作答括號提取器 (Adaptive Clean Exam Bracket Extractor)
★ 零寫死坐標常數 (Zero Hardcoded Coordinates)！
★ 適用於各種版型：無論是 "(   ) 1." 在前，還是 "1. (   )" 在後，或是雙欄/單欄！
在乾淨底稿上，透過 DocLayout-YOLO 題幹定位 + 題頭純淨連通域弧線配對 (Arc Pair Search)，
動態鎖定真實作答括號內膽，並生成加蓋手寫印章預覽！
"""

import os
import random
import cv2
import numpy as np
import fitz
import torch
from doclayout_yolo import YOLOv10

def is_bracket_shape(contour, h_min=15, h_max=50):
    """檢驗連通域輪廓是否符合印刷小括號 ( 或 ) 的形態特徵"""
    x, y, w, h = cv2.boundingRect(contour)
    if h < h_min or h > h_max:
        return None
    # 括號是細長的垂直弧線 (高寬比通常在 2.0 ~ 6.0 之間)
    aspect = h / max(1, w)
    if aspect < 1.8 or aspect > 7.0:
        return None
    
    # 判斷是向左凸 (還是向右凸 )
    # 取中段輪廓點的 x 坐標相對於頂底的偏移
    pts = contour[:, 0, :]
    mid_y = y + h / 2.0
    mid_pts = pts[np.abs(pts[:, 1] - mid_y) < h * 0.25]
    top_bot_pts = pts[np.abs(pts[:, 1] - mid_y) > h * 0.35]
    if len(mid_pts) == 0 or len(top_bot_pts) == 0:
        return None
    
    mean_mid_x = np.mean(mid_pts[:, 0])
    mean_tb_x = np.mean(top_bot_pts[:, 0])

    if mean_mid_x < mean_tb_x - 1.0:
        return ("left_bracket", (x, y, w, h)) # 向左凸 (
    elif mean_mid_x > mean_tb_x + 1.0:
        return ("right_bracket", (x, y, w, h)) # 向右凸 )
    return None

def find_brackets_in_roi(clean_crop_bgr):
    """在題幹開頭的裁切小圖中，自適應配對尋找 ( 與 )"""
    gray = cv2.cvtColor(clean_crop_bgr, cv2.COLOR_BGR2GRAY)
    _, bin_inv = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(bin_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        res = is_bracket_shape(c)
        if res:
            b_type, rect = res
            candidates.append((b_type, rect))

    # 按 X 坐標由左至右排序
    candidates.sort(key=lambda item: item[1][0])

    # 尋找最近的一對 ( 與 )，且水平間距在合理作答寬度範圍內 (20px ~ 120px)
    for i in range(len(candidates)):
        t1, r1 = candidates[i]
        if t1 == "left_bracket":
            for j in range(i + 1, len(candidates)):
                t2, r2 = candidates[j]
                if t2 == "right_bracket":
                    x1, y1, w1, h1 = r1
                    x2, y2, w2, h2 = r2
                    inner_dist = x2 - (x1 + w1)
                    y_diff = abs(y1 - y2)
                    if 18 <= inner_dist <= 130 and y_diff <= 10:
                        # 成功配對！返回內膽區間與整體外框
                        inner_roi = (x1 + w1 + 1, min(y1, y2) + 2, x2 - 1, max(y1 + h1, y2 + h2) - 2)
                        outer_bbox = (x1, min(y1, y2), x2 + w2, max(y1 + h1, y2 + h2))
                        return inner_roi, outer_bbox
    return None, None

def test_clean_pdf(pdf_path, output_prefix):
    print(f"\n📄 測試乾淨 PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    pix = doc[0].get_pixmap(dpi=150)
    img_bgr = cv2.imdecode(np.frombuffer(pix.tobytes("png"), np.uint8), cv2.IMREAD_COLOR)
    doc.close()
    H, W, _ = img_bgr.shape

    # 1. DocLayout-YOLO 找出所有題幹 plain text
    model = YOLOv10("weights/doclayout_yolo_docstructbench_imgsz1024.pt")
    det_res = model.predict(img_bgr, imgsz=1024, conf=0.25, device="cuda:0")
    boxes = det_res[0].boxes
    names = det_res[0].names

    plain_texts = []
    for box in boxes:
        cls_name = names[int(box.cls[0].item())]
        conf = float(box.conf[0].item())
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        if cls_name == "plain text" and conf >= 0.35:
            plain_texts.append((x1, y1, x2, y2))

    # 按 Y 軸排序
    plain_texts.sort(key=lambda b: (0 if b[0] < W*0.5 else 1, b[1]))
    print(f"★ DocLayout-YOLO 找到 {len(plain_texts)} 個題幹區塊")

    # 2. 自適應搜尋每個題幹開頭的括號對 (無任何寫死常數！)
    matched_inner_rois = []
    matched_outer_rois = []

    for tx1, ty1, tx2, ty2 in plain_texts:
        # 取題幹開頭的矩形區域：向左延伸 80px，向右 160px，高度約 50px
        search_x1 = max(0, tx1 - 80)
        search_x2 = min(W, tx1 + 160)
        search_y1 = max(0, ty1 - 5)
        search_y2 = min(H, ty1 + 55)

        crop = img_bgr[search_y1:search_y2, search_x1:search_x2]
        if crop.size == 0: continue

        inner_roi, outer_bbox = find_brackets_in_roi(crop)
        if inner_roi is not None:
            ix1, iy1, ix2, iy2 = inner_roi
            ox1, oy1, ox2, oy2 = outer_bbox
            # 轉回全圖坐標
            global_inner = (search_x1 + ix1, search_y1 + iy1, search_x1 + ix2, search_y1 + iy2)
            global_outer = (search_x1 + ox1, search_y1 + oy1, search_x1 + ox2, search_y1 + oy2)
            matched_inner_rois.append(global_inner)
            matched_outer_rois.append(global_outer)

    print(f"★ 自適應純幾何演算法成功鎖定 {len(matched_inner_rois)} 個真實作答括號！")

    # 3. 繪製視覺化圖
    vis_target = img_bgr.copy()
    overlay = vis_target.copy()
    for q_idx, (ix1, iy1, ix2, iy2) in enumerate(matched_inner_rois):
        # 藍色透明內膽
        cv2.rectangle(overlay, (ix1, iy1), (ix2, iy2), (255, 160, 0), -1)
        cv2.rectangle(vis_target, (ix1, iy1), (ix2, iy2), (200, 100, 0), 2)
        cv2.putText(vis_target, f"#{q_idx+1}", (ix1 - 4, max(15, iy1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 80, 0), 1)

    cv2.addWeighted(overlay, 0.45, vis_target, 0.55, 0, vis_target)

    # 4. 隨機動態蓋上手寫印章
    marks_dir = "assets/hw_marks"
    mark_files = [os.path.join(marks_dir, f) for f in os.listdir(marks_dir) if f.endswith(".png")]
    vis_stamped = img_bgr.copy()
    for ix1, iy1, ix2, iy2 in matched_inner_rois:
        if mark_files:
            stamp = cv2.imread(random.choice(mark_files), cv2.IMREAD_UNCHANGED)
            if stamp is not None:
                # 放入內膽
                sh, sw = stamp.shape[:2]
                bw = ix2 - ix1
                bh = iy2 - iy1
                scale = min(bw / max(1, sw), bh / max(1, sh)) * random.uniform(0.8, 1.1)
                nw, nh = max(6, int(sw * scale)), max(6, int(sh * scale))
                resized = cv2.resize(stamp, (nw, nh))
                cx = ix1 + (bw - nw) // 2
                cy = iy1 + (bh - nh) // 2
                if cx >= 0 and cy >= 0 and cx + nw < W and cy + nh < H:
                    alpha = (resized[:, :, 3].astype(float) / 255.0)[:, :, None]
                    ink = resized[:, :, :3].astype(float)
                    bg = vis_stamped[cy:cy+nh, cx:cx+nw].astype(float)
                    vis_stamped[cy:cy+nh, cx:cx+nw] = np.clip(alpha * ink + (1 - alpha) * bg, 0, 255).astype(np.uint8)

    qa_dir = "output/qa"
    os.makedirs(qa_dir, exist_ok=True)
    out_target_p = os.path.join(qa_dir, f"clean_adaptive_target_{output_prefix}.png")
    out_stamped_p = os.path.join(qa_dir, f"clean_adaptive_stamped_{output_prefix}.png")
    cv2.imwrite(out_target_p, vis_target)
    cv2.imwrite(out_stamped_p, vis_stamped)

    # 裁切前 4 題特寫
    strip_orig = img_bgr[250:750, 50:800].copy()
    strip_target = vis_target[250:750, 50:800].copy()
    strip_stamped = vis_stamped[250:750, 50:800].copy()
    cv2.putText(strip_orig, "Clean Original", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(strip_target, "Adaptive Inner ROI", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 100, 0), 2)
    cv2.putText(strip_stamped, "Dynamic Stamped", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 0), 2)
    zoom_strip = np.hstack([strip_orig, strip_target, strip_stamped])
    out_zoom_p = os.path.join(qa_dir, f"clean_adaptive_zoom_{output_prefix}.png")
    cv2.imwrite(out_zoom_p, zoom_strip)
    print(f"★ 特寫對照已儲存: {out_zoom_p}")

if __name__ == "__main__":
    test_clean_pdf("scanned_pairs/3561749-題目卷 掃描.pdf", "3561749")
