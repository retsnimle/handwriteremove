#!/usr/bin/env python3
"""
真實手寫筆跡加蓋驗證 V3 (大印章 + 濃郁真實墨水)
★ 徹底解決：
   1. 墨水顏色：拋棄原本素材的淡粉紅，全面渲染為濃郁飽滿的學生真實原子筆(深藍)與中性筆(純黑)！
   2. 印章尺寸：放大至真實學生作答比例 (高度 95%~115% 充滿括號甚至微貼邊緣，題號旁字母放大 1.2~1.4x)！
"""

import os
import sys
sys.path.insert(0, r"G:\CODE\handwriteremove")
import random
import glob
import cv2
import numpy as np
import fitz
from doclayout_yolo import YOLOv10
from tools.find_brackets_adaptive import find_brackets_in_roi
from tools.generate_qnum_stamp_preview import find_qnum_in_crop

HW_MARKS_DIR = r"G:\CODE\handwriteremove\assets\hw_marks"
all_mark_paths = sorted(glob.glob(os.path.join(HW_MARKS_DIR, "*.png")))

real_marks = []
for p in all_mark_paths:
    img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if img is not None and img.shape[2] == 4:
        alpha = img[:, :, 3]
        h, w = alpha.shape
        # 篩選非全空
        if 20 <= h <= 180 and 15 <= w <= 180 and np.count_nonzero(alpha > 30) > 80:
            real_marks.append(img)

print(f"📦 載入真實手寫骨架印章庫：共 {len(real_marks)} 個！")

def render_real_student_ink(raw_rgba):
    """
    將手寫素材的輪廓/筆壓骨架，渲染為真實學生常用的濃郁黑筆 / 藍筆
    """
    alpha = raw_rgba[:, :, 3].astype(np.float32) / 255.0
    h, w = alpha.shape
    
    # 增強筆墨濃度：原素材較淡，進行非線性伽瑪增強，讓筆芯主體飽滿，同時保留筆緣微毛細
    enhanced_alpha = np.power(alpha, 0.75) # 提亮筆芯
    
    # 決定筆種：65% 深藍原子筆，35% 黑色中性筆
    is_blue = random.random() < 0.65
    
    if is_blue:
        # 濃郁藍筆：藍色深邃，筆芯略深
        ink_b = random.randint(140, 190)
        ink_g = random.randint(45, 75)
        ink_r = random.randint(15, 35)
    else:
        # 濃郁黑筆 / 碳素墨水
        val = random.randint(20, 45)
        ink_b = ink_g = ink_r = val
        
    colored = np.zeros((h, w, 4), dtype=np.uint8)
    colored[:, :, 0] = ink_b
    colored[:, :, 1] = ink_g
    colored[:, :, 2] = ink_r
    # 筆墨 Alpha 提升至充沛飽和度 (0.92 ~ 1.0)
    final_alpha = np.clip(enhanced_alpha * random.uniform(0.92, 1.0) * 255.0, 0, 255).astype(np.uint8)
    colored[:, :, 3] = final_alpha
    return colored

def stamp_large_mark(canvas_bgr, raw_mark, target_roi, mode='bracket'):
    """
    以真實學生作答比例 (豪邁、大字、充滿括號或微交疊) 蓋印
    """
    mark_rgba = render_real_student_ink(raw_mark)
    tx1, ty1, tx2, ty2 = target_roi
    tw = max(10, tx2 - tx1)
    th = max(10, ty2 - ty1)
    
    mh, mw = mark_rgba.shape[:2]
    
    # 微旋轉 (手腕傾斜)
    angle = random.uniform(-10, 10)
    M_rot = cv2.getRotationMatrix2D((mw / 2, mh / 2), angle, 1.0)
    cos = np.abs(M_rot[0, 0])
    sin = np.abs(M_rot[0, 1])
    new_mw = int((mh * sin) + (mw * cos))
    new_mh = int((mh * cos) + (mw * sin))
    M_rot[0, 2] += (new_mw / 2) - mw / 2
    M_rot[1, 2] += (new_mh / 2) - mh / 2
    rotated = cv2.warpAffine(mark_rgba, M_rot, (new_mw, new_mh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    
    # ★ 關鍵調整：印章尺寸放大！
    if mode == 'bracket':
        # 括號作答：真實學生寫的字很大，高度填滿甚至略微突出括號 (95% ~ 115%)
        scale_h = (th * random.uniform(0.95, 1.15)) / new_mh
        scale_w = (tw * random.uniform(0.85, 1.05)) / new_mw
        scale = min(scale_h, scale_w) # 保持比例不變形太誇張
    elif mode == 'sat_margin':
        # 題號左側手寫答案：字型大於題號，顯眼豪邁 (1.2 ~ 1.4 倍)
        scale = (th * random.uniform(1.20, 1.45)) / new_mh
    else: # sat_overlay
        # 圈選或劃線：略大於題號 (1.3 ~ 1.6 倍)
        scale = max(tw / new_mw, th / new_mh) * random.uniform(1.25, 1.55)
        
    scaled_w = max(6, int(new_mw * scale))
    scaled_h = max(6, int(new_mh * scale))
    scaled_mark = cv2.resize(rotated, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
    
    # 放置坐標
    if mode == 'bracket':
        # 置中偏下 (學生重力落筆微靠底線)
        paste_x = tx1 + (tw - scaled_w) // 2 + random.randint(-1, 2)
        paste_y = ty1 + (th - scaled_h) // 2 + random.randint(0, 3)
    elif mode == 'sat_margin':
        paste_x = tx1 - scaled_w - random.randint(12, 25)
        paste_y = ty1 + (th - scaled_h) // 2 + random.randint(-2, 2)
    else:
        paste_x = tx1 + (tw - scaled_w) // 2 + random.randint(-2, 2)
        paste_y = ty1 + (th - scaled_h) // 2 + random.randint(-2, 2)
        
    cH, cW = canvas_bgr.shape[:2]
    x_start = max(0, paste_x)
    y_start = max(0, paste_y)
    x_end = min(cW, paste_x + scaled_w)
    y_end = min(cH, paste_y + scaled_h)
    
    if x_end <= x_start or y_end <= y_start:
        return canvas_bgr, np.zeros((cH, cW), dtype=np.uint8)
        
    src_x = x_start - paste_x
    src_y = y_start - paste_y
    src_w = x_end - x_start
    src_h = y_end - y_start
    
    mark_crop = scaled_mark[src_y:src_y+src_h, src_x:src_x+src_w]
    mark_bgr = mark_crop[:, :, :3]
    mark_alpha = (mark_crop[:, :, 3] / 255.0)[:, :, None]
    
    # 正片疊底物理咬入
    roi_bg = canvas_bgr[y_start:y_end, x_start:x_end].astype(np.float32)
    blended = roi_bg * (1.0 - mark_alpha) + (roi_bg * (mark_bgr.astype(np.float32) / 255.0)) * mark_alpha
    canvas_bgr[y_start:y_end, x_start:x_end] = np.clip(blended, 0, 255).astype(np.uint8)
    
    hw_mask = np.zeros((cH, cW), dtype=np.uint8)
    hw_mask[y_start:y_end, x_start:x_end] = (mark_crop[:, :, 3] > 35).astype(np.uint8) * 255
    
    return canvas_bgr, hw_mask

# ===========================
# 執行驗證生成
# ===========================
print("🚀 執行場景 1：括號考卷大印章濃墨蓋印...")
doc1 = fitz.open(r"G:\CODE\handwriteremove\scanned_pairs\3561749-題目卷 掃描.pdf")
pix1 = doc1[0].get_pixmap(dpi=150)
clean_bracket_img = cv2.imdecode(np.frombuffer(pix1.tobytes("png"), np.uint8), cv2.IMREAD_COLOR)
doc1.close()

model = YOLOv10(r"G:\CODE\handwriteremove\weights\doclayout_yolo_docstructbench_imgsz1024.pt")
det1 = model.predict(clean_bracket_img, imgsz=1024, conf=0.25, device="cuda:0")

boxes1 = det1[0].boxes
names = det1[0].names
plain_texts1 = []
for box in boxes1:
    if names[int(box.cls[0].item())] == 'plain text' and float(box.conf[0].item()) >= 0.35:
        plain_texts1.append(box.xyxy[0].cpu().numpy().astype(int))
plain_texts1.sort(key=lambda b: b[1])

bracket_results = []
for tx1, ty1, tx2, ty2 in plain_texts1:
    crop_h = min(45, ty2 - ty1)
    crop_w = min(160, tx2 - tx1)
    if crop_h < 15 or crop_w < 50:
        continue
    roi_bgr = clean_bracket_img[ty1:ty1+crop_h, tx1:tx1+crop_w]
    inner_roi, outer_bbox = find_brackets_in_roi(roi_bgr)
    if inner_roi is not None:
        ix1, iy1, ix2, iy2 = inner_roi
        abs_inner = (tx1 + ix1, ty1 + iy1, tx1 + ix2, ty1 + iy2)
        bracket_results.append(abs_inner)

stamped_bracket_img = clean_bracket_img.copy()
bracket_zooms = []

# 挑選有英文字母形態的手寫印章 (A, B, C, D)
sample_marks = [real_marks[i % len(real_marks)] for i in [4, 14, 8, 15]]

for i, inner in enumerate(bracket_results[:4]):
    mark = sample_marks[i]
    ix1, iy1, ix2, iy2 = inner
    cx1, cy1 = max(0, ix1 - 35), max(0, iy1 - 15)
    cx2, cy2 = min(clean_bracket_img.shape[1], ix2 + 130), min(clean_bracket_img.shape[0], iy2 + 20)
    
    crop_orig = clean_bracket_img[cy1:cy2, cx1:cx2].copy()
    stamped_bracket_img, cur_m = stamp_large_mark(stamped_bracket_img, mark, inner, mode='bracket')
    
    crop_stamped = stamped_bracket_img[cy1:cy2, cx1:cx2].copy()
    crop_mask = cv2.cvtColor(cur_m[cy1:cy2, cx1:cx2], cv2.COLOR_GRAY2BGR)
    
    hdr1 = np.full((22, crop_orig.shape[1], 3), 240, dtype=np.uint8)
    hdr2 = np.full((22, crop_orig.shape[1], 3), 230, dtype=np.uint8)
    hdr3 = np.full((22, crop_orig.shape[1], 3), 220, dtype=np.uint8)
    cv2.putText(hdr1, f"Clean Q{i+1}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60,60,60), 1)
    cv2.putText(hdr2, "Large Bold HW Ink", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20,20,120), 1)
    cv2.putText(hdr3, "Synth HW Mask", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0,100,0), 1)
    
    row = np.hstack([np.vstack([hdr1, crop_orig]), np.vstack([hdr2, crop_stamped]), np.vstack([hdr3, crop_mask])])
    bracket_zooms.append(row)

pad_w = max(r.shape[1] for r in bracket_zooms)
final_bracket_rows = []
for r in bracket_zooms:
    if r.shape[1] < pad_w:
        r = np.hstack([r, np.full((r.shape[0], pad_w - r.shape[1], 3), 255, dtype=np.uint8)])
    final_bracket_rows.append(r)
    final_bracket_rows.append(np.full((6, pad_w, 3), 200, dtype=np.uint8))
bracket_compare_img = np.vstack(final_bracket_rows[:-1])


# ===========================
# 場景 2：學測純題號大印章
# ===========================
print("🚀 執行場景 2：學測純題號大印章蓋印...")
doc2 = fitz.open(r"G:\CODE\handwriteremove\scanned_pairs\22-114學測自然試題定稿 掃描.pdf")
pix2 = doc2[1].get_pixmap(dpi=150)
clean_sat_img = cv2.imdecode(np.frombuffer(pix2.tobytes("png"), np.uint8), cv2.IMREAD_COLOR)
doc2.close()

H2, W2 = clean_sat_img.shape[:2]
det2 = model.predict(clean_sat_img, imgsz=1024, conf=0.25, device="cuda:0")
boxes2 = det2[0].boxes
plain_texts2 = []
for box in boxes2:
    if names[int(box.cls[0].item())] == 'plain text' and float(box.conf[0].item()) >= 0.3:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        if y1 >= 280 and y2 <= H2 - 80:
            plain_texts2.append((x1, y1, x2, y2))
            
plain_texts2.sort(key=lambda b: b[1])

qnum_results = []
for tx1, ty1, tx2, ty2 in plain_texts2:
    crop_h = min(50, ty2 - ty1)
    crop_w = min(150, tx2 - tx1)
    if crop_h < 15 or crop_w < 40:
        continue
    roi_bgr = clean_sat_img[ty1:ty1+crop_h, tx1:tx1+crop_w]
    q_rel = find_qnum_in_crop(roi_bgr)
    if q_rel is not None:
        qx, qy, qw, qh = q_rel
        qnum_results.append((tx1 + qx, ty1 + qy, qw, qh))

stamped_sat_img = clean_sat_img.copy()
sat_zooms = []
modes = ['sat_margin', 'sat_overlay', 'sat_overlay', 'sat_margin']

for i, (qx, qy, qw, qh) in enumerate(qnum_results[:4]):
    mark = sample_marks[i]
    mode = modes[i % len(modes)]
    
    cx1 = max(0, qx - 75)
    cy1 = max(0, qy - 20)
    cx2 = min(clean_sat_img.shape[1], qx + qw + 180)
    cy2 = min(clean_sat_img.shape[0], qy + qh + 28)
    
    crop_orig = clean_sat_img[cy1:cy2, cx1:cx2].copy()
    target_roi = (qx, qy, qx + qw, qy + qh)
    stamped_sat_img, cur_m = stamp_large_mark(stamped_sat_img, mark, target_roi, mode=mode)
    
    crop_stamped = stamped_sat_img[cy1:cy2, cx1:cx2].copy()
    crop_mask = cv2.cvtColor(cur_m[cy1:cy2, cx1:cx2], cv2.COLOR_GRAY2BGR)
    
    hdr1 = np.full((22, crop_orig.shape[1], 3), 240, dtype=np.uint8)
    hdr2 = np.full((22, crop_orig.shape[1], 3), 230, dtype=np.uint8)
    hdr3 = np.full((22, crop_orig.shape[1], 3), 220, dtype=np.uint8)
    cv2.putText(hdr1, f"Clean Q{i+1}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60,60,60), 1)
    cv2.putText(hdr2, f"Large Ink ({mode})", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20,20,120), 1)
    cv2.putText(hdr3, "Synth HW Mask", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0,100,0), 1)
    
    row = np.hstack([np.vstack([hdr1, crop_orig]), np.vstack([hdr2, crop_stamped]), np.vstack([hdr3, crop_mask])])
    sat_zooms.append(row)

pad_w2 = max(r.shape[1] for r in sat_zooms)
final_sat_rows = []
for r in sat_zooms:
    if r.shape[1] < pad_w2:
        r = np.hstack([r, np.full((r.shape[0], pad_w2 - r.shape[1], 3), 255, dtype=np.uint8)])
    final_sat_rows.append(r)
    final_sat_rows.append(np.full((6, pad_w2, 3), 200, dtype=np.uint8))
sat_compare_img = np.vstack(final_sat_rows[:-1])

# 儲存
OUT_DIR = r"G:\CODE\handwriteremove\output\qa"
ART_DIR = r"C:\Users\retsn\.gemini\antigravity\brain\f687af4b-740f-482b-a1f5-e84a253db0a3\qa"

cv2.imwrite(os.path.join(OUT_DIR, "real_handwriting_bracket_compare.png"), bracket_compare_img)
cv2.imwrite(os.path.join(ART_DIR, "real_handwriting_bracket_compare.png"), bracket_compare_img)

cv2.imwrite(os.path.join(OUT_DIR, "real_handwriting_sat_compare.png"), sat_compare_img)
cv2.imwrite(os.path.join(ART_DIR, "real_handwriting_sat_compare.png"), sat_compare_img)

print("🎉 濃墨大印章比對圖生成完成！")
