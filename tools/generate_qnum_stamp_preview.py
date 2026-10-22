#!/usr/bin/env python3
"""
生成「無作答框、純題號考卷」的學生筆跡模擬與落點分析
適用情境：大考學測/指考/模考，題目卷無 (   ) 括號，學生直接在題號周遭作答或標記。
"""

import os
import random
import cv2
import numpy as np
import fitz
import torch
from doclayout_yolo import YOLOv10

MODEL_WEIGHTS = r"G:\CODE\handwriteremove\weights\doclayout_yolo_docstructbench_imgsz1024.pt"
PDF_PATH = r"G:\CODE\handwriteremove\scanned_pairs\22-114學測自然試題定稿 掃描.pdf"
OUTPUT_DIR = r"G:\CODE\handwriteremove\output\qa"
ARTIFACT_DIR = r"C:\Users\retsn\.gemini\antigravity\brain\f687af4b-740f-482b-a1f5-e84a253db0a3\qa"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def find_qnum_in_crop(crop_bgr):
    """
    在題幹開頭裁切圖中，自適應提取題號 (例如 '1.', '2.', '15.')
    回傳題號本體的 bbox: (qx, qy, qw, qh)
    """
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    _, bin_inv = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(bin_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    digit_candidates = []
    dot_candidates = []
    
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        # 排除整行雜訊或太大的區塊
        if h > 45 or w > 60:
            continue
        # 題號通常在題幹的最左側 (x < 120)
        if x > 120:
            continue
            
        # 可能是點 '.'
        if w <= 10 and h <= 10 and area >= 4:
            dot_candidates.append((x, y, w, h))
        # 可能是數字 '1', '2', '3'...
        elif 12 <= h <= 35 and 4 <= w <= 28:
            digit_candidates.append((x, y, w, h))
            
    if not digit_candidates:
        return None
        
    # 取最左邊的數字連通域
    digit_candidates.sort(key=lambda b: b[0])
    first_digit = digit_candidates[0]
    
    # 尋找緊跟在數字右側的點 '.'
    fx, fy, fw, fh = first_digit
    q_x0, q_y0, q_x1, q_y1 = fx, fy, fx + fw, fy + fh
    
    for dx, dy, dw, dh in dot_candidates:
        if 0 <= dx - q_x1 <= 15 and abs((dy + dh/2) - (fy + fh)) <= 15:
            q_x1 = max(q_x1, dx + dw)
            q_y1 = max(q_y1, dy + dh)
            break
            
    return (q_x0, q_y0, q_x1 - q_x0, q_y1 - q_y0)

def draw_handwritten_mark(canvas, mark_type, q_bbox, color=(35, 30, 20), thickness=2):
    """
    在畫布上根據題號坐標模擬學生的手寫筆跡
    mark_type:
      - 'answer_left': 在題號左側白邊寫手寫答案字母 (如 'B', 'C', 'D')
      - 'circle': 在題號本體外圈一個手寫圓圈/橢圓
      - 'check': 在題號本體上打勾
    """
    qx, qy, qw, qh = q_bbox
    cx = qx + qw // 2
    cy = qy + qh // 2
    
    if mark_type == 'answer_left':
        # 在題號左邊 25~45px 處寫手寫英文字母 (以幾何線條手繪風格呈現)
        ans_x = max(10, qx - 35 + random.randint(-4, 4))
        ans_y = qy + random.randint(-3, 3)
        choice = random.choice(['A', 'B', 'C', 'D'])
        # 繪製稍微歪斜、帶抖動的手寫風格字母
        font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX if random.random() > 0.5 else cv2.FONT_HERSHEY_PLAIN
        cv2.putText(canvas, choice, (ans_x, ans_y + qh - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, thickness, cv2.LINE_AA)
        
    elif mark_type == 'circle':
        # 圍繞題號畫一個手繪不規則橢圓
        axes = (int(qw * 0.9 + random.randint(4, 8)), int(qh * 0.75 + random.randint(4, 8)))
        angle = random.randint(-15, 15)
        # 用折線繪製略帶不規則感的手寫圓
        pts = cv2.ellipse2Poly((cx, cy), axes, angle, 0, 360, 15)
        jitter = np.random.randint(-1, 2, pts.shape)
        pts = pts + jitter
        cv2.polylines(canvas, [pts], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)
        
    elif mark_type == 'check':
        # 在題號上打勾 (V 字型手繪線)
        p1 = (qx - random.randint(4, 8), cy + random.randint(0, 4))
        p2 = (cx, qy + qh + random.randint(2, 6))
        p3 = (qx + qw + random.randint(8, 14), qy - random.randint(4, 10))
        cv2.line(canvas, p1, p2, color, thickness, cv2.LINE_AA)
        cv2.line(canvas, p2, p3, color, thickness, cv2.LINE_AA)

def main():
    print("🚀 載入乾淨學測題目卷 (無作答框，僅有題號)...")
    doc = fitz.open(PDF_PATH)
    page = doc[1] # 第 2 頁 (第 1~5 題)
    pix = page.get_pixmap(dpi=150)
    img_bgr = cv2.imdecode(np.frombuffer(pix.tobytes("png"), np.uint8), cv2.IMREAD_COLOR)
    doc.close()
    
    H, W, _ = img_bgr.shape
    
    # 1. GPU DocLayout-YOLO 檢測版面
    print("🔍 執行 DocLayout-YOLO (純 GPU)...")
    model = YOLOv10(MODEL_WEIGHTS)
    det_res = model.predict(img_bgr, imgsz=1024, conf=0.25, device="cuda:0")
    boxes = det_res[0].boxes
    names = det_res[0].names
    
    plain_texts = []
    for box in boxes:
        cls_name = names[int(box.cls[0].item())]
        conf = float(box.conf[0].item())
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        if cls_name == "plain text" and conf >= 0.3:
            plain_texts.append((x1, y1, x2, y2))
            
    plain_texts.sort(key=lambda b: b[1])
    
    # 建立可視化畫布
    vis_target = img_bgr.copy() # 熱區展示
    vis_stamped = img_bgr.copy() # 模擬蓋上手寫筆跡
    
    question_records = []
    
    # 逐一在題幹題頭尋找題號
    for idx, (tx1, ty1, tx2, ty2) in enumerate(plain_texts):
        # 題頭取高 50px, 寬 140px
        crop_h = min(50, ty2 - ty1)
        crop_w = min(150, tx2 - tx1)
        if crop_h < 15 or crop_w < 40:
            continue
            
        crop_roi = img_bgr[ty1:ty1+crop_h, tx1:tx1+crop_w]
        q_rel = find_qnum_in_crop(crop_roi)
        
        if q_rel is not None:
            qx, qy, qw, qh = q_rel
            abs_qx = tx1 + qx
            abs_qy = ty1 + qy
            abs_qbbox = (abs_qx, abs_qy, qw, qh)
            question_records.append((idx, abs_qbbox, (tx1, ty1, tx2, ty2)))
            
    print(f"✅ 成功自適應定位到 {len(question_records)} 個題號！")
    
    # 挑選前 4 個題號做三種經典手寫情境特寫展示
    zoom_slices = []
    mark_modes = ['answer_left', 'circle', 'check', 'answer_left']
    
    for i, (q_idx, (qx, qy, qw, qh), block) in enumerate(question_records[:4]):
        mode = mark_modes[i % len(mark_modes)]
        
        # 裁切該題題頭 (包含左側邊界，呈現學生在空白處寫字的效果)
        cx1 = max(0, qx - 65)
        cy1 = max(0, qy - 18)
        cx2 = min(W, qx + qw + 200)
        cy2 = min(H, qy + qh + 25)
        
        orig_crop = img_bgr[cy1:cy2, cx1:cx2].copy()
        
        # 1. 熱區可視化圖 (標出題號與手寫筆跡預定著陸區)
        roi_crop = orig_crop.copy()
        # 綠框：印刷題號本體 (受到 Text-Aware 嚴格保護，絕不可被當成筆跡誤殺)
        rel_qx = qx - cx1
        rel_qy = qy - cy1
        cv2.rectangle(roi_crop, (rel_qx, rel_qy), (rel_qx + qw, rel_qy + qh), (0, 200, 0), 2)
        
        # 藍色透明框：筆跡預計落點熱區
        overlay = roi_crop.copy()
        if mode == 'answer_left':
            cv2.rectangle(overlay, (rel_qx - 45, rel_qy - 5), (rel_qx - 5, rel_qy + qh + 5), (255, 120, 0), -1)
            cv2.addWeighted(overlay, 0.4, roi_crop, 0.6, 0, roi_crop)
            cv2.putText(roi_crop, "[Left Answer Zone]", (rel_qx - 55, rel_qy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 80, 0), 1)
        elif mode == 'circle':
            cv2.ellipse(overlay, (rel_qx + qw//2, rel_qy + qh//2), (int(qw*0.9), int(qh*0.8)), 0, 0, 360, (0, 120, 255), -1)
            cv2.addWeighted(overlay, 0.4, roi_crop, 0.6, 0, roi_crop)
            cv2.putText(roi_crop, "[Circle Question Zone]", (rel_qx - 10, rel_qy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 80, 255), 1)
        elif mode == 'check':
            cv2.rectangle(overlay, (rel_qx - 10, rel_qy - 10), (rel_qx + qw + 15, rel_qy + qh + 10), (200, 0, 200), -1)
            cv2.addWeighted(overlay, 0.4, roi_crop, 0.6, 0, roi_crop)
            cv2.putText(roi_crop, "[Checkmark Zone]", (rel_qx - 10, rel_qy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 0, 180), 1)
            
        # 2. 模擬蓋上手寫筆跡圖
        stamp_crop = orig_crop.copy()
        draw_handwritten_mark(stamp_crop, mode, (rel_qx, rel_qy, qw, qh), color=(30, 25, 20), thickness=2)
        
        # 加上標題欄
        pad_h = 24
        header_orig = np.full((pad_h, orig_crop.shape[1], 3), 245, dtype=np.uint8)
        header_roi = np.full((pad_h, orig_crop.shape[1], 3), 235, dtype=np.uint8)
        header_stamp = np.full((pad_h, orig_crop.shape[1], 3), 225, dtype=np.uint8)
        
        cv2.putText(header_orig, f"Clean Original (Q{i+1})", (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1)
        cv2.putText(header_roi, f"Target Zone ({mode})", (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 50, 0), 1)
        cv2.putText(header_stamp, f"Simulated Student Mark", (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 120), 1)
        
        col1 = np.vstack([header_orig, orig_crop])
        col2 = np.vstack([header_roi, roi_crop])
        col3 = np.vstack([header_stamp, stamp_crop])
        
        sep = np.full((col1.shape[0], 6, 3), 210, dtype=np.uint8)
        row = np.hstack([col1, sep, col2, sep, col3])
        zoom_slices.append(row)
        
    # 垂直拼接所有展示條
    max_w = max(r.shape[1] for r in zoom_slices)
    padded_rows = []
    for r in zoom_slices:
        if r.shape[1] < max_w:
            pad = np.full((r.shape[0], max_w - r.shape[1], 3), 255, dtype=np.uint8)
            r = np.hstack([r, pad])
        padded_rows.append(r)
        # 行間分隔
        h_sep = np.full((8, max_w, 3), 200, dtype=np.uint8)
        padded_rows.append(h_sep)
        
    combined_zoom = np.vstack(padded_rows[:-1])
    
    out_zoom = os.path.join(OUTPUT_DIR, "clean_sat_qnum_marks_zoom.png")
    cv2.imwrite(out_zoom, combined_zoom)
    
    art_zoom = os.path.join(ARTIFACT_DIR, "clean_sat_qnum_marks_zoom.png")
    cv2.imwrite(art_zoom, combined_zoom)
    
    print(f"🎉 產出完成！已儲存至:\n  -> {out_zoom}\n  -> {art_zoom}")

if __name__ == "__main__":
    main()
