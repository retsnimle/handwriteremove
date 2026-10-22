#!/usr/bin/env python3
"""
全量考卷作答錨點離線建立器 V3 (嚴格排除選項與漢字，僅鎖定空白作答框與真實阿拉伯數字題號)
★ 修正核心：
   1. 嚴格過濾選項：檢查括號內膽內部，若包含印刷文字 (如選項 (A)、(B)) 100% 剔除！僅保留內部空白的真實作答括號！
   2. 嚴格鎖定真實題號：
      - 題幹最左側字符必須是瘦長的阿拉伯數字 (w: 4~14, h: 12~28)
      - 後方必須緊鄰點 '.' (小於 8px)
      - 徹底杜絕將「野外調查時...」、「說明：...」等中文漢字段落誤判為題號！
"""

import os
import sys
sys.path.insert(0, r"G:\CODE\handwriteremove")
import json
import time
import cv2
import numpy as np
import fitz
import torch
from doclayout_yolo import YOLOv10
from tools.find_brackets_adaptive import find_brackets_in_roi

DPI = 150
MODEL_WEIGHTS = r"G:\CODE\handwriteremove\weights\doclayout_yolo_docstructbench_imgsz1024.pt"
SCANNED_DIR = r"G:\CODE\handwriteremove\scanned_pairs"
OUTPUT_CATALOG = r"G:\CODE\handwriteremove\assets\exam_anchors_catalog.json"

def find_strict_qnum_in_roi(crop_bgr):
    """
    嚴格題號檢測器：必須是瘦長阿拉伯數字 + 緊鄰的點 '.'
    """
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    _, bin_inv = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(bin_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours]
    
    items = []
    for b in boxes:
        x, y, w, h = b
        if x < 100 and 2 <= w <= 40 and 2 <= h <= 40:
            items.append(b)
            
    if not items:
        return None
        
    items.sort(key=lambda b: b[0])
    main_items = [b for b in items if b[2] >= 4 and b[3] >= 10]
    if not main_items:
        return None
        
    first_item = main_items[0]
    fx, fy, fw, fh = first_item
    
    # 阿拉伯數字特徵：瘦長，寬度 4~14，高度 12~28，且高度大於寬度
    if not (4 <= fw <= 14 and 12 <= fh <= 28 and fh >= fw * 1.05):
        return None
        
    # 支援兩位數字 (例如 24.)
    second_digit = None
    if len(main_items) > 1:
        cand2 = main_items[1]
        if 0 <= cand2[0] - (fx + fw) <= 6 and abs(cand2[1] - fy) <= 6:
            if 4 <= cand2[2] <= 14 and 12 <= cand2[3] <= 28:
                second_digit = cand2
                
    num_right = (second_digit[0] + second_digit[2]) if second_digit else (fx + fw)
    num_bottom = max(fy + fh, (second_digit[1] + second_digit[3]) if second_digit else (fy + fh))
    
    # 尋找緊鄰數字右下方的點 '.'
    dots = [b for b in items if b[2] <= 8 and b[3] <= 8 and b[2] * b[3] >= 4]
    matched_dot = None
    for dot in dots:
        dx, dy, dw, dh = dot
        if 1 <= dx - num_right <= 15 and abs((dy + dh/2) - num_bottom) <= 12:
            matched_dot = dot
            break
            
    if matched_dot is None:
        return None
        
    qx0 = fx
    qy0 = min(fy, matched_dot[1])
    qx1 = matched_dot[0] + matched_dot[2]
    qy1 = max(num_bottom, matched_dot[1] + matched_dot[3])
    
    return (qx0, qy0, qx1 - qx0, qy1 - qy0)

def process_all_pages():
    print(f"🚀 初始化 DocLayout-YOLO 模型 (CUDA:0)...")
    model = YOLOv10(MODEL_WEIGHTS)
    
    with open(r"G:\CODE\handwriteremove\assets\scanned_catalog.json", "r", encoding="utf-8") as f:
        catalog = json.load(f)
        
    print(f"📄 開始嚴格掃描 {len(catalog)} 頁考卷底稿 (排除選項，排除漢字，僅鎖定空白作答框與真實阿拉伯數字題號)...")
    start_time = time.time()
    
    results = []
    total_brackets = 0
    total_qnums = 0
    
    current_doc = None
    current_pdf_name = ""
    
    for idx, item in enumerate(catalog):
        pdf_name = item["pdf"]
        page_num = item["page"]
        
        if pdf_name != current_pdf_name:
            if current_doc is not None:
                current_doc.close()
            pdf_path = os.path.join(SCANNED_DIR, pdf_name)
            current_doc = fitz.open(pdf_path)
            current_pdf_name = pdf_name
            
        page = current_doc[page_num]
        pix = page.get_pixmap(dpi=DPI)
        img_bgr = cv2.imdecode(np.frombuffer(pix.tobytes("png"), np.uint8), cv2.IMREAD_COLOR)
        H, W = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # 1. GPU YOLO 佈局檢測
        det = model.predict(img_bgr, imgsz=1024, conf=0.25, device="cuda:0", verbose=False)
        boxes = det[0].boxes
        names = det[0].names
        
        plain_texts = []
        for box in boxes:
            cls_name = names[int(box.cls[0].item())]
            conf = float(box.conf[0].item())
            if cls_name == "plain text" and conf >= 0.30:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                if y1 >= 60 and y2 <= H - 50 and (y2 - y1) >= 15:
                    plain_texts.append((x1, y1, x2, y2))
                    
        plain_texts.sort(key=lambda b: b[1])
        page_anchors = []
        
        # 2. 針對每一題幹尋找錨點
        for tx1, ty1, tx2, ty2 in plain_texts:
            crop_h = min(50, ty2 - ty1)
            crop_w = min(160, tx2 - tx1)
            if crop_h < 15 or crop_w < 40:
                continue
            crop_roi = img_bgr[ty1:ty1+crop_h, tx1:tx1+crop_w]
            
            # 尋找空白作答括號
            inner_roi, outer_bbox = find_brackets_in_roi(crop_roi)
            if inner_roi is not None:
                ix1, iy1, ix2, iy2 = inner_roi
                ox1, oy1, ox2, oy2 = outer_bbox
                abs_ix1, abs_iy1, abs_ix2, abs_iy2 = tx1 + ix1, ty1 + iy1, tx1 + ix2, ty1 + iy2
                
                inner_patch = gray[abs_iy1:abs_iy2, abs_ix1:abs_ix2]
                dark_pixels = np.count_nonzero(inner_patch < 160)
                inner_w = abs_ix2 - abs_ix1
                
                if inner_w >= 18 and dark_pixels < 12:
                    page_anchors.append({
                        "type": "bracket",
                        "inner_bbox": [int(abs_ix1), int(abs_iy1), int(abs_ix2), int(abs_iy2)],
                        "outer_bbox": [int(tx1 + ox1), int(ty1 + oy1), int(tx1 + ox2), int(ty1 + oy2)]
                    })
                    total_brackets += 1
            else:
                # 嚴格尋找阿拉伯數字題號 (必須是瘦長數字且緊鄰小點 '.')
                q_rel = find_strict_qnum_in_roi(crop_roi)
                if q_rel is not None:
                    qx, qy, qw, qh = q_rel
                    page_anchors.append({
                        "type": "qnum",
                        "bbox": [int(tx1 + qx), int(ty1 + qy), int(tx1 + qx + qw), int(ty1 + qy + qh)]
                    })
                    total_qnums += 1
                    
        results.append({
            "pdf": pdf_name,
            "page": page_num,
            "w": W,
            "h": H,
            "anchors": page_anchors
        })
        
        if (idx + 1) % 25 == 0 or idx == len(catalog) - 1:
            elapsed = time.time() - start_time
            print(f"[{idx+1}/{len(catalog)}] 已處理 {idx+1} 頁 (耗時: {elapsed:.1f}s, 累計純淨作答括號={total_brackets}, 真實題號={total_qnums})")
            
    if current_doc is not None:
        current_doc.close()
        
    with open(OUTPUT_CATALOG, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    elapsed = time.time() - start_time
    print(f"\n🎉 錨點重新生成完成！總耗時: {elapsed:.1f}s")
    print(f"📊 最終統計：總頁數 {len(results)}，純淨空白作答括號 {total_brackets} 個，嚴格阿拉伯數字題號 {total_qnums} 個！")

if __name__ == "__main__":
    process_all_pages()
