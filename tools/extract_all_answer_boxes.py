#!/usr/bin/env python3
"""
全純 GPU 加速之 106 頁臺灣考卷作答框與題號精確萃取器
(GPU Accelerated Exam Answer Boxes Extractor)
使用 DirectML GPU 原生加速 RapidOCR，精確掃描 106 頁底稿中所有的「題號 + 作答括號」坐標，
並計算括號內部的「作答內膽區間 (Inner Box ROI)」，儲存為結構化目錄供動態訓練蓋印章！
"""

import os
import sys
import glob
import json
import time
import re
import cv2
import numpy as np
import fitz
from rapidocr_onnxruntime import RapidOCR

def extract_boxes():
    print("=== [GPU Accelerated] 啟動 106 頁考卷作答框自動化萃取 ===")
    
    # 啟用純 GPU (DirectML) 推論後端
    ocr = RapidOCR(Det_use_dml=True, Rec_use_dml=True)
    print("★ RapidOCR DirectML GPU 後端已就緒！")

    scanned_dir = "scanned_pairs"
    pdf_files = sorted(glob.glob(os.path.join(scanned_dir, "*.pdf")))
    print(f"找到 {len(pdf_files)} 個掃描 PDF 檔案。")

    catalog = []
    qa_dir = "output/qa"
    os.makedirs(qa_dir, exist_ok=True)
    os.makedirs("assets", exist_ok=True)

    total_pages = 0
    total_boxes = 0
    t0_all = time.time()

    # 正則表達式：尋找包含題號或括號的特徵
    # 例如: "1.(", "2. (", "3.", "( )", "（ ）", "5.(A)", "）如圖", "）有關" 等
    q_pattern = re.compile(r"(\d+[\.、]?\s*[\(（]|[\(（]\s*[\)）]|[\(（][A-Za-z0-9\s]*[\)）]|^\d+[\.、]\s*)")

    for pdf_path in pdf_files:
        pdf_name = os.path.basename(pdf_path)
        doc = fitz.open(pdf_path)
        n_pages = len(doc)
        print(f"\n📄 處理文件: {pdf_name} ({n_pages} 頁)...")

        for pi in range(n_pages):
            total_pages += 1
            t_page0 = time.time()
            page = doc[pi]
            pix = page.get_pixmap(dpi=150)
            img_bgr = cv2.imdecode(np.frombuffer(pix.tobytes("png"), np.uint8), cv2.IMREAD_COLOR)
            H, W, _ = img_bgr.shape

            ocr_results, _ = ocr(img_bgr)
            page_boxes = []

            if ocr_results:
                for item in ocr_results:
                    bbox, text, score = item
                    pts = np.array(bbox, dtype=np.int32)
                    x_min, y_min = pts.min(axis=0)
                    x_max, y_max = pts.max(axis=0)
                    box_w = x_max - x_min
                    box_h = y_max - y_min

                    # 條件 1: 符合正則特徵且長度短小（作答框通常寬度 < 160px，高度 15~45px）
                    is_bracket_text = bool(q_pattern.search(text))
                    
                    # 條件 2: 或是正文緊鄰開頭有右括號（例如 "）如圖所示"）
                    is_prompt_start = text.startswith("）") or text.startswith(")")

                    if is_bracket_text and box_w < 200 and box_h < 60:
                        # 這是標準的題號作答框
                        inner_x1 = int(x_min + box_w * 0.35)
                        inner_x2 = int(x_max - box_w * 0.15)
                        inner_y1 = int(y_min + 2)
                        inner_y2 = int(y_max - 2)
                        page_boxes.append({
                            "type": "direct_bracket",
                            "bbox": [int(x_min), int(y_min), int(x_max), int(y_max)],
                            "inner_roi": [inner_x1, inner_y1, inner_x2, inner_y2],
                            "text": text,
                            "score": float(score)
                        })
                    elif is_prompt_start:
                        # 這是被手寫字阻隔的正文起點，往左推算作答框
                        est_x1 = max(0, x_min - 110)
                        est_x2 = x_min + 15
                        est_y1 = max(0, y_min - 4)
                        est_y2 = min(H, y_max + 4)
                        inner_x1 = est_x1 + 35
                        inner_x2 = est_x2 - 15
                        page_boxes.append({
                            "type": "anchor_inferred",
                            "bbox": [est_x1, est_y1, est_x2, est_y2],
                            "inner_roi": [inner_x1, est_y1 + 2, inner_x2, est_y2 - 2],
                            "text": text[:8],
                            "score": float(score)
                        })

            # 過濾重疊框
            filtered_boxes = []
            for b in page_boxes:
                bx1, by1, bx2, by2 = b["bbox"]
                overlap = False
                for fb in filtered_boxes:
                    fx1, fy1, fx2, fy2 = fb["bbox"]
                    # 若兩框中心極近
                    if abs((bx1+bx2)/2 - (fx1+fx2)/2) < 30 and abs((by1+by2)/2 - (fy1+fy2)/2) < 20:
                        overlap = True
                        break
                if not overlap:
                    filtered_boxes.append(b)

            catalog.append({
                "pdf": pdf_name,
                "page": pi,
                "width": W,
                "height": H,
                "boxes": filtered_boxes
            })

            total_boxes += len(filtered_boxes)
            elapsed = time.time() - t_page0
            if (pi + 1) % 10 == 0 or pi == 0 or pi == n_pages - 1:
                print(f"  [頁 {pi+1:02d}/{n_pages:02d}] 抓取 {len(filtered_boxes):02d} 個作答框 (耗時: {elapsed:.2f}s)")

            # 抽樣視覺化第一頁
            if pi == 0:
                vis = img_bgr.copy()
                for b in filtered_boxes:
                    bx1, by1, bx2, by2 = b["bbox"]
                    ix1, iy1, ix2, iy2 = b["inner_roi"]
                    cv2.rectangle(vis, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
                    cv2.rectangle(vis, (ix1, iy1), (ix2, iy2), (0, 255, 0), 2)
                    cv2.putText(vis, b["text"][:6], (bx1, max(15, by1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                safe_name = pdf_name.replace(".pdf", "").replace(" ", "_")
                cv2.imwrite(os.path.join(qa_dir, f"extract_boxes_vis_{safe_name}_p01.png"), vis)

        doc.close()

    total_time = time.time() - t0_all
    out_json = "assets/answer_boxes_catalog.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)

    print("\n=======================================================")
    print(f"🎉 106 頁作答框全數萃取完成！")
    print(f"   總計處理頁數: {total_pages} 頁")
    print(f"   總計鎖定作答框: {total_boxes} 個 (平均每頁 {total_boxes/total_pages:.1f} 題)")
    print(f"   總耗時: {total_time:.2f} 秒 (平均單頁: {total_time/total_pages:.2f} 秒)")
    print(f"   結構化目錄已儲存至: {out_json}")
    print("=======================================================")

if __name__ == "__main__":
    extract_boxes()
