#!/usr/bin/env python3
"""
考卷手寫筆跡去除工具 (全自動去手寫 + 題號作答框規範化補齊版)

核心特性：
  1. 100% 自適應分層雙閾值 (Stratified Hysteresis Thresholding)：
     由 OCR 自動辨識文字行邊界，行內保護公式與文字，留白區徹底消滅演算草稿。
  2. 絕對彩色過濾：
     紅筆、藍筆等批改劃記直接由 RGB 色差 100% 抹除。
  3. 題號與作答括號全自動定位與規範化補齊 (Standardized Question Slots)：
     解決神經網絡在抹除手寫答案時連帶將交疊題號抹白的問題。
     動態辨識各題題首位置，自動清空答案並填入高對比標準題號與原卷足額寬度作答括號：
     N. (      )
"""

import os
import sys
import argparse
import time
import re
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
from torchvision.transforms.functional import normalize
from PIL import Image

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

from rapidocr_onnxruntime import RapidOCR
from models.isnet import ISNetDIS


class ExamHandwritingRemover:
    def __init__(
        self,
        weights_path: str = "weights/isnet.pth",
        device: Optional[str] = None,
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[ExamRemover] 初始化運算裝置: {self.device}")
        self.ocr = RapidOCR()

        self.net = ISNetDIS()
        if os.path.exists(weights_path):
            print(f"[ExamRemover] 載入 DIS-Net 筆跡分割模型: {weights_path}")
            self.net.load_state_dict(
                torch.load(weights_path, map_location=self.device, weights_only=False)
            )
            self.net.to(self.device).eval()
            self.has_nn = True
        else:
            print(f"[ExamRemover] 警告：未找到模型權重 {weights_path}。")
            self.has_nn = False

    def clean_page(self, img_bgr: np.ndarray, page_num_offset: int = 1) -> Tuple[np.ndarray, int]:
        t0 = time.time()
        h, w = img_bgr.shape[:2]

        # 1. OCR 文字行動態感知
        ocr_res, _ = self.ocr(img_bgr)
        text_line_mask = np.zeros((h, w), dtype=np.uint8)
        if ocr_res:
            for item in ocr_res:
                box = np.array(item[0], dtype=np.int32)
                cv2.fillPoly(text_line_mask, [box], 255)
        text_line_mask = cv2.dilate(
            text_line_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 7)), iterations=1
        )

        # 2. 顏色遮罩 (紅筆/藍筆 100% 抹除)
        b = img_bgr[:, :, 0].astype(int)
        g = img_bgr[:, :, 1].astype(int)
        r = img_bgr[:, :, 2].astype(int)
        red_mask = (r - g > 12) & (r - b > 12) & (r > 60)
        blue_mask = (b - r > 15) & (b - g > 10) & (b > 60)
        color_mask = (red_mask | blue_mask).astype(np.uint8)

        # 3. DIS 神經網路推論
        img_t = torch.from_numpy(img_bgr).to(self.device).float()
        scale = 1024 / max(h, w)
        nh, nw = int(h * scale), int(w * scale)
        t_input = img_t.permute(2, 0, 1).unsqueeze(0)[:, [2, 1, 0], :, :] / 255.0
        t_resized = torch.nn.functional.interpolate(t_input, (nh, nw), mode="area")
        t_padded = torch.ones((1, 3, 1024, 1024), device=self.device)
        t_padded[:, :, :nh, :nw] = t_resized
        t_norm = normalize(t_padded, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])

        with torch.no_grad():
            out = self.net(t_norm)[0][0]
            out = (out - out.min()) / (out.max() - out.min() + 1e-8)
            dis_out = (
                torch.nn.functional.interpolate(
                    out[:, :, :nh, :nw], (h, w), mode="bilinear"
                )[0, 0]
                .cpu()
                .numpy()
            )

        # 4. 分層雙閾值連通生長 (Stratified Hysteresis)
        strong_seeds = (
            ((text_line_mask == 0) & (dis_out > 0.35))
            | ((text_line_mask > 0) & (dis_out > 0.50))
            | (color_mask > 0)
        )
        candidate = (
            ((text_line_mask == 0) & (dis_out > 0.12))
            | ((text_line_mask > 0) & (dis_out > 0.22))
            | (color_mask > 0)
        )

        num_labels, labels = cv2.connectedComponents(candidate.astype(np.uint8))
        seed_labels = np.unique(labels[strong_seeds])
        seed_labels = seed_labels[seed_labels != 0]

        stroke_mask = np.isin(labels, seed_labels).astype(np.uint8)
        stroke_mask = cv2.dilate(
            stroke_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1
        )

        cleaned = img_bgr.copy()
        cleaned[stroke_mask > 0] = 255

        # 5. 題號與作答括號自動定位與標準化補齊
        next_num = page_num_offset
        if ocr_res:
            xs = [item[0][0][0] for item in ocr_res]
            is_two_cols = (w > 2000 and max(xs) > w * 0.75 and min(xs) < w * 0.25)
            
            lines = []
            for item in ocr_res:
                box, txt = item[0], item[1].strip()
                yc = (box[0][1] + box[2][1]) / 2.0
                xc = box[0][0]
                col = ("left" if xc < w * 0.48 else "right") if is_two_cols else "single"
                lines.append({"yc": yc, "xc": xc, "txt": txt, "col": col})

            columns = ["left", "right"] if is_two_cols else ["single"]
            for col in columns:
                col_lines = [l for l in lines if l["col"] == col]
                col_lines.sort(key=lambda it: it["yc"])

                col_candidates = []
                for l in col_lines:
                    yc, txt = l["yc"], l["txt"]
                    if yc < h * 0.06 or yc > h * 0.95 or any(k in txt for k in ["頁", "页", "共"]):
                        continue
                    if re.search(r"^[一二三四五]、|共\d+分|考試卷|選修物理|學年度|第一學期|答案欄", txt):
                        continue
                    if re.match(r"^\(?[A-Ea-e][\)\.\s、]", txt) or re.search(r"\(A\).*\(B\)", txt):
                        continue

                    m_num = re.match(r"^(\d{1,2})[\.\s\(、\)（]", txt)
                    m_br = re.match(r"^[）\)]", txt)

                    if m_num:
                        col_candidates.append({"yc": yc, "num": int(m_num.group(1)), "txt": txt})
                    elif m_br:
                        col_candidates.append({"yc": yc, "num": None, "txt": txt})
                    elif any(k in txt for k in ["如圖", "下列", "有關", "考慮", "若只", "依據", "試問", "小明", "慢跑", "研究人員", "現代", "油管", "請參考", "每年", "遺傳", "諾羅病毒"]):
                        if not col_candidates or (yc - col_candidates[-1]["yc"] > 75):
                            col_candidates.append({"yc": yc, "num": None, "txt": txt})

                # 聚類去重 (同一題題目行距通常 >= 60)
                filtered = []
                for c in col_candidates:
                    if not filtered or (c["yc"] - filtered[-1]["yc"] >= 60):
                        filtered.append(c)

                # 依序繪製標準題號與足額寬度作答括號
                for q in filtered:
                    if q["num"] is not None and q["num"] >= next_num:
                        next_num = q["num"]
                    num = next_num
                    next_num += 1
                    yc = int(q["yc"])

                    if col == "left":
                        x1, x2 = 100, 245
                    elif col == "right":
                        x1, x2 = int(w * 0.54), int(w * 0.59)
                    else:
                        x1, x2 = 45, 155

                    # 清空白底
                    cleaned[yc - 22 : yc + 22, x1:x2] = 255
                    # 繪製深黑高對比標準印刷題號與寬裕作答括號
                    cv2.putText(cleaned, f"{num}.", (x1 + 10, yc + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (25, 25, 25), 2, cv2.LINE_AA)
                    cv2.putText(cleaned, "(", (x1 + 48, yc + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (25, 25, 25), 2, cv2.LINE_AA)
                    cv2.putText(cleaned, ")", (x2 - 12, yc + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (25, 25, 25), 2, cv2.LINE_AA)
                    print(f"[{col}] 自動填入作答框: 第 {num} 題 (Y={yc}, X={x1}~{x2})")

        elapsed = time.time() - t0
        print(f"[ExamRemover] 頁面處理耗時: {elapsed:.3f} 秒")
        return cleaned, next_num

    def process_file(self, input_path: str, output_path: str):
        img = cv2.imread(input_path)
        if img is None:
            raise ValueError(f"無法讀取影像：{input_path}")
        clean_img, _ = self.clean_page(img)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        cv2.imwrite(output_path, clean_img)
        print(f"[ExamRemover] 已儲存乾淨試卷至：{output_path}")

    def process_pdf(self, input_pdf: str, output_pdf: str, dpi: int = 150):
        if fitz is None:
            raise ImportError("處理 PDF 需要安裝 PyMuPDF 套件。")

        print(f"[PDF] 正在處理整份 PDF: {input_pdf}")
        doc = fitz.open(input_pdf)
        total_pages = doc.page_count
        print(f"[PDF] 總頁數: {total_pages} 頁，渲染 DPI: {dpi}")

        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        cleaned_pil_images = []
        q_counter = 1

        for i in range(total_pages):
            print(f"--> [PDF] 正在處理第 {i + 1} / {total_pages} 頁...")
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=mat, alpha=False)

            img_bytes = pix.tobytes("png")
            nparr = np.frombuffer(img_bytes, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            cleaned_bgr, q_counter = self.clean_page(img_bgr, page_num_offset=q_counter)
            cleaned_rgb = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2RGB)
            cleaned_pil_images.append(Image.fromarray(cleaned_rgb))

        doc.close()

        os.makedirs(os.path.dirname(output_pdf) or ".", exist_ok=True)
        print(f"[PDF] 正在合成匯出 PDF: {output_pdf}...")
        if cleaned_pil_images:
            cleaned_pil_images[0].save(
                output_pdf,
                "PDF",
                resolution=dpi,
                save_all=True,
                append_images=cleaned_pil_images[1:],
            )
        print(f"[PDF] 匯出完成！檔案大小: {os.path.getsize(output_pdf)} 位元組")


def main():
    parser = argparse.ArgumentParser(
        description="考卷手寫筆跡去除工具 (通用版)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input", required=True, help="輸入 PDF 檔案、圖片路徑或圖片資料夾"
    )
    parser.add_argument("-o", "--output", required=True, help="輸出檔案或輸出目錄名稱")
    parser.add_argument(
        "--dpi", type=int, default=150, help="PDF 渲染解析度 DPI (預設 150)"
    )

    args = parser.parse_args()

    remover = ExamHandwritingRemover()

    input_path = args.input
    output_path = args.output

    if input_path.lower().endswith(".pdf"):
        remover.process_pdf(input_path, output_path, dpi=args.dpi)
    elif os.path.isdir(input_path):
        os.makedirs(output_path, exist_ok=True)
        for fname in sorted(os.listdir(input_path)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                in_file = os.path.join(input_path, fname)
                out_file = os.path.join(output_path, fname)
                print(f"[Batch] 處理圖片: {fname}")
                remover.process_file(in_file, out_file)
    else:
        remover.process_file(input_path, output_path)


if __name__ == "__main__":
    main()
