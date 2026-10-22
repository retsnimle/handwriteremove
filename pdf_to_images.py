#!/usr/bin/env python3
"""
簡單的 PDF -> 逐頁影像匯出工具
預設會讀取當前目錄下的 20251025101835564.pdf，並將每頁儲存到 output_images 資料夾。
使用 PyMuPDF (pymupdf) 來渲染頁面，避免需要系統層級的 poppler。
"""

import os
import sys
import argparse

try:
    import fitz  # PyMuPDF
except Exception as e:
    print("錯誤：無法 import fitz (pymupdf)。請先安裝：pip install pymupdf")
    raise


def pdf_to_images(input_pdf: str, out_dir: str, dpi: int = 150, fmt: str = "png") -> int:
    """把 input_pdf 每頁匯出成影像儲存在 out_dir，傳回產生的檔案數。"""
    if not os.path.isfile(input_pdf):
        raise FileNotFoundError(f"找不到 PDF 檔：{input_pdf}")

    os.makedirs(out_dir, exist_ok=True)

    doc = fitz.open(input_pdf)
    total = doc.page_count

    # scale factor: PyMuPDF uses 72 dpi as baseline
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    created = 0
    for i in range(total):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        filename = os.path.join(out_dir, f"page_{i+1:03d}.{fmt}")
        pix.save(filename)
        created += 1

    doc.close()
    return created


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="把 PDF 逐頁儲存成圖片。")
    parser.add_argument("pdf", nargs="?", default="20251020202938305.pdf",
                        help="要處理的 PDF 檔，預設為 20251020202938305.pdf")
    parser.add_argument("-o", "--out", default="output_images",
                        help="輸出目錄，預設 output_images")
    parser.add_argument("-d", "--dpi", type=int, default=150,
                        help="輸出影像 DPI，預設 150 (越大解析度越高)")
    parser.add_argument("-f", "--format", choices=["png", "jpg", "jpeg"], default="png",
                        help="輸出影像格式 (png/jpg)，預設 png")

    args = parser.parse_args()

    try:
        count = pdf_to_images(args.pdf, args.out, dpi=args.dpi, fmt=("jpg" if args.format in ("jpg","jpeg") else "png"))
        print(f"完成：產生 {count} 張影像，存在 {os.path.abspath(args.out)}")
        sys.exit(0)
    except Exception as e:
        print(f"處理失敗：{e}")
        sys.exit(2)
