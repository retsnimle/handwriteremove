#!/usr/bin/env python3
"""
純淨考卷版面結構預解析器 (Layout Catalog Builder)

功能：
  遍歷 clean_pdfs/ 下的所有純淨試卷 PDF (353 頁)，
  利用 PyMuPDF 結構化解析每一頁的：
    1. 括號作答區 (brackets): 題號作答括號
    2. 選項標籤 (options): (A) ~ (E) 等選項位置
    3. 題幹文字行 (text_lines): 印刷文字所在區域 (供刪除線、塗改壓字使用)
    4. 頁面尺寸 (width_pt, height_pt)
  輸出儲存為 assets/layout_catalog.json (約 2~3 MB)，
  供本機合成測試與 Colab 邊訓練邊合成時極速直接查詢。
"""

import os
import re
import json
import glob
import time
import fitz

def parse_page_layout(page):
    w_pt, h_pt = page.rect.width, page.rect.height
    text_dict = page.get_text("dict")
    
    brackets = []
    options = []
    text_lines = []
    
    # 選項標籤正規表達式
    re_opt = re.compile(r'^[（\(][A-Ga-g1-5][\）\)]')
    
    for block in text_dict.get("blocks", []):
        if "lines" not in block:
            continue
        for line in block["lines"]:
            l_bbox = line["bbox"]
            text_lines.append([round(c, 1) for c in l_bbox])
            
            line_text = "".join(span.get("text", "") for span in line.get("spans", "")).strip()
            if not line_text:
                continue
                
            # 排除選項行 (例如 (A) 開頭的不是作答括號)
            is_option_line = bool(re_opt.match(line_text))
            
            # 1. 檢測選項標籤 (A), (B), (C), (D), (E)
            for span in line.get("spans", []):
                stxt = span.get("text", "").strip()
                sbox = span.get("bbox", [])
                if re_opt.match(stxt):
                    options.append([round(c, 1) for c in sbox])
                    
            # 2. 檢測作答括號 (例如行首的題號作答區，且不是選項行)
            # 例如: (   ) 1. 或 1. (  ) 或 ( ) 23.
            if not is_option_line:
                if line_text.startswith("(") or line_text.startswith("（") or line_text.startswith("【"):
                    brackets.append([round(l_bbox[0], 1), round(l_bbox[1], 1), round(min(l_bbox[0] + 35.0, l_bbox[2]), 1), round(l_bbox[3], 1)])
                elif any(kw in line_text for kw in ["（　）", "(   )", "（  ）", "( )", "（ ）"]):
                    for span in line.get("spans", []):
                        st = span.get("text", "")
                        if any(kw in st for kw in ["（　）", "(   )", "（  ）", "( )", "（ ）"]):
                            brackets.append([round(c, 1) for c in span.get("bbox", l_bbox)])

    return {
        "width_pt": round(w_pt, 1),
        "height_pt": round(h_pt, 1),
        "brackets": brackets,
        "options": options,
        "text_lines": text_lines
    }

def main():
    os.makedirs("assets", exist_ok=True)
    pdf_files = sorted(glob.glob("clean_pdfs/*.pdf"))
    print(f"[Layout] 找到 {len(pdf_files)} 份純淨試卷 PDF，開始解析版面...")
    
    catalog = []
    t0 = time.time()
    total_pages = 0
    
    for pdf_path in pdf_files:
        rel_pdf = os.path.relpath(pdf_path, ".").replace("\\", "/")
        doc = fitz.open(pdf_path)
        for pi in range(doc.page_count):
            page = doc[pi]
            layout = parse_page_layout(page)
            layout["pdf"] = rel_pdf
            layout["page"] = pi
            catalog.append(layout)
            total_pages += 1
        doc.close()
        
    out_path = "assets/layout_catalog.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False)
        
    file_size_kb = os.path.getsize(out_path) / 1024.0
    print("=" * 60)
    print(f"🎉 解析完成！共 {total_pages} 頁版面資訊 (耗時: {time.time() - t0:.2f}s)")
    print(f"儲存於: {out_path} ({file_size_kb:.1f} KB)")
    print("=" * 60)

if __name__ == "__main__":
    main()
