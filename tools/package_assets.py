#!/usr/bin/env python3
"""
打包在線合成所需的輕量資產包 (exam_assets_lightweight.zip)
包含：
  - assets/ (42頁手寫層, 391個真實手寫元素, layout_catalog.json)
  - clean_pdfs/ (35份純淨考卷PDF)
  - tools/ (synthetic_engine_v2.py, dataset_on_the_fly.py, train_on_the_fly.py)
"""

import os
import zipfile

def make_lightweight_zip(output_zip="exam_assets_lightweight.zip"):
    print(f"開始打包輕量資產包至: {output_zip} ...")
    
    total_files = 0
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        # 1. assets/ 目錄
        for root, dirs, files in os.walk("assets"):
            for f in files:
                fpath = os.path.join(root, f)
                z.write(fpath, arcname=fpath)
                total_files += 1
                
        # 2. clean_pdfs/ 目錄
        for root, dirs, files in os.walk("clean_pdfs"):
            for f in files:
                fpath = os.path.join(root, f)
                z.write(fpath, arcname=fpath)
                total_files += 1
                
        # 3. tools/ 核心腳本
        tool_scripts = [
            ("tools/synthetic_engine_v2.py", "tools/synthetic_engine_v2.py"),
            ("tools/dataset_on_the_fly.py", "tools/dataset_on_the_fly.py"),
            ("tools/train_colab_onthefly.py", "tools/train_colab_onthefly.py"),
            ("tools/train_on_the_fly.py", "tools/train_on_the_fly.py"),
            # 根目錄也放一份方便 Colab 直接引用
            ("tools/synthetic_engine_v2.py", "synthetic_engine_v2.py"),
            ("tools/dataset_on_the_fly.py", "dataset_on_the_fly.py"),
            ("tools/train_colab_onthefly.py", "train_colab_onthefly.py"),
        ]
        for src, arc in tool_scripts:
            if os.path.exists(src):
                z.write(src, arcname=arc)
                total_files += 1

    size_mb = os.path.getsize(output_zip) / (1024 * 1024)
    print(f"🎉 打包完成！共收錄 {total_files} 個檔案，壓縮後體積僅: {size_mb:.2f} MB")

if __name__ == "__main__":
    make_lightweight_zip()
