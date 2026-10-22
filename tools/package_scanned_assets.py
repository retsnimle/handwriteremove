#!/usr/bin/env python3
"""
打包基於真實掃描考卷的訓練資產包 (exam_scanned_bundle.zip)
包含：
  - scanned_pairs/ (106 頁真實掃描考卷)
  - assets/ (42 頁草稿層, 391 個真實手寫記號, scanned_catalog.json)
  - 核心訓練腳本
"""

import os
import zipfile

def make_scanned_zip(output_zip="exam_scanned_bundle.zip"):
    print(f"開始打包真實掃描訓練資產包至: {output_zip} ...")

    total_files = 0
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        # 1. scanned_pairs/ 目錄 (106 頁真實考卷)
        for root, dirs, files in os.walk("scanned_pairs"):
            for f in files:
                fpath = os.path.join(root, f)
                z.write(fpath, arcname=fpath)
                total_files += 1

        # 2. assets/ 目錄 (手寫層與記號)
        for root, dirs, files in os.walk("assets"):
            for f in files:
                fpath = os.path.join(root, f)
                z.write(fpath, arcname=fpath)
                total_files += 1

        # 3. 核心訓練腳本
        tool_scripts = [
            ("tools/dataset_scanned_onthefly.py", "tools/dataset_scanned_onthefly.py"),
            ("tools/train_colab_scanned.py", "tools/train_colab_scanned.py"),
            ("tools/dataset_scanned_onthefly.py", "dataset_scanned_onthefly.py"),
            ("tools/train_colab_scanned.py", "train_colab_scanned.py"),
        ]
        for src, arc in tool_scripts:
            if os.path.exists(src):
                z.write(src, arcname=arc)
                total_files += 1

    size_mb = os.path.getsize(output_zip) / (1024 * 1024)
    print(f"🎉 打包完成！共收錄 {total_files} 個檔案，壓縮後體積僅: {size_mb:.2f} MB")

if __name__ == "__main__":
    make_scanned_zip()
