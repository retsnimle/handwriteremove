#!/usr/bin/env python3
"""
全頁考卷訓練封裝包打包工具 (Package Full-Page Train Bundle)
將生成好的 300 頁全頁資料集與 Colab 訓練腳本打包為 exam_fullpage_train_bundle.zip
"""

import os
import zipfile

def package_bundle(out_zip="exam_fullpage_train_bundle.zip", data_dir="dataset_fullpage"):
    print(f"開始封裝全頁考卷訓練包至: {out_zip} ...")
    total = 0
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=5) as z:
        # 1. 訓練腳本
        script_src = os.path.join("tools", "train_fullpage_colab.py")
        if os.path.exists(script_src):
            z.write(script_src, arcname="train_fullpage_colab.py")
            total += 1

        # 2. 資料集目錄
        for root, dirs, files in os.walk(data_dir):
            for f in files:
                fpath = os.path.join(root, f)
                z.write(fpath, arcname=fpath)
                total += 1

    size_mb = os.path.getsize(out_zip) / (1024 * 1024)
    print(f"🎉 打包完成！共封裝 {total} 個檔案，總體積: {size_mb:.2f} MB")

if __name__ == "__main__":
    package_bundle()
