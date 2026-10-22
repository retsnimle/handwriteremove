#!/usr/bin/env python3
"""
訓練資料集瘦身與 50MB 分卷打包腳本 (Prepare Split Chunks)
1. 僅打包訓練必要的 images/ 與 masks/ 以及 train_fullpage_colab.py (徹底去除 750MB 的 targets)
2. 將 zip 分割為 50MB 的小塊 (chunk_00, chunk_01...)，完美繞過 Jupyter API 的單檔上傳上限。
"""

import os
import zipfile
import shutil

def prepare_chunks(data_dir="dataset_fullpage", chunk_size_mb=50, out_dir="chunks_upload"):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    temp_zip = "temp_train_dataset.zip"
    print("正在打包必要訓練資源 (images + masks + training script)...")
    with zipfile.ZipFile(temp_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=4) as z:
        # 腳本
        script_path = os.path.join("tools", "train_fullpage_colab.py")
        if os.path.exists(script_path):
            z.write(script_path, arcname="train_fullpage_colab.py")

        # images
        in_dir = os.path.join(data_dir, "images")
        for f in os.listdir(in_dir):
            p = os.path.join(in_dir, f)
            z.write(p, arcname=os.path.join("dataset_fullpage", "images", f))

        # masks
        mk_dir = os.path.join(data_dir, "masks")
        for f in os.listdir(mk_dir):
            p = os.path.join(mk_dir, f)
            z.write(p, arcname=os.path.join("dataset_fullpage", "masks", f))

    zip_size_mb = os.path.getsize(temp_zip) / (1024 * 1024)
    print(f"打包完成！壓縮後體積僅: {zip_size_mb:.2f} MB")

    # 分割為 50MB 分卷
    chunk_bytes = chunk_size_mb * 1024 * 1024
    chunk_idx = 0
    with open(temp_zip, "rb") as f_in:
        while True:
            data = f_in.read(chunk_bytes)
            if not data:
                break
            chunk_file = os.path.join(out_dir, f"part_{chunk_idx:02d}.bin")
            with open(chunk_file, "wb") as f_out:
                f_out.write(data)
            sz = len(data) / (1024 * 1024)
            print(f"  產出分卷: {chunk_file} ({sz:.2f} MB)")
            chunk_idx += 1

    os.remove(temp_zip)
    print(f"🎉 全部切分完成！共 {chunk_idx} 個 50MB 分卷儲存於 {out_dir}/")

if __name__ == "__main__":
    prepare_chunks()
