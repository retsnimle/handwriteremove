#!/usr/bin/env python3
"""
打包動態抗過擬合訓練包 (Dynamic Scanned Training Bundle Packager)
"""
import os
import zipfile

def main():
    zip_path = "exam_dynamic_bundle.zip"
    print(f"正在打包動態抗過擬合訓練包至 {zip_path} ...")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. 訓練腳本與動態資料集
        zf.write("tools/train_colab_dynamic.py", "train_colab_dynamic.py")
        zf.write("tools/dataset_dynamic_scanned.py", "dataset_dynamic_scanned.py")
        zf.write("assets/scanned_catalog.json", "assets/scanned_catalog.json")

        # 2. 106 頁真實掃描考卷
        scanned_files = [os.path.join("scanned_pairs", f) for f in os.listdir("scanned_pairs") if f.endswith(".pdf")]
        for sf in scanned_files:
            zf.write(sf, sf)
            print(f"  加入掃描檔: {sf}")

        # 3. 手寫草稿層 (hw_layers)
        layer_files = [os.path.join("assets/hw_layers", f) for f in os.listdir("assets/hw_layers") if f.endswith(".png")]
        for lf in layer_files:
            zf.write(lf, lf)

        # 4. 手寫記號層 (hw_marks)
        mark_files = [os.path.join("assets/hw_marks", f) for f in os.listdir("assets/hw_marks")]
        for mf in mark_files:
            zf.write(mf, mf)

    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"★ 打包完成: {zip_path} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    main()
