#!/usr/bin/env python3
"""
雙軌混合微調資產打包器 (Package Hybrid Finetune Bundle)
包含：
1. 修正後的混合訓練腳本與資料集 (80% 在地考卷 + 20% 外國考卷)
2. 最新清洗過、100% 無假錨點的 exam_anchors_catalog.json
3. 311 個真實手寫字母庫 (hw_letter_marks.json) 與 80 個記號庫 (hw_symbol_marks.json)
4. 106 頁真實掃描考卷 PDF
5. 42 頁手寫草稿層與全部手寫印章
6. 現有最佳熱啟動權重 best_exam_remover.pth
"""
import os
import zipfile

def main():
    os.chdir(r"G:\CODE\handwriteremove")
    zip_path = "exam_hybrid_bundle.zip"
    print(f"📦 正在打包雙軌混合微調包至 {zip_path} ...")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. 訓練腳本與資料集
        zf.write("tools/train_colab_hybrid.py", "train_colab_hybrid.py")
        zf.write("tools/dataset_hybrid.py", "dataset_hybrid.py")
        zf.write("assets/scanned_catalog.json", "assets/scanned_catalog.json")
        zf.write("assets/exam_anchors_catalog.json", "assets/exam_anchors_catalog.json")
        zf.write("assets/hw_letter_marks.json", "assets/hw_letter_marks.json")
        zf.write("assets/hw_symbol_marks.json", "assets/hw_symbol_marks.json")

        # 2. 106 頁真實掃描考卷
        for f in os.listdir("scanned_pairs"):
            if f.endswith(".pdf"):
                p = os.path.join("scanned_pairs", f)
                zf.write(p, p)
                print(f"  加入考卷底稿: {p}")

        # 3. 手寫草稿層
        for f in os.listdir("assets/hw_layers"):
            if f.endswith(".png"):
                p = os.path.join("assets/hw_layers", f)
                zf.write(p, p)

        # 4. 手寫記號印章
        for f in os.listdir("assets/hw_marks"):
            p = os.path.join("assets/hw_marks", f)
            zf.write(p, p)

        # 5. 熱啟動基礎模型權重
        if os.path.exists("weights/best_exam_remover.pth"):
            zf.write("weights/best_exam_remover.pth", "weights/best_exam_remover.pth")
            print("  加入基礎權重: weights/best_exam_remover.pth")

    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"★ 打包完成: {zip_path} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    main()
