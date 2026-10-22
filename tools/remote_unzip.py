import zipfile, os, sys

print("[Colab] 正在使用 Python 內建 zipfile 解壓縮 exam_assets_lightweight.zip...")
if not os.path.exists("exam_assets_lightweight.zip"):
    print("[Colab 錯誤] 找不到 exam_assets_lightweight.zip！")
    sys.exit(1)

with zipfile.ZipFile("exam_assets_lightweight.zip", "r") as z:
    z.extractall(".")

print("[Colab] ★ 解壓縮大功告成！檢查目錄:")
os.system("ls -lh assets clean_pdfs")
