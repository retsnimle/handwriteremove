import os, subprocess

print("[Colab] 查看 /content/ExamInk-Seg 目錄結構:")
for root, dirs, files in os.walk("/content/ExamInk-Seg"):
    print(f"{root}: {len(dirs)} dirs, {len(files)} files")
    if files:
        print("  Sample files:", files[:5])
