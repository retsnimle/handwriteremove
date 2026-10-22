#!/bin/bash
/home/retsnimle/.local/bin/colab exec -s exam-remover << 'EOF'
import zipfile, os, subprocess
print("正在將 /exam_assets_lightweight.zip 解壓縮至 /content ...")
with zipfile.ZipFile('/exam_assets_lightweight.zip', 'r') as z:
    z.extractall('/content')
print("解壓縮完成！目錄狀態:")
subprocess.run("ls -lh /content/assets /content/clean_pdfs", shell=True)
EOF
