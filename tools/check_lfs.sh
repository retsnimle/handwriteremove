#!/bin/bash
set -euo pipefail

SESSION="exam-remover-huggingface"
COLAB_BIN="/home/retsnimle/.local/bin/colab"

"$COLAB_BIN" exec -s "$SESSION" << 'EOF'
import os
print("檢查文件類型與大小:")
f = "/content/ExamInk-Seg/data/train/source/0.png"
if os.path.exists(f):
    size = os.path.getsize(f)
    print(f"{f} size: {size} bytes")
    with open(f, "rb") as fp:
        header = fp.read(100)
        print("Header:", header[:40])
EOF
