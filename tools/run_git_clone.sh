#!/bin/bash
set -euo pipefail

SESSION="exam-remover-huggingface"
COLAB_BIN="/home/retsnimle/.local/bin/colab"

echo "=== 執行 Git Clone ExamInk-Seg ==="
"$COLAB_BIN" exec -s "$SESSION" --timeout 1800 << 'EOF'
import subprocess, os
print("[Colab] 正在以 git clone 下載 ExamInk-Seg 數據集...")
cmd = "git clone --depth 1 https://huggingface.co/datasets/ynyg/ExamInk-Seg /content/ExamInk-Seg"
p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
print(p.stdout)
print(p.stderr)
print("[Colab] 檢查目錄狀態:")
subprocess.run("ls -lh /content/ExamInk-Seg/data/train", shell=True)
EOF
