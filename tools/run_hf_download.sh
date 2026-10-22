#!/bin/bash
set -euo pipefail

SESSION="exam-remover-huggingface"
COLAB_BIN="/home/retsnimle/.local/bin/colab"

echo "=== 執行 HuggingFace 下載腳本 ==="
"$COLAB_BIN" exec -s "$SESSION" --timeout 1800 << 'EOF'
import subprocess
subprocess.run("python3 /content/download_hf_colab.py", shell=True, check=True)
EOF
