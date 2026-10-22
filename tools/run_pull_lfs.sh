#!/bin/bash
set -euo pipefail

SESSION="exam-remover-huggingface"
COLAB_BIN="/home/retsnimle/.local/bin/colab"

"$COLAB_BIN" upload -s "$SESSION" /mnt/g/CODE/handwriteremove/tools/check_and_pull_lfs.py /content/check_and_pull_lfs.py

"$COLAB_BIN" exec -s "$SESSION" --timeout 1800 << 'EOF'
import subprocess, sys

process = subprocess.Popen("python3 -u /content/check_and_pull_lfs.py", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in iter(process.stdout.readline, ''):
    sys.stdout.write(line)
    sys.stdout.flush()

process.wait()
EOF
