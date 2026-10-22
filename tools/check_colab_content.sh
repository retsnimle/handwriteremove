#!/bin/bash
set -euo pipefail
SESSION="exam-remover-huggingface"
/home/retsnimle/.local/bin/colab exec -s "$SESSION" --timeout 30 << 'EOF'
import os
print("Files in /content:")
for f in os.listdir("/content"):
    if not f.startswith("."):
        print(" ", f)
EOF
