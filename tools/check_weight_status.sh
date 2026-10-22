#!/bin/bash
set -euo pipefail
SESSION="exam-remover-huggingface"
/home/retsnimle/.local/bin/colab exec -s "$SESSION" --timeout 30 << 'EOF'
import os
p = "/content/weights/best_exam_remover_hybrid.pth"
print("Hybrid weight exists:", os.path.exists(p))
if os.path.exists(p):
    print("Hybrid weight size:", os.path.getsize(p), "bytes")
EOF
