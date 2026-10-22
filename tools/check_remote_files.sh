#!/bin/bash
set -euo pipefail
SESSION="exam-remover-huggingface"
/home/retsnimle/.local/bin/colab exec -s "$SESSION" --timeout 30 << 'EOF'
import os
for f in ["/content/exam_dynamic_bundle.zip", "/content/dataset_hybrid.py", "/content/train_colab_hybrid.py"]:
    print(f, "exists:", os.path.exists(f), "size:", os.path.getsize(f) if os.path.exists(f) else 0)
EOF
