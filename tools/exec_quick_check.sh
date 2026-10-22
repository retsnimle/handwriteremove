#!/bin/bash
set -euo pipefail
/home/retsnimle/.local/bin/colab upload -s "exam-remover-huggingface" /mnt/g/CODE/handwriteremove/tools/quick_lfs_check.py /content/quick_lfs_check.py
/home/retsnimle/.local/bin/colab exec -s "exam-remover-huggingface" --timeout 30 << 'EOF'
import subprocess
print(subprocess.check_output("python3 /content/quick_lfs_check.py", shell=True, text=True))
EOF
