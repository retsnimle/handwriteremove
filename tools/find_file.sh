#!/bin/bash
/home/retsnimle/.local/bin/colab exec -s exam-remover << 'EOF'
import subprocess, os
print("Current dir:", os.getcwd())
out = subprocess.check_output('find / -name "*exam_assets*" 2>/dev/null', shell=True).decode()
print("Find results:\n", out)
EOF
