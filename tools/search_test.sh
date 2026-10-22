#!/bin/bash
/home/retsnimle/.local/bin/colab exec -s exam-remover << 'EOF'
import os
for root, dirs, files in os.walk('/'):
    if 'test.txt' in files:
        print("FOUND test.txt AT:", os.path.join(root, 'test.txt'))
        break
    # skip some huge dirs
    if root.startswith(('/proc', '/sys', '/usr')):
        dirs.clear()
else:
    print("test.txt was NOT found anywhere on /")
EOF
