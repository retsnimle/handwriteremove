#!/bin/bash
/home/retsnimle/.local/bin/colab exec -s exam-remover << 'EOF'
import os
print("Is /exam_assets_lightweight.zip present?", os.path.exists('/exam_assets_lightweight.zip'))
if os.path.exists('/exam_assets_lightweight.zip'):
    print("Size:", os.path.getsize('/exam_assets_lightweight.zip'))
EOF
