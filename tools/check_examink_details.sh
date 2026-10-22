#!/bin/bash
set -euo pipefail

SESSION="exam-remover-huggingface"
COLAB_BIN="/home/retsnimle/.local/bin/colab"

"$COLAB_BIN" exec -s "$SESSION" << 'EOF'
import subprocess, glob
sources = glob.glob("/content/ExamInk-Seg/data/train/source/*")
masks = glob.glob("/content/ExamInk-Seg/data/train/mask/*")
targets = glob.glob("/content/ExamInk-Seg/data/train/target/*")
print(f"ExamInk-Seg 實際就緒文件數量:")
print(f"  Source (含手寫考卷): {len(sources)}")
print(f"  Mask (手寫真值遮罩): {len(masks)}")
print(f"  Target (乾淨印刷底圖): {len(targets)}")
if sources:
    import cv2
    s_im = cv2.imread(sources[0])
    m_im = cv2.imread(masks[0])
    print(f"  範例尺寸: Source={s_im.shape}, Mask={m_im.shape}")
EOF
