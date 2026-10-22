#!/bin/bash
set -euo pipefail

SESSION="exam-remover"
COLAB_BIN="/home/retsnimle/.local/bin/colab"

echo "=== 正式啟動 1,500 頁全頁題目卷在線深度訓練 ==="

"$COLAB_BIN" exec -s "$SESSION" --timeout 7200 << 'EOF'
import subprocess, sys

cmd = "python3 train_colab_onthefly.py --epochs 15 --steps_per_epoch 50 --val_steps 10 --batch_size 2 --lr 3e-4 --save_path weights/best_exam_remover.pth"
print(f"[Colab] 執行命令: {cmd}")

# 即時輸出訓練進度
process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in iter(process.stdout.readline, ''):
    sys.stdout.write(line)
    sys.stdout.flush()

process.wait()
if process.returncode != 0:
    raise RuntimeError(f"訓練異常終止，Exit code: {process.returncode}")
EOF

echo "=== 訓練圓滿結束！正在回傳最佳模型權重至本地... ==="
"$COLAB_BIN" download -s "$SESSION" weights/best_exam_remover.pth /mnt/g/CODE/handwriteremove/weights/best_exam_remover.pth

echo "🎉 權重已成功落地 G:\CODE\handwriteremove\weights\best_exam_remover.pth"
