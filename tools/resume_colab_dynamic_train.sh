#!/bin/bash
set -euo pipefail

SESSION="exam-remover-dynamic"
COLAB_BIN="/home/retsnimle/.local/bin/colab"

echo "=== 1. 正式啟動雙端全動態 (底圖擾動 + 印章形變 + 文字懲罰) 深度訓練 ==="
"$COLAB_BIN" exec -s "$SESSION" --timeout 7200 << 'EOF'
import subprocess, sys

cmd = "python3 train_colab_dynamic.py --epochs 15 --steps_per_epoch 100 --val_steps 20 --batch_size 4 --lr 2e-4 --text_penalty 3.0 --save_path weights/best_exam_remover_dynamic.pth"
print(f"[Colab] 啟動訓練: {cmd}")

process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in iter(process.stdout.readline, ''):
    sys.stdout.write(line)
    sys.stdout.flush()

process.wait()
if process.returncode != 0:
    raise RuntimeError(f"訓練異常終止，Exit code: {process.returncode}")
EOF

echo "=== 2. 下載訓練完成之動態抗擬合模型權重 ==="
mkdir -p /mnt/g/CODE/handwriteremove/weights
"$COLAB_BIN" download -s "$SESSION" /content/weights/best_exam_remover_dynamic.pth /mnt/g/CODE/handwriteremove/weights/best_exam_remover_dynamic.pth
cp -f /mnt/g/CODE/handwriteremove/weights/best_exam_remover_dynamic.pth /mnt/g/CODE/handwriteremove/weights/best_exam_remover.pth

echo "=== 3. 釋放 Colab T4 實例 ==="
"$COLAB_BIN" stop -s "$SESSION" || true

echo "🎉 全流程圓滿完成！雙端全動態新模型權重已同步更新至 G:\CODE\handwriteremove\weights\best_exam_remover.pth"
