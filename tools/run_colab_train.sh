#!/bin/bash
set -euo pipefail

SESSION="exam-remover"
COLAB_BIN="/home/retsnimle/.local/bin/colab"

echo "=== 1. 確認/建立 Colab T4 實例 ($SESSION) ==="
"$COLAB_BIN" new -s "$SESSION" --gpu T4 || true

echo "=== 2. 安裝遠端依賴套件 ==="
"$COLAB_BIN" install -s "$SESSION" segmentation-models-pytorch opencv-python-headless pymupdf

echo "=== 3. 正式啟動 1,500 頁全頁題目卷在線合成深度訓練 ==="
"$COLAB_BIN" exec -s "$SESSION" --timeout 7200 << 'EOF'
import subprocess, sys

cmd = "python3 train_colab_onthefly.py --epochs 15 --steps_per_epoch 100 --val_steps 20 --batch_size 2 --lr 3e-4 --save_path weights/best_exam_remover.pth"
print(f"[Colab] 啟動訓練: {cmd}")

process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in iter(process.stdout.readline, ''):
    sys.stdout.write(line)
    sys.stdout.flush()

process.wait()
if process.returncode != 0:
    raise RuntimeError(f"訓練異常終止，Exit code: {process.returncode}")
EOF

echo "=== 4. 下載訓練完成之最佳模型權重 ==="
mkdir -p /mnt/g/CODE/handwriteremove/weights
"$COLAB_BIN" download -s "$SESSION" weights/best_exam_remover.pth /mnt/g/CODE/handwriteremove/weights/best_exam_remover.pth

echo "=== 5. 釋放 Colab T4 實例 ==="
"$COLAB_BIN" delete -s "$SESSION" -y || true

echo "🎉 全流程圓滿完成！權重已成功回傳至 G:\CODE\handwriteremove\weights\best_exam_remover.pth"
