#!/bin/bash
set -euo pipefail

SESSION="exam-remover"
ZIP_FILE="/mnt/g/CODE/handwriteremove/exam_scanned_bundle.zip"
COLAB_BIN="/home/retsnimle/.local/bin/colab"

echo "=== 1. 建立/連接 Colab T4 實例 ($SESSION) ==="
"$COLAB_BIN" new -s "$SESSION" --gpu T4 || true

echo "=== 2. 安裝遠端依賴套件 ==="
"$COLAB_BIN" install -s "$SESSION" segmentation-models-pytorch opencv-python-headless pymupdf

echo "=== 3. 上傳真實掃描考卷資產包 (50.7 MB) ==="
"$COLAB_BIN" upload -s "$SESSION" "$ZIP_FILE" /content/exam_scanned_bundle.zip

echo "=== 4. 遠端以 Python zipfile 解壓縮至 /content ==="
"$COLAB_BIN" exec -s "$SESSION" << 'EOF'
import zipfile, os, subprocess
print("[Colab] 正在以 Python 內建 zipfile 解壓縮 /content/exam_scanned_bundle.zip...")
with zipfile.ZipFile("/content/exam_scanned_bundle.zip", "r") as z:
    z.extractall("/content")
print("[Colab] ★ 解壓縮大功告成！檢查目錄狀態:")
subprocess.run("ls -lh /content/scanned_pairs /content/assets", shell=True)
EOF

echo "=== 5. 正式啟動 100% 真實掃描試卷在線深度訓練 ==="
"$COLAB_BIN" exec -s "$SESSION" --timeout 7200 << 'EOF'
import subprocess, sys

cmd = "python3 train_colab_scanned.py --epochs 15 --steps_per_epoch 100 --val_steps 20 --batch_size 2 --lr 3e-4 --save_path weights/best_exam_remover.pth"
print(f"[Colab] 啟動訓練: {cmd}")

process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in iter(process.stdout.readline, ''):
    sys.stdout.write(line)
    sys.stdout.flush()

process.wait()
if process.returncode != 0:
    raise RuntimeError(f"訓練異常終止，Exit code: {process.returncode}")
EOF

echo "=== 6. 下載訓練完成之最佳模型權重 ==="
mkdir -p /mnt/g/CODE/handwriteremove/weights
"$COLAB_BIN" download -s "$SESSION" /content/weights/best_exam_remover.pth /mnt/g/CODE/handwriteremove/weights/best_exam_remover.pth

echo "=== 7. 釋放 Colab T4 實例 ==="
"$COLAB_BIN" stop -s "$SESSION" || true

echo "🎉 全流程圓滿完成！權重已成功回傳至 G:\CODE\handwriteremove\weights\best_exam_remover.pth"
