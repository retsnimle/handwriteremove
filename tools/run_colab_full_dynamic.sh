#!/bin/bash
set -euo pipefail

SESSION="exam-remover-dynamic"
ZIP_FILE="/mnt/g/CODE/handwriteremove/exam_dynamic_bundle.zip"
COLAB_BIN="/home/retsnimle/.local/bin/colab"

echo "=== 1. 安裝遠端依賴套件 (uv 極速模式) ==="
"$COLAB_BIN" install -s "$SESSION" segmentation-models-pytorch opencv-python-headless pymupdf

echo "=== 2. 上傳雙端全動態抗擬合資產包 (50.4 MB) ==="
"$COLAB_BIN" upload -s "$SESSION" "$ZIP_FILE" /content/exam_dynamic_bundle.zip

echo "=== 3. 遠端解壓縮資產包 ==="
"$COLAB_BIN" exec -s "$SESSION" << 'EOF'
import zipfile, os, subprocess
print("[Colab] 正在解壓縮 /content/exam_dynamic_bundle.zip...")
with zipfile.ZipFile("/content/exam_dynamic_bundle.zip", "r") as z:
    z.extractall("/content")
print("[Colab] ★ 解壓縮大功告成！檢查目錄狀態:")
subprocess.run("ls -lh /content/scanned_pairs /content/assets", shell=True)
EOF

echo "=== 4. 正式啟動雙端全動態 (底圖擾動 + 印章形變 + 文字懲罰) 深度訓練 ==="
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

echo "=== 5. 下載訓練完成之動態抗擬合模型權重 ==="
mkdir -p /mnt/g/CODE/handwriteremove/weights
"$COLAB_BIN" download -s "$SESSION" /content/weights/best_exam_remover_dynamic.pth /mnt/g/CODE/handwriteremove/weights/best_exam_remover_dynamic.pth
cp -f /mnt/g/CODE/handwriteremove/weights/best_exam_remover_dynamic.pth /mnt/g/CODE/handwriteremove/weights/best_exam_remover.pth

echo "=== 6. 釋放 Colab T4 實例 ==="
"$COLAB_BIN" stop -s "$SESSION" || true

echo "🎉 全流程圓滿完成！雙端全動態新模型權重已同步更新至 G:\CODE\handwriteremove\weights\best_exam_remover.pth"
