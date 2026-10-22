#!/bin/bash
set -euo pipefail

SESSION="exam-remover"
ZIP_FILE="/mnt/g/CODE/handwriteremove/exam_dynamic_bundle.zip"
COLAB_BIN="/home/retsnimle/.local/bin/colab"

echo "=== 1. 建立 Colab T4 實例 ($SESSION) ==="
MAX_RETRIES=5
RETRY_COUNT=0
SESSION_READY=false

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    echo "[Colab] 嘗試建立 Session (第 $((RETRY_COUNT+1)) 次)..."
    if "$COLAB_BIN" new -s "$SESSION" --gpu T4; then
        echo "[Colab] ★ Session 建立成功！"
        SESSION_READY=true
        break
    else
        echo "[Colab] 建立超時或繁忙，5 秒後重試..."
        "$COLAB_BIN" stop -s "$SESSION" 2>/dev/null || true
        sleep 5
        RETRY_COUNT=$((RETRY_COUNT+1))
    fi
done

if [ "$SESSION_READY" != "true" ]; then
    echo "錯誤: 無法建立 Colab Session，已達最大重試次數。"
    exit 1
fi

echo "=== 2. 安裝遠端依賴套件 ==="
"$COLAB_BIN" install -s "$SESSION" segmentation-models-pytorch opencv-python-headless pymupdf

echo "=== 3. 上傳雙端動態抗擬合資產包 (50.4 MB) ==="
"$COLAB_BIN" upload -s "$SESSION" "$ZIP_FILE" /content/exam_dynamic_bundle.zip

echo "=== 4. 遠端以 Python zipfile 解壓縮至 /content ==="
"$COLAB_BIN" exec -s "$SESSION" << 'EOF'
import zipfile, os, subprocess
print("[Colab] 正在解壓縮 /content/exam_dynamic_bundle.zip...")
with zipfile.ZipFile("/content/exam_dynamic_bundle.zip", "r") as z:
    z.extractall("/content")
print("[Colab] ★ 解壓縮大功告成！檢查目錄狀態:")
subprocess.run("ls -lh /content/scanned_pairs /content/assets", shell=True)
EOF

echo "=== 5. 正式啟動雙端全動態抗過擬合深度訓練 ==="
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

echo "=== 6. 下載訓練完成之動態抗擬合模型權重 ==="
mkdir -p /mnt/g/CODE/handwriteremove/weights
"$COLAB_BIN" download -s "$SESSION" /content/weights/best_exam_remover_dynamic.pth /mnt/g/CODE/handwriteremove/weights/best_exam_remover_dynamic.pth
cp -f /mnt/g/CODE/handwriteremove/weights/best_exam_remover_dynamic.pth /mnt/g/CODE/handwriteremove/weights/best_exam_remover.pth

echo "=== 7. 釋放 Colab T4 實例 ==="
"$COLAB_BIN" stop -s "$SESSION" || true

echo "🎉 全流程圓滿完成！雙端全動態新模型權重已同步更新至 G:\CODE\handwriteremove\weights\best_exam_remover.pth"
