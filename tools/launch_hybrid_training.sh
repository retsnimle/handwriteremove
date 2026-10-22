#!/bin/bash
set -euo pipefail

SESSION="exam-remover-huggingface"
COLAB_BIN="/home/retsnimle/.local/bin/colab"

echo "[1/3] 上傳最新代碼至 Colab..."
"$COLAB_BIN" upload -s "$SESSION" /mnt/g/CODE/handwriteremove/tools/dataset_hybrid.py /content/dataset_hybrid.py
"$COLAB_BIN" upload -s "$SESSION" /mnt/g/CODE/handwriteremove/tools/train_colab_hybrid.py /content/train_colab_hybrid.py

echo "[2/3] 安裝必要依賴與驗證環境..."
"$COLAB_BIN" exec -s "$SESSION" --timeout 180 << 'EOF'
import subprocess, sys

print("[Colab] 安裝 segmentation-models-pytorch, PyMuPDF...")
subprocess.run("pip install -q segmentation-models-pytorch PyMuPDF opencv-python-headless", shell=True, check=True)
import segmentation_models_pytorch as smp
import fitz
print("[Colab] 依賴庫安裝完成！")
EOF

echo "[3/3] 啟動雙軌混合深度訓練 (15 Epochs, Real+Dynamic 50:50, ImageNet Pretrained, Text-Aware Penalty 3.0x)..."
"$COLAB_BIN" exec -s "$SESSION" --timeout 3600 << 'EOF'
import subprocess, sys

cmd = "python3 -u /content/train_colab_hybrid.py --epochs 15 --batch_size 4 --lr 2e-4 --real_ratio 0.50 --text_penalty 3.0 --steps_per_epoch 150 --val_steps 25 --save_path /content/weights/best_exam_remover_hybrid.pth --resume ''"
process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in iter(process.stdout.readline, ''):
    sys.stdout.write(line)
    sys.stdout.flush()

process.wait()
if process.returncode != 0:
    print(f"訓練異常退出，Return Code: {process.returncode}")
    sys.exit(process.returncode)
EOF

echo "★ 雙軌混合訓練完畢！"
