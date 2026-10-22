#!/bin/bash
set -euo pipefail

SESSION="exam-hybrid"
COLAB_BIN="/home/retsnimle/.local/bin/colab"
BASE_DIR="/mnt/g/CODE/handwriteremove"

echo "=========================================================="
echo "★ 雙軌混合考卷去手寫 一體化自動訓練管線 (Hybrid Pipeline)"
echo "   配置：在地真實試卷動態合成 80% + ExamInk-Seg 真實試卷 20%"
echo "   保護：100% 括號作答 + 題號/選項劃記 + 3.0x 印刷字保護"
echo "=========================================================="

echo "=== [1/6] 檢查 Colab 實例與環境 ==="
if ! "$COLAB_BIN" ls | grep -q "$SESSION"; then
    echo "[Colab] 建立 T4 實例..."
    "$COLAB_BIN" new -s "$SESSION" --gpu T4
    "$COLAB_BIN" install -s "$SESSION" segmentation-models-pytorch opencv-python-headless pymupdf huggingface_hub
fi

echo "=== [2/6] 同步最新清洗之題號錨點庫與 DataLoader ==="
"$COLAB_BIN" upload -s "$SESSION" "$BASE_DIR/tools/dataset_dynamic_scanned.py" /content/dataset_dynamic_scanned.py
"$COLAB_BIN" upload -s "$SESSION" "$BASE_DIR/tools/dataset_hybrid.py" /content/dataset_hybrid.py
"$COLAB_BIN" upload -s "$SESSION" "$BASE_DIR/tools/train_colab_hybrid.py" /content/train_colab_hybrid.py
"$COLAB_BIN" upload -s "$SESSION" "$BASE_DIR/assets/exam_anchors_catalog.json" /content/assets/exam_anchors_catalog.json

echo "=== [3/6] 等待 ExamInk-Seg 真實試卷庫就緒 ==="
"$COLAB_BIN" exec -s "$SESSION" << 'EOF'
import os, glob, time
from huggingface_hub import snapshot_download

local_dir = "/content/ExamInk-Seg"
srcs = glob.glob(f"{local_dir}/data/train/source/*.*")
if len(srcs) < 500:
    print("[Colab] 正在補充/確保 ExamInk-Seg 下載完成...")
    snapshot_download(repo_id="ynyg/ExamInk-Seg", repo_type="dataset", local_dir=local_dir, max_workers=8)

srcs = glob.glob(f"{local_dir}/data/train/source/*.*")
print(f"[Colab] ★ ExamInk-Seg 就緒: 訓練樣本數 = {len(srcs)}")
EOF

echo "=== [4/6] 啟動雙軌真實混合深度訓練 (15 Epochs) ==="
"$COLAB_BIN" exec -s "$SESSION" --timeout 7200 << 'EOF'
import subprocess, sys

cmd = (
    "python3 train_colab_hybrid.py "
    "--epochs 15 "
    "--steps_per_epoch 150 "
    "--val_steps 25 "
    "--batch_size 4 "
    "--lr 1e-4 "
    "--real_ratio 0.20 "
    "--text_penalty 3.0 "
    "--save_path weights/best_exam_remover_hybrid.pth "
    "--resume weights/best_exam_remover.pth"
)
print(f"[Colab] 執行混合訓練: {cmd}")
p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in iter(p.stdout.readline, ''):
    sys.stdout.write(line)
    sys.stdout.flush()
p.wait()
if p.returncode != 0:
    raise RuntimeError(f"訓練異常退出，Return Code: {p.returncode}")
EOF

echo "=== [5/6] 回收最佳模型權重並同步部署 ==="
mkdir -p "$BASE_DIR/weights"
"$COLAB_BIN" download -s "$SESSION" /content/weights/best_exam_remover_hybrid.pth "$BASE_DIR/weights/best_exam_remover_hybrid.pth"
cp -f "$BASE_DIR/weights/best_exam_remover_hybrid.pth" "$BASE_DIR/weights/best_exam_remover.pth"
echo "★ 權重已更新至本地！"

echo "=== [6/6] 釋放 Colab T4 實例 ==="
"$COLAB_BIN" stop -s "$SESSION" || true
echo "★ Colab 實例已釋放。"
