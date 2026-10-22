#!/bin/bash
set -euo pipefail

SESSION="exam-hybrid"
COLAB_BIN="/home/retsnimle/.local/bin/colab"
BASE_DIR="/mnt/g/CODE/handwriteremove"

echo "=========================================================="
echo "★ 雙軌/三軌混合考卷去手寫 一體化自動訓練管線 (Tri-Track Decoupled)"
echo "   配置：在地真實試卷動態包覆 60% + ExamInk-Seg 20% + 飛輪難例 20%"
echo "   損失：DecoupledExamLoss (5px 安全緩衝 + 4.0x 跨字召回激勵)"
echo "=========================================================="

echo "=== [1/7] 檢查 Colab 實例與環境 ==="
if ! "$COLAB_BIN" ls | grep -q "$SESSION"; then
    echo "[Colab] 建立 T4 實例..."
    "$COLAB_BIN" new -s "$SESSION" --gpu T4
    "$COLAB_BIN" install -s "$SESSION" segmentation-models-pytorch opencv-python-headless pymupdf huggingface_hub
fi

echo "=== [2/7] 同步最新清洗之題號錨點庫、損失函數與 DataLoader ==="
"$COLAB_BIN" upload -s "$SESSION" "$BASE_DIR/tools/dataset_dynamic_scanned.py" /content/dataset_dynamic_scanned.py
"$COLAB_BIN" upload -s "$SESSION" "$BASE_DIR/tools/dataset_hybrid.py" /content/dataset_hybrid.py
"$COLAB_BIN" upload -s "$SESSION" "$BASE_DIR/tools/train_colab_hybrid.py" /content/train_colab_hybrid.py
"$COLAB_BIN" upload -s "$SESSION" "$BASE_DIR/losses/decoupled_exam_loss.py" /content/losses/decoupled_exam_loss.py
"$COLAB_BIN" upload -s "$SESSION" "$BASE_DIR/assets/exam_anchors_catalog.json" /content/assets/exam_anchors_catalog.json

echo "=== [3/7] 上傳並解壓 70 組飛輪大字偽標籤四元組 ==="
"$COLAB_BIN" upload -s "$SESSION" "$BASE_DIR/output/pseudo_hard_dataset.zip" /content/pseudo_hard_dataset.zip
"$COLAB_BIN" exec -s "$SESSION" << 'EOF'
import os, zipfile
if os.path.exists('/content/pseudo_hard_dataset.zip'):
    os.makedirs('/content/output', exist_ok=True)
    with zipfile.ZipFile('/content/pseudo_hard_dataset.zip', 'r') as zf:
        zf.extractall('/content/output')
    print("[Colab] ★ 飛輪難例解壓完成！目錄已就緒。")
EOF

echo "=== [4/7] 等待 ExamInk-Seg 真實試卷庫就緒 ==="
"$COLAB_BIN" exec -s "$SESSION" << 'EOF'
import os, glob, time
from huggingface_hub import snapshot_download

local_dir = "/content/ExamInk-Seg"
srcs = glob.glob(f"{local_dir}/data/train/source/*.*")
if len(srcs) < 500:
    print("[Colab] 正在確保 ExamInk-Seg 下載完成...")
    snapshot_download(repo_id="ynyg/ExamInk-Seg", repo_type="dataset", local_dir=local_dir, max_workers=8)

srcs = glob.glob(f"{local_dir}/data/train/source/*.*")
print(f"[Colab] ★ ExamInk-Seg 就緒: 訓練樣本數 = {len(srcs)}")
EOF

echo "=== [5/7] 啟動三軌解耦微調訓練 (8 Epochs 快速收斂) ==="
"$COLAB_BIN" exec -s "$SESSION" --timeout 7200 << 'EOF'
import subprocess, sys

cmd = (
    "python3 train_colab_hybrid.py "
    "--epochs 8 "
    "--steps_per_epoch 150 "
    "--val_steps 25 "
    "--batch_size 4 "
    "--lr 8e-5 "
    "--real_ratio 0.20 "
    "--pseudo_ratio 0.20 "
    "--text_penalty 3.0 "
    "--pseudo_root output/pseudo_hard_dataset "
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

echo "=== [6/7] 回收最佳模型權重並同步部署 ==="
mkdir -p "$BASE_DIR/weights"
"$COLAB_BIN" download -s "$SESSION" /content/weights/best_exam_remover_hybrid.pth "$BASE_DIR/weights/best_exam_remover_hybrid.pth"
cp -f "$BASE_DIR/weights/best_exam_remover_hybrid.pth" "$BASE_DIR/weights/best_exam_remover.pth"
echo "★ 權重已更新至本地！"

echo "=== [7/7] 釋放 Colab T4 實例 ==="
"$COLAB_BIN" stop -s "$SESSION" || true
echo "★ Colab 實例已釋放。"
