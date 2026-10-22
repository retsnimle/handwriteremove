#!/bin/bash
set -euo pipefail

SESSION="exam-hybrid"
BASE_DIR="/mnt/g/CODE/handwriteremove"
COLAB_BIN="/home/retsnimle/.local/bin/colab"

DYNAMIC_BUNDLE="$BASE_DIR/exam_dynamic_bundle.zip"
PSEUDO_BUNDLE="$BASE_DIR/pseudo_hard_bundle.zip"

echo "=========================================================="
echo "★ 三軌解耦考卷去手寫 一體化自動訓練管線 (1,500 Steps 標準版)"
echo "   配置：在地動態包覆 72% + ExamInk-Seg 20% + 飛輪難例 8%"
echo "   損失：DecoupledExamLoss (5px 安全緩衝 + 4.0x 跨字召回激勵)"
echo "   總步數：15 Epochs × 100 Steps = 1,500 Steps"
echo "=========================================================="

echo "=== [1/7] 建立全新 Colab T4 實例 ($SESSION) ==="
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

echo "=== [2/7] 安裝遠端依賴套件 (segmentation-models-pytorch, PyMuPDF, huggingface_hub) ==="
"$COLAB_BIN" install -s "$SESSION" segmentation-models-pytorch opencv-python-headless pymupdf huggingface_hub
sleep 5

echo "=== [3/7] 上傳兩大核心資產包 (動態底圖包 50.4MB + 偽標籤與最新代碼包 30.1MB) ==="
"$COLAB_BIN" upload -s "$SESSION" "$DYNAMIC_BUNDLE" /content/exam_dynamic_bundle.zip
"$COLAB_BIN" upload -s "$SESSION" "$PSEUDO_BUNDLE" /content/pseudo_hard_bundle.zip

echo "=== [4/7] 遠端以 Python 一次性正規化解壓縮至 /content ==="
"$COLAB_BIN" exec -s "$SESSION" --timeout 180 << 'EOF'
import zipfile, os, glob

print("[Colab] 正在解壓縮 exam_dynamic_bundle.zip (106 頁在地底圖)...")
with zipfile.ZipFile("/content/exam_dynamic_bundle.zip", "r") as z:
    for m in z.infolist():
        clean_name = m.filename.replace('\\', '/')
        target = os.path.join('/content', clean_name)
        if clean_name.endswith('/'):
            os.makedirs(target, exist_ok=True)
        else:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with z.open(m) as src, open(target, 'wb') as dst:
                dst.write(src.read())

print("[Colab] 正在解壓縮 pseudo_hard_bundle.zip (70 組飛輪偽標籤 + 最新解耦代碼)...")
with zipfile.ZipFile("/content/pseudo_hard_bundle.zip", "r") as z:
    for m in z.infolist():
        clean_name = m.filename.replace('\\', '/')
        target = os.path.join('/content', clean_name)
        if clean_name.endswith('/'):
            os.makedirs(target, exist_ok=True)
        else:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with z.open(m) as src, open(target, 'wb') as dst:
                dst.write(src.read())

os.makedirs('/content/weights', exist_ok=True)

pdfs = glob.glob('/content/scanned_pairs/*.pdf')
hard_imgs = glob.glob('/content/output/pseudo_hard_dataset/images/*.png')
print(f"[Colab] ★ 資產解壓完成！在地 PDF 底稿: {len(pdfs)} 份, 飛輪難例: {len(hard_imgs)} 組。")
EOF

echo "=== [5/7] 下載並驗證 ExamInk-Seg 真實試卷庫 (1,511 對) ==="
"$COLAB_BIN" exec -s "$SESSION" --timeout 600 << 'EOF'
import os, glob
from huggingface_hub import snapshot_download

local_dir = "/content/ExamInk-Seg"
srcs = glob.glob(f"{local_dir}/data/train/source/*.*")
if len(srcs) < 500:
    print("[Colab] 正在從 HuggingFace 下載 ExamInk-Seg 真實試卷庫...")
    snapshot_download(repo_id="ynyg/ExamInk-Seg", repo_type="dataset", local_dir=local_dir, max_workers=8)

srcs = glob.glob(f"{local_dir}/data/train/source/*.*")
print(f"[Colab] ★ ExamInk-Seg 驗證通過: 訓練樣本數 = {len(srcs)}")
EOF

echo "=== [6/7] 正式啟動三軌解耦微調訓練 (15 Epochs × 100 Steps = 1,500 Steps) ==="
"$COLAB_BIN" exec -s "$SESSION" --timeout 14400 << 'EOF'
import subprocess, sys

cmd = (
    "python3 -u train_colab_hybrid.py "
    "--epochs 15 "
    "--steps_per_epoch 100 "
    "--val_steps 20 "
    "--batch_size 4 "
    "--lr 8e-5 "
    "--real_ratio 0.20 "
    "--pseudo_ratio 0.08 "
    "--text_penalty 3.0 "
    "--pseudo_root output/pseudo_hard_dataset "
    "--save_path weights/best_exam_remover_hybrid.pth"
)
print(f"[Colab] 啟動 1,500 Steps 完整訓練: {cmd}")
p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in iter(p.stdout.readline, ''):
    sys.stdout.write(line)
    sys.stdout.flush()
p.wait()
if p.returncode != 0:
    raise RuntimeError(f"訓練異常退出，Return Code: {p.returncode}")
print("[Colab] ★ 三軌解耦微調 1,500 Steps 訓練大功告成！")
EOF

echo "=== [7/7] 回收最佳模型權重並釋放實例 ==="
mkdir -p "$BASE_DIR/weights"
"$COLAB_BIN" download -s "$SESSION" /content/weights/best_exam_remover_hybrid.pth "$BASE_DIR/weights/best_exam_remover_hybrid.pth"
cp -f "$BASE_DIR/weights/best_exam_remover_hybrid.pth" "$BASE_DIR/weights/best_exam_remover.pth"
echo "★ 權重已成功更新至本地 weights/best_exam_remover_hybrid.pth 與 weights/best_exam_remover.pth！"

echo "[Colab] 釋放 T4 雲端實例..."
"$COLAB_BIN" stop -s "$SESSION" || true
echo "★ 管線全流程順利收官！"
