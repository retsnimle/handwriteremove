#!/bin/bash
set -euo pipefail

SESSION="exam-docunfold"
BASE_DIR="/mnt/g/CODE/handwriteremove"
COLAB_BIN="/home/retsnimle/.local/bin/colab"

DYNAMIC_BUNDLE="$BASE_DIR/exam_dynamic_bundle.zip"
PSEUDO_BUNDLE="$BASE_DIR/pseudo_hard_bundle.zip"

echo "=========================================================="
echo "🏆 DocUnfold + 百度大賽冠軍端到端試卷手寫擦除自動化訓練管線"
echo "   架構：DocUnfold 7 階物理展開網絡 (12.15M)"
echo "   損失：Championship DeHW Loss (Hole 3.0x + Valid 1.0x + Sobel 0.5x)"
echo "   訓練目標：1:1 像素級端到端乾淨試卷重構 (免外部 Inpaint，徹底杜絕文字掏空)"
echo "=========================================================="

echo "=== [1/7] 建立全新 Colab 實例 ($SESSION, GPU T4/V100) ==="
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

echo "=== [2/7] 安裝遠端依賴套件 (timm, einops, PyMuPDF, huggingface_hub) ==="
"$COLAB_BIN" install -s "$SESSION" timm einops opencv-python-headless pymupdf huggingface_hub torchmetrics

echo "=== [3/7] 上傳核心資產包與最新程式碼 ==="
"$COLAB_BIN" upload -s "$SESSION" "$DYNAMIC_BUNDLE" /content/exam_dynamic_bundle.zip
"$COLAB_BIN" upload -s "$SESSION" "$PSEUDO_BUNDLE" /content/pseudo_hard_bundle.zip

echo "=== [4/7] 遠端解壓縮資產至 /content ==="
"$COLAB_BIN" exec -s "$SESSION" --timeout 180 << 'EOF'
import zipfile, os

for zpath in ["/content/exam_dynamic_bundle.zip", "/content/pseudo_hard_bundle.zip"]:
    if os.path.exists(zpath):
        print(f"[Colab] 正在解壓縮 {zpath}...")
        with zipfile.ZipFile(zpath, "r") as z:
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
print("[Colab] ★ 資產解壓完成！")
EOF

echo "=== [5/7] 下載 HW5K / ExamInk-Seg 真實試卷庫 ==="
"$COLAB_BIN" exec -s "$SESSION" --timeout 600 << 'EOF'
import os, glob
from huggingface_hub import snapshot_download

local_dir = "/content/ExamInk-Seg"
srcs = glob.glob(f"{local_dir}/data/train/source/*.*")
if len(srcs) < 500:
    print("[Colab] 正在從 HuggingFace 下載真實試卷庫...")
    try:
        snapshot_download(repo_id="ynyg/ExamInk-Seg", repo_type="dataset", local_dir=local_dir, max_workers=8)
    except Exception as e:
        print(f"ExamInk 下載跳過或失敗: {e}")

print("[Colab] 資料集檢查完畢！")
EOF

echo "=== [6/7] 正式啟動 DocUnfold + 百度冠軍改良訓練 ==="
"$COLAB_BIN" exec -s "$SESSION" --timeout 14400 << 'EOF'
import subprocess, sys

cmd = (
    "python3 -u tools/train_colab_docunfold_championship.py "
    "--epochs 30 "
    "--batch_size 2 "
    "--lr 2e-4 "
    "--patch_size 512 "
    "--save_path weights/best_docunfold_championship.pth"
)
print(f"[Colab] 啟動端到端訓練: {cmd}")
p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in iter(p.stdout.readline, ''):
    sys.stdout.write(line)
    sys.stdout.flush()
p.wait()
if p.returncode != 0:
    raise RuntimeError(f"訓練異常退出，Return Code: {p.returncode}")
print("[Colab] ★ DocUnfold 訓練大功告成！")
EOF

echo "=== [7/7] 回收最佳模型權重並釋放實例 ==="
mkdir -p "$BASE_DIR/weights"
"$COLAB_BIN" download -s "$SESSION" /content/weights/best_docunfold_championship.pth "$BASE_DIR/weights/best_docunfold_championship.pth"
echo "★ 權重已同步至: $BASE_DIR/weights/best_docunfold_championship.pth"
