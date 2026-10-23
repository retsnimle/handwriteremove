#!/bin/bash
set -euo pipefail

SESSION="exam-docunfold"
BASE_DIR="/mnt/g/CODE/handwriteremove"
COLAB_BIN="/home/retsnimle/.local/bin/colab"

DYNAMIC_BUNDLE="$BASE_DIR/exam_dynamic_bundle.zip"
PSEUDO_BUNDLE="$BASE_DIR/pseudo_hard_bundle.zip"

echo "=========================================================="
echo "🏆 終極端到端試卷手寫擦除自動化訓練管線 (HW5K + ExamInk-Seg + 在地合成)"
echo "   真實世界數據量：6,511 組 (HW5K 5,000 + ExamInk 1,511)"
echo "   三軌黃金取樣：真實試卷 70% + 台灣在地考卷 25% + 難例 5%"
echo "   模型：DocUnfold 7 階物理展開網絡 (12.15M)"
echo "   損失：百度冠軍非對稱損失 (Hole 3.0x + Valid 1.0x + Sobel 0.5x)"
echo "=========================================================="

echo "=== [1/6] 檢查 Colab 實例 ($SESSION) ==="
if ! "$COLAB_BIN" ls -s "$SESSION" 2>/dev/null; then
    echo "[Colab] 建立全新 Session ($SESSION)..."
    "$COLAB_BIN" new -s "$SESSION" --gpu T4
fi

echo "=== [2/6] 安裝系統與 Python 依賴 ==="
"$COLAB_BIN" install -s "$SESSION" timm einops opencv-python-headless pymupdf huggingface_hub p7zip-full

echo "=== [3/6] 下載 ExamInk-Seg 與 HW5K 真實試卷庫 (共 6,511 組) ==="
"$COLAB_BIN" exec -s "$SESSION" --timeout 1200 << 'EOF'
import os, glob, subprocess
from huggingface_hub import snapshot_download, hf_hub_download

# A. 下載 ExamInk-Seg (1,511 組)
local_examink = "/content/ExamInk-Seg"
srcs = glob.glob(f"{local_examink}/data/train/source/*.*")
if len(srcs) < 500:
    print("[Colab] 正在下載 ExamInk-Seg 真實試卷庫...")
    try:
        snapshot_download(repo_id="ynyg/ExamInk-Seg", repo_type="dataset", local_dir=local_examink, max_workers=8)
    except Exception as e:
        print(f"ExamInk 下載提醒: {e}")

# B. 下載並解壓 HW5K.7z (5,000 組)
hw5k_train = "/content/dataset/HW5K/train/input"
if not os.path.exists(hw5k_train) or len(os.listdir(hw5k_train)) < 1000:
    print("[Colab] 正在高速下載 HW5K.7z (9.31 GB)...")
    raw_dir = "/content/hw5k_raw"
    os.makedirs(raw_dir, exist_ok=True)
    z_path = hf_hub_download(repo_id="lkljty/HW5K", filename="HW5K.7z", repo_type="dataset", local_dir=raw_dir)
    
    print("[Colab] 下載完成！正在解壓縮 HW5K.7z...")
    os.makedirs("/content/dataset", exist_ok=True)
    subprocess.run(["7z", "x", z_path, "-o/content/dataset/HW5K/", "-y"], check=True)
    
    # 立即刪除壓縮檔釋放磁碟
    print("[Colab] 解壓成功！立即刪除 HW5K.7z 釋放 9.3GB 磁碟...")
    os.remove(z_path)

print("[Colab] ★ 真實世界數據庫就緒！")
EOF

echo "=== [4/6] 上傳資產包並遠端解壓縮 ==="
"$COLAB_BIN" upload -s "$SESSION" "$DYNAMIC_BUNDLE" /content/exam_dynamic_bundle.zip
"$COLAB_BIN" upload -s "$SESSION" "$PSEUDO_BUNDLE" /content/pseudo_hard_bundle.zip

"$COLAB_BIN" exec -s "$SESSION" --timeout 180 << 'EOF'
import zipfile, os

for zpath in ["/content/exam_dynamic_bundle.zip", "/content/pseudo_hard_bundle.zip"]:
    if os.path.exists(zpath):
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
print("[Colab] ★ 資產解壓覆蓋完成！")
EOF

echo "=== [5/6] 正式啟動終極端到端微調訓練 (30 Epochs, 3,000 Steps) ==="
"$COLAB_BIN" exec -s "$SESSION" --timeout 14400 << 'EOF'
import subprocess, os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

cmd = (
    "python3 -u tools/train_colab_docunfold_ultimate.py "
    "--epochs 30 "
    "--steps_per_epoch 100 "
    "--batch_size 1 "
    "--grad_accum 4 "
    "--lr 2e-4 "
    "--patch_size 256 "
    "--save_path weights/best_docunfold_championship.pth"
)
print(f"[Colab] 啟動訓練指令: {cmd}")
p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in iter(p.stdout.readline, ''):
    print(line, end='', flush=True)
p.wait()
if p.returncode != 0:
    raise RuntimeError(f"訓練異常退出，Return Code: {p.returncode}")
print("[Colab] ★ 終極微調訓練大功告成！")
EOF

echo "=== [6/6] 回收最佳權重 ==="
mkdir -p "$BASE_DIR/weights"
"$COLAB_BIN" download -s "$SESSION" /content/weights/best_docunfold_championship.pth "$BASE_DIR/weights/best_docunfold_championship.pth"
echo "★ 權重已同步至: $BASE_DIR/weights/best_docunfold_championship.pth"
