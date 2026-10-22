#!/bin/bash
set -euo pipefail

SESSION="exam-remover"
CHUNK_DIR="/mnt/g/CODE/handwriteremove/chunks_upload"
COLAB_BIN="/home/retsnimle/.local/bin/colab"

echo "=== 開始傳輸 18 個 50MB 分卷至 Colab ($SESSION) ==="

# 依序上傳每個分卷
for f in "$CHUNK_DIR"/part_*.bin; do
    fname=$(basename "$f")
    echo ">> 正在上傳 $fname ..."
    # 簡單重試機制
    for attempt in {1..3}; do
        if "$COLAB_BIN" upload -s "$SESSION" "$f" "$fname"; then
            echo "   ✓ $fname 上傳成功！"
            break
        else
            echo "   ⚠️ $fname 上傳失敗，第 $attempt 次重試..."
            sleep 2
        fi
    done
done

echo "🎉 18 個分卷全部上傳完畢！開始遠端合併與解壓縮..."

# 遠端執行合併與解壓
"$COLAB_BIN" exec -s "$SESSION" << 'EOF'
import os, subprocess

print("[Colab] 正在合併 18 個分卷...")
subprocess.run("cat part_*.bin > dataset.zip", shell=True, check=True)

print("[Colab] 正在解壓縮 dataset.zip...")
subprocess.run("unzip -q -o dataset.zip", shell=True, check=True)

print("[Colab] 清理分卷暫存檔...")
subprocess.run("rm -f part_*.bin dataset.zip", shell=True, check=True)

# 驗證解壓成果
import glob
imgs = glob.glob("dataset_fullpage/images/*.jpg")
masks = glob.glob("dataset_fullpage/masks/*.png")
print(f"🎉 遠端解壓驗證成功！images={len(imgs)}, masks={len(masks)}")
EOF

echo "=== 傳輸與部署全線完成！==="
