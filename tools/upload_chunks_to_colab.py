import os
import shutil
import subprocess

CHUNK_SIZE = 35 * 1024 * 1024 # 35MB
ZIP_PATH = r"G:\CODE\handwriteremove\exam_hybrid_bundle.zip"
OUT_DIR = r"G:\CODE\handwriteremove\chunks_upload"

if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

# 分割 zip
with open(ZIP_PATH, 'rb') as f:
    idx = 0
    while True:
        chunk = f.read(CHUNK_SIZE)
        if not chunk:
            break
        chunk_file = os.path.join(OUT_DIR, f"chunk_{idx:02d}")
        with open(chunk_file, 'wb') as cf:
            cf.write(chunk)
        print(f"Created chunk: {chunk_file} ({len(chunk)/1024/1024:.2f} MB)")
        idx += 1

print(f"Total {idx} chunks created.")

# 透過 WSL 上傳
for i in range(idx):
    wsl_src = f"/mnt/g/CODE/handwriteremove/chunks_upload/chunk_{i:02d}"
    remote_dst = f"/content/chunk_{i:02d}"
    cmd = ["wsl", "-e", "/home/retsnimle/.local/bin/colab", "upload", "-s", "exam-hybrid", wsl_src, remote_dst]
    print(f"Uploading {wsl_src} to Colab...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Failed to upload chunk_{i:02d}: {res.stderr}")
        break
    else:
        print(f"Uploaded chunk_{i:02d} successfully!")

print("All chunks uploaded! Merging on Colab...")
# 遠端合併並解壓
merge_cmd = """wsl -e /home/retsnimle/.local/bin/colab exec -s exam-hybrid << 'EOF'
cd /content
cat chunk_* > exam_hybrid_bundle.zip
unzip -q -o exam_hybrid_bundle.zip
rm -f chunk_*
echo "Successfully unpacked bundle on Colab!"
ls -la /content
EOF"""

os.system(merge_cmd)
