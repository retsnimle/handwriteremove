import os, subprocess, sys

print("[Colab] 安裝 git-lfs 系統套件...")
subprocess.run("apt-get update -qq && apt-get install -y -qq git-lfs", shell=True, check=True)
print("[Colab] git-lfs 安裝完成！")

f = "/content/ExamInk-Seg/data/train/source/1215.jpg"
print(f"[Colab] 目前 {f} 大小: {os.path.getsize(f)} bytes")

print("[Colab] 正在執行 git lfs pull 下載真實圖片...")
cmd = "cd /content/ExamInk-Seg && git lfs install && git lfs pull"
proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in iter(proc.stdout.readline, ''):
    sys.stdout.write(line)
    sys.stdout.flush()
proc.wait()

print(f"[Colab] ★ git lfs pull 完成！Exit code: {proc.returncode}")
print(f"[Colab] 新檔案大小: {os.path.getsize(f)} bytes")
