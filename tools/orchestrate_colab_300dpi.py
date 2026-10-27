#!/usr/bin/env python3
r"""
Colab 雲端 300 DPI 端到端全自動生產總調度器 (終極驗收加固版)
===========================================================
依據 Aider 戰略級架構顧問 (550B 大模型) 終極驗收審查全數加固：
1. 自動權重打包保障：自動檢查並打包 models_and_weights.zip (3 份核心權重 + DocUnfoldV2 模組定義)。
2. 超時保護提升：單批推論逾時放寬至 10800 秒 (3小時)，訓練逾時放寬至 9000 秒 (2.5小時)。
3. 微批次分卷傳輸防斷線：35MB 智慧分卷 + cat 合併，徹底杜絕 WSL/Colab SSL EOF。
4. 嚴格目標範圍約束：僅限 F:\【教學】自然科段考卷搜集 下之 [高中] (160份) 與 [國中] (83份)。
5. 100% 排除 [99課綱封存] 與 [已修圖]。
6. ZIP 完整性檢驗 (testzip) + 作業系統級同目錄原子替換 (os.replace)。
"""

import os
import sys
import time
import json
import glob
import shutil
import zipfile
import subprocess
import unicodedata
import tempfile
import argparse

TARGET_ROOT = r"F:\【教學】自然科段考卷搜集"
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = PROJECT_DIR
STAGING_DIR = os.path.join(PROJECT_DIR, "colab_staging")
PROGRESS_FILE = os.path.join(STAGING_DIR, "colab_300dpi_production_progress.json")
SESSION = "exam-v2-prod-300"
COLAB_BIN = "/home/retsnimle/.local/bin/colab"
MICRO_BATCH_SIZE = 18  # 每批 18 份 (約 40MB)
EXEC_TIMEOUT = 10800   # 3 小時


def to_wsl_path(win_path):
    """將 Windows 路徑轉換為 WSL /mnt/<drive>/... 路徑格式"""
    p = os.path.abspath(win_path).replace("\\", "/")
    if len(p) >= 2 and p[1] == ":":
        drive = p[0].lower()
        return f"/mnt/{drive}{p[2:]}"
    return p


class CmdResult:
    def __init__(self, returncode, stdout, stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def run_wsl_cmd(cmd, check=True, retries=3, retry_delay=30):
    print(f">> [WSL Command] {cmd}", flush=True)
    attempt = 0
    while True:
        p = subprocess.Popen(
            ["wsl", "bash", "-c", cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace"
        )
        collected = []
        for line in p.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            collected.append(line)
        p.wait()

        stdout_all = "".join(collected)
        res = CmdResult(p.returncode, stdout_all, "")

        if p.returncode == 0 or not check:
            return res
        attempt += 1
        if attempt >= retries:
            raise subprocess.CalledProcessError(p.returncode, cmd, output=stdout_all)
        print(f"⚠️ [網路/連線暫態異常] 指令返回碼 {p.returncode}，將於 {retry_delay}s 後進行第 {attempt}/{retries} 次重試...", flush=True)
        time.sleep(retry_delay)


def scan_target_files():
    """
    精確掃描 [高中] 與 [國中] 目錄，嚴格排除 [99課綱封存] 與 [已修圖]
    支援 Unicode NFC 規範化比對
    """
    target_dirs = ["高中", "國中"]
    raw_files = []
    for td in target_dirs:
        dir_path = os.path.join(TARGET_ROOT, td)
        if not os.path.exists(dir_path):
            continue
        for root, dirs, files in os.walk(dir_path):
            norm_root = unicodedata.normalize("NFC", root)
            if "已修圖" in norm_root or "99課綱封存" in norm_root:
                continue
            for f in files:
                norm_f = unicodedata.normalize("NFC", f)
                if norm_f.lower().endswith(".pdf") and not norm_f.endswith("_去手寫.pdf"):
                    full = os.path.join(root, f)
                    rel = os.path.relpath(full, TARGET_ROOT)
                    raw_files.append({"full": full, "rel": rel, "size": os.path.getsize(full)})
    raw_files.sort(key=lambda x: x["rel"])
    return raw_files


def ensure_models_zip_exists():
    """確保 models_and_weights.zip 存在，若不存在則自動打包"""
    models_zip = os.path.join(STAGING_DIR, "models_and_weights.zip")
    if os.path.exists(models_zip) and os.path.getsize(models_zip) > 10 * 1024 * 1024:
        return models_zip

    print(f"📦 [自動打包基礎模型] 正在打包核心權重與模組至 {models_zip} ...", flush=True)
    os.makedirs(STAGING_DIR, exist_ok=True)
    with zipfile.ZipFile(models_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. 核心權重
        weights_to_include = [
            "best_docunfold_v2_dual_300dpi.pth",
            "best_docunfold_v2_5ch.pth",
            "best_exam_remover.pth",
            "doclayout_yolo_docstructbench_imgsz1024.pt"
        ]
        for w in weights_to_include:
            wp = os.path.join(PROJECT_DIR, "weights", w)
            if os.path.exists(wp):
                zf.write(wp, arcname=f"weights/{w}")

        # 2. 模組定義
        for m in ["DocUnfold.py", "DocUnfoldV2.py"]:
            mp_root = os.path.join(PROJECT_DIR, m)
            mp_models = os.path.join(PROJECT_DIR, "models", m)
            if os.path.exists(mp_root):
                zf.write(mp_root, arcname=m)
            elif os.path.exists(mp_models):
                zf.write(mp_models, arcname=m)

    sz_mb = os.path.getsize(models_zip) / (1024 * 1024)
    print(f"★ 基礎模型包建立就緒: {models_zip} ({sz_mb:.1f} MB)", flush=True)
    return models_zip


def package_training_bundle(zip_path):
    print(f"📦 [打包訓練資產] 正在打包 300 DPI 筆跡與底稿資產包至 {zip_path} ...", flush=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. 300 DPI 筆跡層
        for f in glob.glob(os.path.join(PROJECT_DIR, "assets", "hw_layers_300dpi", "*.png")):
            zf.write(f, arcname=f"assets/hw_layers_300dpi/{os.path.basename(f)}")
        # 2. 300 DPI 記號與元數據
        for f in glob.glob(os.path.join(PROJECT_DIR, "assets", "hw_marks_300dpi", "*")):
            zf.write(f, arcname=f"assets/hw_marks_300dpi/{os.path.basename(f)}")
        # 3. 錨點與目錄
        for cat in ["scanned_catalog.json", "exam_anchors_catalog_points.json", "exam_anchors_catalog.json"]:
            cp = os.path.join(PROJECT_DIR, "assets", cat)
            if os.path.exists(cp):
                zf.write(cp, arcname=f"assets/{cat}")
        # 4. 底稿 PDF (scanned_pairs)
        for f in glob.glob(os.path.join(PROJECT_DIR, "scanned_pairs", "*.pdf")):
            zf.write(f, arcname=f"scanned_pairs/{os.path.basename(f)}")

    sz_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"★ 訓練資產包打包完成: {zip_path} ({sz_mb:.1f} MB)", flush=True)


def init_colab_environment():
    print(f"=== [Colab 實例初始化] Session: {SESSION} ===", flush=True)
    res = subprocess.run(["wsl", "bash", "-c", f"{COLAB_BIN} ls -s {SESSION}"], capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[Colab] 建立全新 T4 GPU 實例 ({SESSION})...", flush=True)
        run_wsl_cmd(f"{COLAB_BIN} new -s {SESSION} --gpu T4")

    print(f"[Colab] 檢查並安裝 Python 依賴套件...", flush=True)
    run_wsl_cmd(f"{COLAB_BIN} install -s {SESSION} timm einops opencv-python-headless pymupdf huggingface_hub torchvision ultralytics doclayout-yolo segmentation-models-pytorch")

    # 檢查遠端是否已存在完整的核心權重 (包含 300 DPI 專用微調權重)
    check_code = """
import os
needed = ['best_docunfold_v2_dual_300dpi.pth', 'best_exam_remover.pth', 'doclayout_yolo_docstructbench_imgsz1024.pt']
all_exist = all(os.path.exists(f'/content/weights/{w}') and os.path.getsize(f'/content/weights/{w}') > 10*1024*1024 for w in needed)
print('ALL_WEIGHTS_PRESENT' if all_exist else 'WEIGHTS_MISSING')
"""
    check_res = run_wsl_cmd(f"{COLAB_BIN} exec -s {SESSION} << 'EOF'\n{check_code}\nEOF")
    if check_res.stdout and "ALL_WEIGHTS_PRESENT" in check_res.stdout:
        print("★ [Colab] 遠端核心權重已完整就緒 (包含 300 DPI 專用權重)，略過重複上傳！", flush=True)
        return

    # 同步基礎模型與權重包 models_and_weights.zip
    models_zip = ensure_models_zip_exists()
    chunk_size = 35 * 1024 * 1024
    temp_dir = os.path.join(STAGING_DIR, "temp_models_chunks")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    print(f"[Colab] 正在同步基礎模型架構與熱啟動權重...", flush=True)
    chunk_paths = []
    with open(models_zip, "rb") as f:
        idx = 0
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            cp = os.path.join(temp_dir, f"m_{idx:02d}")
            with open(cp, "wb") as cf:
                cf.write(data)
            chunk_paths.append(cp)
            idx += 1

    for i, cp in enumerate(chunk_paths):
        wsl_cp = to_wsl_path(cp)
        rem_cp = f"/content/mchunk_{i:02d}"
        run_wsl_cmd(f"{COLAB_BIN} upload -s {SESSION} {wsl_cp} {rem_cp}")

    merge_code = """
import os, subprocess, zipfile
print('[Remote] 合併模型權重分卷...')
subprocess.run('cat /content/mchunk_* > /content/models_and_weights.zip && rm -f /content/mchunk_*', shell=True, check=True)
os.makedirs('/content/weights', exist_ok=True)
with zipfile.ZipFile('/content/models_and_weights.zip', 'r') as zf:
    for member in zf.namelist():
        fname = os.path.basename(member)
        if not fname:
            continue
        if fname.endswith('.pt') or fname.endswith('.pth'):
            target = os.path.join('/content/weights', fname)
        else:
            target = os.path.join('/content', fname)
        with open(target, 'wb') as f_out, zf.open(member) as f_in:
            f_out.write(f_in.read())
print('[Remote] 基礎權重與模組解壓縮就緒！')
"""
    cmd = f"{COLAB_BIN} exec -s {SESSION} << 'EOF'\n{merge_code}\nEOF"
    run_wsl_cmd(cmd)

    try:
        shutil.rmtree(temp_dir)
    except Exception:
        pass


def run_stage1_training():
    """階段一：雲端 300 DPI 雙尺度原生微調訓練 (1500 steps)"""
    print(f"\n{'='*75}", flush=True)
    print(f"🏋️ [階段一啟動] 雲端 300 DPI 雙尺度原生微調訓練 (DocUnfoldV2 512x512)", flush=True)
    print(f"{'='*75}", flush=True)

    train_bundle_zip = os.path.join(STAGING_DIR, "train_bundle_300dpi.zip")
    package_training_bundle(train_bundle_zip)

    chunk_size = 35 * 1024 * 1024
    temp_dir = os.path.join(STAGING_DIR, "temp_train_chunks")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    print(f"[Colab] 正在上傳 300 DPI 訓練素材包至雲端...", flush=True)
    chunk_paths = []
    with open(train_bundle_zip, "rb") as f:
        idx = 0
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            cp = os.path.join(temp_dir, f"t_{idx:02d}")
            with open(cp, "wb") as cf:
                cf.write(data)
            chunk_paths.append(cp)
            idx += 1

    for i, cp in enumerate(chunk_paths):
        wsl_cp = to_wsl_path(cp)
        rem_cp = f"/content/tchunk_{i:02d}"
        run_wsl_cmd(f"{COLAB_BIN} upload -s {SESSION} {wsl_cp} {rem_cp}")

    wsl_trainer = to_wsl_path(os.path.join(PROJECT_DIR, "tools", "train_colab_docunfold_dual_300dpi.py"))
    wsl_docunfold = to_wsl_path(os.path.join(PROJECT_DIR, "models", "DocUnfold.py"))
    wsl_docunfoldv2 = to_wsl_path(os.path.join(PROJECT_DIR, "models", "DocUnfoldV2.py"))
    run_wsl_cmd(f"{COLAB_BIN} upload -s {SESSION} {wsl_trainer} /content/train_colab_docunfold_dual_300dpi.py")
    run_wsl_cmd(f"{COLAB_BIN} upload -s {SESSION} {wsl_docunfold} /content/DocUnfold.py")
    run_wsl_cmd(f"{COLAB_BIN} upload -s {SESSION} {wsl_docunfoldv2} /content/DocUnfoldV2.py")

    unzip_train_code = """
import os, subprocess, zipfile
print('[Remote] 合併並解壓訓練資產包...')
subprocess.run('cat /content/tchunk_* > /content/train_bundle_300dpi.zip && rm -f /content/tchunk_*', shell=True, check=True)
with zipfile.ZipFile('/content/train_bundle_300dpi.zip', 'r') as zf:
    zf.extractall('/content')
print('[Remote] 300 DPI 訓練資產就緒！')
"""
    run_wsl_cmd(f"{COLAB_BIN} exec -s {SESSION} << 'EOF'\n{unzip_train_code}\nEOF" )

    try:
        shutil.rmtree(temp_dir)
    except Exception:
        pass

    print(f"[Colab] 🚀 開始在 Tesla T4 (15GB) 上執行 1500 steps 微調訓練...", flush=True)
    t0_train = time.time()
    train_exec = """
import subprocess, os
subprocess.run([
    'python3', '/content/train_colab_docunfold_dual_300dpi.py',
    '--steps', '1500',
    '--batch_size', '1',
    '--grad_accum', '8',
    '--lr', '4e-5',
    '--weights_in', '/content/weights/best_docunfold_v2_5ch.pth',
    '--weights_out', '/content/weights/best_docunfold_v2_dual_300dpi.pth'
], check=True)

if not os.path.exists('/content/weights/best_docunfold_v2_dual_300dpi.pth'):
    raise RuntimeError("訓練未產出 best_docunfold_v2_dual_300dpi.pth！")
print('[Remote] ★ 300 DPI 專用權重驗證存在且就緒！')
"""
    cmd = f"{COLAB_BIN} exec -s {SESSION} --timeout 9000 << 'EOF'\n{train_exec}\nEOF"
    run_wsl_cmd(cmd)

    dt_train = (time.time() - t0_train) / 60.0
    print(f"🎉 [階段一完成] 300 DPI 專用微調權重訓練成功！耗時 {dt_train:.1f} 分鐘\n", flush=True)

    # 將訓練好的最新 300 DPI 專用權重完整下載回本機永久留存
    local_weights_dir = os.path.join(REPO_ROOT, "weights")
    os.makedirs(local_weights_dir, exist_ok=True)
    local_weights_path = os.path.join(local_weights_dir, "best_docunfold_v2_dual_300dpi.pth")
    wsl_local_weights = to_wsl_path(local_weights_path)

    print(f"[Colab] 📥 正在將 300 DPI 專用微調權重下載回本機永久保存: {local_weights_path} ...", flush=True)
    run_wsl_cmd(f"{COLAB_BIN} download -s {SESSION} /content/weights/best_docunfold_v2_dual_300dpi.pth {wsl_local_weights}")

    if os.path.exists(local_weights_path) and os.path.getsize(local_weights_path) > 10 * 1024 * 1024:
        sz_mb = os.path.getsize(local_weights_path) / (1024 * 1024)
        print(f"★ [本機保存成功] 300 DPI 權重已落地保存: {local_weights_path} ({sz_mb:.1f} MB)\n", flush=True)
    else:
        raise RuntimeError(f"❌ 300 DPI 權重下載失敗或檔案損毀: {local_weights_path}")


def package_batch_zip(batch_idx, batch_items, zip_path):
    print(f"📦 [打包] 正在打包 Micro-Batch {batch_idx:02d} ({len(batch_items)} 份試卷) 至 {zip_path} ...", flush=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in batch_items:
            zf.write(item["full"], arcname=item["rel"])
    sz_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"★ 打包完成: {zip_path} ({sz_mb:.1f} MB)", flush=True)


def process_micro_batch(batch_idx, batch_items):
    print(f"\n{'='*75}", flush=True)
    print(f"🚀 [執行 Micro-Batch {batch_idx:02d}] 包含 {len(batch_items)} 份試卷", flush=True)
    print(f"{'='*75}", flush=True)

    zip_in_name = f"input_micro_{batch_idx:02d}.zip"
    zip_in_path = os.path.join(STAGING_DIR, zip_in_name)
    wsl_zip_in = to_wsl_path(zip_in_path)

    zip_out_name = f"cleaned_micro_{batch_idx:02d}.zip"
    zip_out_path = os.path.join(STAGING_DIR, zip_out_name)
    wsl_zip_out = to_wsl_path(zip_out_path)

    package_batch_zip(batch_idx, batch_items, zip_in_path)

    # 確保遠端 worker 腳本與模型定義為最新版 (支援熱更新)
    wsl_worker = to_wsl_path(os.path.join(PROJECT_DIR, "tools", "colab_batch_worker_300dpi.py"))
    wsl_docunfold = to_wsl_path(os.path.join(PROJECT_DIR, "models", "DocUnfold.py"))
    wsl_docunfoldv2 = to_wsl_path(os.path.join(PROJECT_DIR, "models", "DocUnfoldV2.py"))
    run_wsl_cmd(f"{COLAB_BIN} upload -s {SESSION} {wsl_worker} /content/colab_batch_worker_300dpi.py")
    run_wsl_cmd(f"{COLAB_BIN} upload -s {SESSION} {wsl_docunfold} /content/DocUnfold.py")
    run_wsl_cmd(f"{COLAB_BIN} upload -s {SESSION} {wsl_docunfoldv2} /content/DocUnfoldV2.py")

    print(f"[Colab] 上傳 {zip_in_name} 至雲端...", flush=True)
    run_wsl_cmd(f"{COLAB_BIN} upload -s {SESSION} {wsl_zip_in} /content/{zip_in_name}")

    remote_batch_run = f"""
import os, shutil, subprocess, zipfile

if os.path.exists('/content/input_batch'):
    shutil.rmtree('/content/input_batch')
if os.path.exists('/content/cleaned_batch'):
    shutil.rmtree('/content/cleaned_batch')

print('[Remote] 解壓縮 {zip_in_name}...')
os.makedirs('/content/input_batch', exist_ok=True)
with zipfile.ZipFile('/content/{zip_in_name}', 'r') as zf:
    zf.extractall('/content/input_batch')

print('[Remote] 啟動雙尺度 300 DPI 批次推論 (colab_batch_worker_300dpi.py)...')
subprocess.run(['python3', '/content/colab_batch_worker_300dpi.py'], check=True)

print('[Remote] 打包清洗成果至 /content/{zip_out_name} ...')
with zipfile.ZipFile('/content/{zip_out_name}', 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk('/content/cleaned_batch'):
        for f in files:
            full = os.path.join(root, f)
            rel = os.path.relpath(full, '/content/cleaned_batch')
            zf.write(full, arcname=rel)

sz_mb = os.path.getsize('/content/{zip_out_name}') / (1024 * 1024)
if not os.path.exists('/content/{zip_out_name}') or sz_mb < 0.001:
    raise RuntimeError("批次推論失敗，未產出有效壓縮包: {zip_out_name}")
print(f'[Remote] ★ 打包完成: /content/{zip_out_name} ({{sz_mb:.1f}} MB)')
"""
    cmd = f"{COLAB_BIN} exec -s {SESSION} --timeout {EXEC_TIMEOUT} << 'EOF'\n{remote_batch_run}\nEOF"
    run_wsl_cmd(cmd)

    print(f"[Colab] 下載雲端清洗成果 {zip_out_name} ...", flush=True)
    run_wsl_cmd(f"{COLAB_BIN} download -s {SESSION} /content/{zip_out_name} {wsl_zip_out}")

    print(f"🔍 [完整性檢驗] 檢查 {zip_out_name} 壓縮包格式...", flush=True)
    with zipfile.ZipFile(zip_out_path, "r") as zf:
        bad_file = zf.testzip()
        if bad_file:
            raise RuntimeError(f"下載的壓縮包損毀: {bad_file}")

    print(f"📥 [原子寫入] 正在將 {zip_out_name} 原子替換回本地 {TARGET_ROOT} ...", flush=True)
    with zipfile.ZipFile(zip_out_path, "r") as zf:
        for member in zf.namelist():
            target_path = os.path.join(TARGET_ROOT, member)
            target_dir = os.path.dirname(target_path)
            os.makedirs(target_dir, exist_ok=True)

            tmp_fd, tmp_file = tempfile.mkstemp(prefix="clean_tmp_", suffix=".pdf", dir=target_dir)
            with os.fdopen(tmp_fd, "wb") as f_tmp, zf.open(member) as f_in:
                shutil.copyfileobj(f_in, f_tmp)

            extracted_sz = os.path.getsize(tmp_file)
            if extracted_sz < 1024:
                os.remove(tmp_file)
                raise RuntimeError(f"解壓產物過小: {member} ({extracted_sz} bytes)")

            os.replace(tmp_file, target_path)
            sz_mb = extracted_sz / (1024 * 1024)
            print(f"  -> ★ 原子覆寫成功: {member} ({sz_mb:.2f} MB)", flush=True)

    try:
        os.remove(zip_in_path)
    except Exception:
        pass

    print(f"★ Micro-Batch {batch_idx:02d} 全流程完成並已安全原子寫入本地硬碟！", flush=True)


def save_progress(completed_batches):
    tmp_prog = PROGRESS_FILE + ".tmp"
    with open(tmp_prog, "w", encoding="utf-8") as f:
        json.dump({"completed_batches": sorted(list(completed_batches))}, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_prog, PROGRESS_FILE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_train", action="store_true", help="略過微調訓練，直接使用現有權重開始批次生產")
    args = parser.parse_args()

    os.makedirs(STAGING_DIR, exist_ok=True)
    raw_files = scan_target_files()
    total_files = len(raw_files)
    batches = [raw_files[i : i + MICRO_BATCH_SIZE] for i in range(0, total_files, MICRO_BATCH_SIZE)]
    num_batches = len(batches)

    print(f"==================================================================", flush=True)
    print(f"🎓 啟動自然科試卷 Colab 雲端雙尺度 300 DPI 端到端生產總管線", flush=True)
    print(f"   目標範圍: 僅限 [高中] 與 [國中] (已徹底排除 99課綱封存 與 已修圖)", flush=True)
    print(f"   目標試卷數: {total_files} 份 | 總批次數: {num_batches} (每批 ~{MICRO_BATCH_SIZE} 份)", flush=True)
    print(f"   架構配置: 150 DPI 宏觀全局語意 + 300 DPI 512x512 微觀極限修復", flush=True)
    print(f"   目標根目錄: {TARGET_ROOT}", flush=True)
    print(f"==================================================================", flush=True)

    init_colab_environment()

    # 階段一：訓練
    if not args.skip_train:
        run_stage1_training()

    # 上傳雙尺度 300 DPI worker 腳本
    wsl_worker = to_wsl_path(os.path.join(PROJECT_DIR, "tools", "colab_batch_worker_300dpi.py"))
    run_wsl_cmd(f"{COLAB_BIN} upload -s {SESSION} {wsl_worker} /content/colab_batch_worker_300dpi.py")

    # 階段二：批次生產
    completed_batches = set()
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                completed_batches = set(data.get("completed_batches", []))
        except Exception:
            pass

    t_global_start = time.time()
    for b_idx, b_items in enumerate(batches, 1):
        if b_idx in completed_batches:
            print(f"⏩ [略過] Micro-Batch {b_idx:02d}/{num_batches:02d} 已於先前完成，跳過。", flush=True)
            continue

        t_b_start = time.time()
        try:
            process_micro_batch(b_idx, b_items)
            completed_batches.add(b_idx)
            save_progress(completed_batches)
            dt_b = (time.time() - t_b_start) / 60.0
            print(f"🎉 Micro-Batch {b_idx:02d}/{num_batches:02d} 圓滿完成！耗時 {dt_b:.1f} 分鐘\n", flush=True)
        except Exception as e:
            print(f"❌ Micro-Batch {b_idx:02d} 執行異常: {e}", flush=True)
            raise e

    t_global_dt = (time.time() - t_global_start) / 60.0
    print(f"\n==================================================================", flush=True)
    print(f"🏆 全數 {total_files} 份 [高中] 與 [國中] 試卷已由雙尺度 300 DPI 超清清洗完成！", flush=True)
    print(f"   總耗時: {t_global_dt:.1f} 分鐘", flush=True)
    print(f"==================================================================", flush=True)

    print(f"[Colab] 生產任務全數結束，釋放雲端 GPU 實例 ({SESSION})...", flush=True)
    run_wsl_cmd(f"{COLAB_BIN} stop -s {SESSION}", check=False)


if __name__ == "__main__":
    main()
