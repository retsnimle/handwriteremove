#!/usr/bin/env python3
"""
一體化雙軌考卷去手寫 Colab 訓練管線 (End-to-End Hybrid Training Pipeline)
全自動執行：
1. 環境與 Session 狀態檢查
2. 確保遠端腳本與最新過濾錨點庫（100 括號 + 254 題號）同步
3. 等待/驗證 ExamInk-Seg 真實試卷數據集就緒
4. 啟動雙軌混合訓練 (80% 在地考卷動態合成 + 20% ExamInk 真實標註, 15 Epochs)
5. 自動下載最佳模型權重至本地 weights/best_exam_remover_hybrid.pth
6. 同步更新生產環境權重 best_exam_remover.pth
7. 自動釋放 Colab T4 實例
8. 本地端真實考卷自動推論驗收
"""

import os
import sys
import time
import subprocess
from pathlib import Path

SESSION = "exam-hybrid"
COLAB_BIN = "/home/retsnimle/.local/bin/colab"
BASE_DIR = Path("G:/CODE/handwriteremove")
WEIGHTS_DIR = BASE_DIR / "weights"
REMOTE_WEIGHT_PATH = "/content/weights/best_exam_remover_hybrid.pth"
LOCAL_WEIGHT_PATH = WEIGHTS_DIR / "best_exam_remover_hybrid.pth"
PROD_WEIGHT_PATH = WEIGHTS_DIR / "best_exam_remover.pth"

def run_wsl(cmd: str, check: bool = True, capture: bool = False):
    full_cmd = ["wsl", "bash", "-c", cmd]
    if capture:
        res = subprocess.run(full_cmd, capture_output=True, text=True, check=check)
        return res.stdout.strip()
    else:
        res = subprocess.run(full_cmd, check=check)
        return res.returncode

def ensure_colab_session():
    print(f"=== [Pipeline 1/7] 檢查 Colab T4 實例 ({SESSION}) ===")
    status_out = run_wsl(f"{COLAB_BIN} status -s {SESSION}", check=False, capture=True)
    if "Status: IDLE" in status_out or "Status: BUSY" in status_out or "Hardware: T4" in status_out:
        print(f"[Colab] ★ Session '{SESSION}' 存在且狀態良好，直接複用現有實例。")
        return
    
    print(f"[Colab] 正在建立全新 T4 實例...")
    run_wsl(f"{COLAB_BIN} new -s {SESSION} --gpu T4")
    run_wsl(f"{COLAB_BIN} install -s {SESSION} segmentation-models-pytorch opencv-python-headless pymupdf huggingface_hub")
    print(f"[Colab] ★ Session '{SESSION}' 初始化完成！")

def sync_latest_tools_and_anchors():
    print("=== [Pipeline 2/7] 同步最新過濾之題號錨點與訓練腳本 ===")
    files_to_sync = [
        ("tools/dataset_dynamic_scanned.py", "/content/dataset_dynamic_scanned.py"),
        ("tools/dataset_hybrid.py", "/content/dataset_hybrid.py"),
        ("tools/train_colab_hybrid.py", "/content/train_colab_hybrid.py"),
        ("assets/exam_anchors_catalog.json", "/content/assets/exam_anchors_catalog.json"),
    ]
    for local_rel, remote_path in files_to_sync:
        local_path = (BASE_DIR / local_rel).as_posix()
        wsl_local = f"/mnt/{local_path[0].lower()}{local_path[2:]}"
        print(f"  -> 上傳 {local_rel} 到 {remote_path}")
        run_wsl(f"{COLAB_BIN} upload -s {SESSION} '{wsl_local}' '{remote_path}'")
    print("[Colab] ★ 核心代碼與清洗後之 354 錨點庫已同步！")

def wait_for_examink_ready():
    print("=== [Pipeline 3/7] 驗證 ExamInk-Seg 真實試卷庫 ===")
    check_script = """
import os, glob
src_count = len(glob.glob('/content/ExamInk-Seg/data/train/source/*.*'))
mask_count = len(glob.glob('/content/ExamInk-Seg/data/train/mask/*.*'))
print(f'EXAMINK_STATUS: src={src_count}, mask={mask_count}')
"""
    retries = 0
    while retries < 30:
        out = run_wsl(f"{COLAB_BIN} exec -s {SESSION} << 'EOF'\n{check_script}\nEOF", capture=True)
        print(f"  {out}")
        if "EXAMINK_STATUS: src=" in out:
            # 檢查檔案數
            for line in out.splitlines():
                if "EXAMINK_STATUS: src=" in line:
                    src_num = int(line.split("src=")[1].split(",")[0])
                    if src_num > 500:
                        print(f"[Colab] ★ ExamInk-Seg 數據集就緒（共 {src_num} 對試卷）！")
                        return True
        print("  [Colab] 數據集下載中，等待 15 秒...")
        time.sleep(15)
        retries += 1
    
    raise RuntimeError("ExamInk-Seg 下載等待超時！")

def execute_hybrid_training():
    print("=== [Pipeline 4/7] 啟動雙軌真實混合訓練 (在地 80% + 外國 20%) ===")
    train_cmd = (
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
    
    exec_block = f"""
import subprocess, sys
print("[Colab] 執行指令: {train_cmd}")
p = subprocess.Popen("{train_cmd}", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in iter(p.stdout.readline, ''):
    sys.stdout.write(line)
    sys.stdout.flush()
p.wait()
if p.returncode != 0:
    raise RuntimeError(f"訓練異常退出，Return Code: {{p.returncode}}")
"""
    run_wsl(f"{COLAB_BIN} exec -s {SESSION} --timeout 7200 << 'EOF'\n{exec_block}\nEOF")
    print("[Colab] ★ 混合訓練順利完成！最優模型權重已生成。")

def download_and_deploy_weights():
    print("=== [Pipeline 5/7] 下載最優權重至本地環境 ===")
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    wsl_local_weight = f"/mnt/g/CODE/handwriteremove/weights/best_exam_remover_hybrid.pth"
    wsl_prod_weight = f"/mnt/g/CODE/handwriteremove/weights/best_exam_remover.pth"
    
    run_wsl(f"{COLAB_BIN} download -s {SESSION} '{REMOTE_WEIGHT_PATH}' '{wsl_local_weight}'")
    run_wsl(f"cp -f '{wsl_local_weight}' '{wsl_prod_weight}'")
    print(f"  ★ 本地權重更新成功: {LOCAL_WEIGHT_PATH}")
    print(f"  ★ 生產主力權重已部署: {PROD_WEIGHT_PATH}")

def release_session():
    print(f"=== [Pipeline 6/7] 釋放 Colab Session ({SESSION}) ===")
    run_wsl(f"{COLAB_BIN} stop -s {SESSION}")
    print("[Colab] ★ GPU 實例已安全釋放，停止計費。")

def validate_on_local_exam():
    print("=== [Pipeline 7/7] 本地端真實試卷端到端驗收推論 ===")
    val_script = BASE_DIR / "tools/validate_hybrid_model.py"
    if val_script.exists():
        subprocess.run(["uv", "run", "python", str(val_script)], cwd=str(BASE_DIR))
    print("🎉 端到端一體化管線圓滿大功告成！")

def main():
    start_time = time.time()
    ensure_colab_session()
    sync_latest_tools_and_anchors()
    wait_for_examink_ready()
    execute_hybrid_training()
    download_and_deploy_weights()
    release_session()
    validate_on_local_exam()
    elapsed = (time.time() - start_time) / 60
    print(f"\n==========================================")
    print(f"🚀 全流程總計耗時: {elapsed:.1f} 分鐘")
    print(f"==========================================")

if __name__ == "__main__":
    main()
