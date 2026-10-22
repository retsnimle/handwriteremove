# 三軌解耦考卷去手寫自動化訓練管線完全手冊
**Anti-Amnesia Production Pipeline Specification for Exam Handwriting Removal**

> **給未來的我（Monika）或接手的開發者：**  
> 當妳因為對話上下文壓縮、換模型或重啟而遺忘先前的除錯歷程時，**只要閱讀並依循這份文件，妳就能 100% 毫無偏差地從零建立、打包、上傳並運行整套雲端 T4 訓練管線**。嚴禁憑直覺修改指令，此處記錄的每一個參數與步驟都是由無數次真實踩坑中總結出的鐵律。

---

## 目錄
1. [問題定義與核心架構哲學](#一-問題定義與核心架構哲學)
2. [本地專案目錄結構全景](#二-本地專案目錄結構全景)
3. [兩大資產包打包 SOP（含可複製代碼）](#三-兩大資產包打包-sop)
4. [一體化 Colab 訓練管線完整腳本](#四-一體化-colab-訓練管線完整腳本)
5. [模型架構、損失函數與 1,500 Steps 訓練配置](#五-模型架構損失函數與-1500-steps-訓練配置)
6. [六大血淚工程避坑指南（嚴禁再犯）](#六-六大血淚工程避坑指南嚴禁再犯)
7. [訓練後端到端推論驗收流程](#七-訓練後端到端推論驗收流程)

---

## 一、 問題定義與核心架構哲學

### 1. 核心問題：跨距包覆型超大號手寫答案（Giant Enclosing Strokes）
學生在考卷題號上以巨大英文字母（如 250px 大「D」）直接騎乘並包裹題號「1.」與選擇題括號「( )」時：
- **局部切片感受野崩潰**：標準 768×768 切片中，有效感受野約 120~150px，神經網路只能看見一條巨大黑弧線，無法將其與印刷圓框區分，機率坍塌至 $P \approx 0.0001$ 導致完全不擦除。
- **交界梯度恐慌（Boundary Dilemma）**：印刷文字誤殺懲罰會迫使模型在手寫與題號接觸處向後退縮，殘留黑斑。

### 2. 解決架構
- **三軌資料集**：真實試卷（20%）+ 在地 106 頁大字包覆合成（72%）+ 飛輪難例偽標籤（8%）。
- **解耦損失函數（DecoupledExamLoss）**：5px 邊緣接觸區 4.0x 召回激勵 + 5px 形態學安全緩衝區（不計算文字誤殺）。
- **雙尺度推論（Dual-Scale TTA）**：0.5x 全域縮放宏觀辨識大字結構 + 1.0x 局部切片精準去筆痕。

---

## 二、 本地專案目錄結構全景

專案根目錄為 `G:\CODE\handwriteremove`（在 WSL 中對應 `/mnt/g/CODE/handwriteremove`）：

```text
G:\CODE\handwriteremove/
├── assets/
│   ├── exam_anchors_catalog.json    # 106 頁在地考卷標定的 354 個題號/括號座標錨點
│   ├── scanned_catalog.json         # 考卷頁碼索引
│   ├── hw_layers/                   # 真實掃描試卷分離出的草稿筆跡圖層
│   └── hw_marks/                    # 獨立手寫筆畫/字母印章
├── scanned_pairs/                   # 4 份在地純淨試卷 PDF (累計 106 頁)
├── losses/
│   └── decoupled_exam_loss.py       # 5px 緩衝 + 4.0x 召回的解耦損失函數
├── output/
│   └── pseudo_hard_dataset/         # 70 組飛輪挖掘之大字難例四元組
│       ├── images/                  # 包含大字遮蓋題號的輸入裁切切片 (.png)
│       ├── clean_gt/                # 乾淨無手寫的考卷真值底稿 (.png)
│       ├── hw_masks/                # 手寫筆跡真實標籤 (.png)
│       └── text_masks/              # 印刷題號文字遮罩 (.png)
├── tools/
│   ├── dataset_dynamic_scanned.py   # 在地動態包覆合成資料集
│   ├── dataset_hybrid.py            # 三軌混合 DataLoader (支援本地與雲端路徑)
│   ├── train_colab_hybrid.py        # 微調訓練主程式
│   ├── run_colab_hybrid_pipeline.sh # 一體化自動訓練管線 Shell 腳本
│   └── dual_scale_remover.py        # 雙尺度全域+局部推論驗收腳本
├── weights/
│   ├── best_exam_remover.pth        # 歷史最優基準模型 (ResNet34-UNet)
│   └── best_exam_remover_hybrid.pth # 最新三軌微調模型權重
├── exam_dynamic_bundle.zip          # [資產包 1] 50.4 MB (考卷底稿與筆劃庫)
└── pseudo_hard_bundle.zip           # [資產包 2] 30.1 MB (70組難例 + 最新程式碼)
```

---

## 三、 兩大資產包打包 SOP

管線**絕對禁止單檔散裝上傳**，必須打包為兩大標準 ZIP 包上傳（體積控制在 30~50MB，保證 100% 傳輸成功）：

### 1. 生成 `pseudo_hard_bundle.zip`（30.1 MB）
此包收錄**所有最新程式碼、解耦損失函數、錨點庫與 70 組飛輪大字難例**。  
執行指令（在 WSL 終端執行）：
```bash
python3 -c "
import os, zipfile

bundle_path = '/mnt/g/CODE/handwriteremove/pseudo_hard_bundle.zip'
if os.path.exists(bundle_path):
    os.remove(bundle_path)

with zipfile.ZipFile(bundle_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    # 1. 核心代碼與配置
    zf.write('/mnt/g/CODE/handwriteremove/tools/dataset_dynamic_scanned.py', 'dataset_dynamic_scanned.py')
    zf.write('/mnt/g/CODE/handwriteremove/tools/dataset_hybrid.py', 'dataset_hybrid.py')
    zf.write('/mnt/g/CODE/handwriteremove/tools/train_colab_hybrid.py', 'train_colab_hybrid.py')
    zf.write('/mnt/g/CODE/handwriteremove/losses/decoupled_exam_loss.py', 'losses/decoupled_exam_loss.py')
    zf.write('/mnt/g/CODE/handwriteremove/assets/exam_anchors_catalog.json', 'assets/exam_anchors_catalog.json')

    # 2. 70 組飛輪偽標籤大字難例 (強制正斜線)
    root_dir = '/mnt/g/CODE/handwriteremove/output/pseudo_hard_dataset'
    for r, dirs, files in os.walk(root_dir):
        for f in files:
            full_p = os.path.join(r, f)
            rel_p = os.path.relpath(full_p, '/mnt/g/CODE/handwriteremove')
            clean_rel = rel_p.replace(chr(92), '/')
            zf.write(full_p, clean_rel)

print('pseudo_hard_bundle.zip 封裝完成！大小:', os.path.getsize(bundle_path) / (1024*1024), 'MB')
"
```

### 2. 確認 `exam_dynamic_bundle.zip`（50.4 MB）
此包為在地試卷底稿基石，包含 `scanned_pairs/*.pdf`（106 頁）與 `assets/hw_layers/*`。  
檢查指令：
```bash
ls -lh /mnt/g/CODE/handwriteremove/exam_dynamic_bundle.zip
```
若需重建，將 `scanned_pairs/` 與 `assets/` 封裝入 `exam_dynamic_bundle.zip` 即可。

---

## 四、 一體化 Colab 訓練管線完整腳本

腳本絕對路徑：`/mnt/g/CODE/handwriteremove/tools/run_colab_hybrid_pipeline.sh`

```bash
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
```

---

## 五、 模型架構、損失函數與 1,500 Steps 訓練配置

| 項目 | 規格 / 設定值 | 說明 |
| :--- | :--- | :--- |
| **Backbone** | `resnet34` | 兼具推論速度與特徵表達能力 |
| **架構** | `smp.Unet(encoder_name="resnet34", in_channels=3, classes=1)` | 標準語意分割 UNet |
| **總訓練步數** | **1,500 Steps** | **15 Epochs × 100 Steps/Epoch**（嚴禁私自縮減步數） |
| **Batch Size** | `4` | 每次迭代隨機裁切 4 個 768×768 切片 |
| **優化器** | `AdamW(lr=8e-5, weight_decay=1e-4)` | 梯度裁剪 `max_norm=1.0` 防止爆炸 |
| **學習率排程** | `CosineAnnealingLR(T_max=15, eta_min=1e-6)` | 餘弦退火平滑收斂 |
| **資料集採樣比** | `在地動態合成: 72%` \| `ExamInk-Seg: 20%` \| `飛輪難例: 8%` | 兼顧自然筆跡與大字難例 |
| **DecoupledExamLoss** | $\mathcal{L} = 0.5 \text{BCE} + 1.0 \text{Dice} + 3.0 \text{TextFP}$ | 5px 安全緩衝區消除交界衝突；接觸交界 4.0x 召回激勵 |

---

## 六、 六大血淚工程避坑指南（嚴禁再犯）

### ❌ 坑 1：誤把 `colab ls` 當成實例檢查指令
- **錯誤寫法**：`if ! colab ls | grep -q "$SESSION"; then ...`
- **真相**：`colab ls` 是「列出遠端 VM 檔案」，輸出永遠是檔案名，導致腳本誤以為實例不存在而無窮重複申請。
- **正解**：查詢實例必須使用 `colab sessions`！

### ❌ 坑 2：忽略 `colab exec` 預設超時（30 秒截斷）
- **錯誤寫法**：`colab exec -s exam-hybrid << 'EOF'`（未帶超時參數）
- **真相**：Colab CLI 預設超時僅 **30.0 秒**。下載大型數據集或執行超過 30 秒的 Python 代碼會直接被拋出 `TimeoutError: Timeout waiting for reply`。
- **正解**：任何資料處理必須設 `--timeout 600`，訓練區塊必須設 `--timeout 14400`（4 小時）。

### ❌ 坑 3：散裝單檔上傳引發 HTTP 500 與大檔 EOF 錯誤
- **錯誤寫法**：使用 `colab upload` 頻繁上傳多個 `.py` 或超過 50MB 的未壓縮 `.pth`。
- **真相**：遠端目錄若未事先 `mkdir`，Jupyter API 會回傳 500 Server Error；且單檔 >50MB 會觸發 SSLEOFError。
- **正解**：所有程式碼與資產**一律先在本機壓成 30~50MB 的標準 ZIP**，一次性上傳後由遠端 Python 解壓。

### ❌ 坑 4：Windows 壓縮檔在 Linux 解壓的路徑反斜線地雷
- **現象**：Windows 生成的 zip 若包含 `output\images\1.png`，Linux 下解壓會變成名為 `output\images\1.png` 的單一平鋪檔案，而不是多層資料夾。
- **正解**：在打包與遠端解壓代碼中，明確執行 `.replace('\\', '/')` 確保目錄階層相容。

### ❌ 坑 5：Allocation refused（孤兒實例殘留佔用額度）
- **現象**：中斷後重新建立實例回報 `Allocation refused (precondition failed)`。
- **正解**：在 WSL 終端執行以下 Python 腳本，透過底層 API 強制釋放雲端孤兒實例：
  ```bash
  python3 -c "
  from colab_cli.common import state
  for a in state.client.list_assignments():
      print('Unassigning:', a.endpoint)
      state.client.unassign(a.endpoint)
  print('全部殘留實例已清除乾淨！')
  "
  ```

### ❌ 坑 6：禁止說謊掩飾腳本缺陷（技術誠信鐵律）
- **教訓**：當新實例跑不通時，不可胡編「以前是復用舊實例」的藉口。每一次管線執行都是全新的 `colab new`，如果報錯，必然是打包清單漏了檔案或參數設錯，務必嚴格調閱日誌反查！

---

## 七、 訓練後端到端推論驗收流程

管線第 [7/7] 步完成後，權重已下載至本地 `weights/best_exam_remover_hybrid.pth`。

### 1. 執行雙尺度推論驗收（大字「D」真實試卷）
在專案根目錄下調用已實作之雙尺度腳本：
```powershell
uv run python tools/dual_scale_remover.py `
  --input "G:\CODE\handwriteremove\20251020202938305.pdf" `
  --page 0 `
  --model "weights/best_exam_remover_hybrid.pth" `
  --output_dir "output/val_hybrid"
```

### 2. 檢驗重點（使用 `read_file` 查驗圖片）：
- **大字覆蓋處**：題號「1.」上方的超大「D」筆劃是否已徹底消除（無灰色半透明殘留）。
- **題號骨架完整度**：題號「1.」與括號「( )」之黑色印刷墨跡是否 100% 完好無損，無任何鋸齒剝落。
- **文字邊緣**：題目字體筆劃是否有被誤擦除的孔洞。

---
*文件編制日期：2026 年 10 月 23 日*  
*維持標準：生產環境（Production-Ready）零造假誠實工程標準*
