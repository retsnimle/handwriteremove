# 雙尺度 300 DPI 原生試卷去手寫與版面修復系統 (Dual-Scale 300 DPI Exam Handwriting Remover)

本專案為專門針對高解析度台灣中學自然科試卷（JIS B4、A4、A3 雙欄）設計的**原生 300 DPI 工業級雙尺度手寫筆跡消除與版面無損修復管線**。
透過 150 DPI 宏觀語意引導結合 300 DPI 微觀 7 階深度物理展開神經網路（`DocUnfoldV2`），精確區分 1px 印刷黑體字、幾何圖表、坐標軸與各類手寫算式、紅藍原子筆、淡鉛筆及巨型大勾批改記號，達成 100% 邊界覆蓋與完全無損的印刷字體保留。

---

## 1. 架構核心：雙尺度粗到細 (Coarse-to-Fine) 300 DPI 修復

傳統 150 DPI 單模型或局部切塊修復在大尺寸試卷面臨「巨大筆劃截斷悖論」（如超過 1600px 的批改大勾或多行跨題刮號在局部視窗內看似印刷直線）。本系統採用兩階段雙尺度架構徹底解決此瓶頸：

```text
               [ 原生 300 DPI PDF 試卷頁面 ]
                             │
            ┌────────────────┴────────────────┐
            ▼ (fitz 150 DPI 零拷貝)           ▼ (fitz 300 DPI 零拷貝)
      [ 宏觀影像 raw_150 ]              [ 微觀影像 raw_300 ]
            │                                 │
       GPU 高斯銳化 (k=0.6)                    │
            │                                 │
       ResNet34-UNet ──► P_hw_150             │
       DocLayout-YOLO ─► M_diag_150           │
            │                                 │
            └─────── 雙線性 / 最近鄰放大 ─────►│
                                              ▼
                                   [ 構建 5 通道輸入張量 ]
                                  (RGB + P_hw + M_diag)
                                              │
                                萬能步長對齊補邊 (stride=384)
                                              │
                                     活躍切塊過濾器
                                    ┌─────────┴─────────┐
                               (空白邊距 / 無墨水)   (手寫筆跡 / 包含圖表)
                                    │                   │
                                 直接透傳          DocUnfoldV2 (FP16 AMP)
                                    │                   │
                                    └─────────┬─────────┘
                                              ▼
                                   [ 2D 漢寧窗平滑加權融合 ]
                                              │
                                    像元級四邊 SOP 終檢
                                              │
                                   PyMuPDF 頁面原點歸零重置
                                              │
                                   [ 輸出完美 300 DPI PDF ]
```

### 核心技術特性
- **5 通道張量特徵調製**：輸入張量由原生 300 DPI RGB（3 通道）、雙線性插值放大之宏觀筆跡機率 $P_{hw}^{300}$（第 4 通道）與最近鄰插值放大之圖表結構遮罩 $M_{diag}^{300}$（第 5 通道）拼接而成，使 7 階展開網絡的每個近端映射步驟均具備全域上下文感知。
- **萬能步長對齊補邊演算法**：針對任意長寬尺寸（A4, JIS B4, A3 及極端比例），採用天花板步長公式：
  $$\mathbf{pad}(\text{dim}) = \max\left( ps,\ \left\lfloor \frac{\text{dim} - ps + \text{stride} - 1}{\text{stride}} \right\rfloor \times \text{stride} + ps \right)$$
  確保最末滑窗嚴格滿足 $x_K + ps \ge \text{dim}$，數學級消除邊界死角與黑邊缺陷。
- **子空間矩陣求逆 FP32 安全防護**：在 `DocUnfold` 的特徵融合塊（`mergeblock`）中，子空間正交投影與矩陣求逆運算被強制限制在單精度 FP32 下執行，杜絕半精度 FP16 數值溢位超過 65504 所引發的 `NaN` 與黑方塊。
- **像元級四邊驗收 SOP**：推論後自動針對右、底、左、頂四條 20px 邊界區域進行黑色像元比例掃描，閾值嚴格要求 $< 0.05$（生產目標 0.00%）。
- **微批次分卷傳輸與同盤原子替換**：主調度器將傳輸檔案切分為 35MB 區塊，免疫 WSL 至 Colab 連線的 SSL EOF 異常；下載後透過 CRC32 檢驗並以 `os.replace` 在同磁碟機上進行原子替換，保證中斷時原檔案毫髮無傷。

---

## 2. 環境安裝與準備

### 系統需求
- 作業系統：Windows 10/11 (含 WSL2 用於 Colab 雲端調度) 或 Linux。
- Python：Python 3.12+。
- 套件管理器：推薦使用 `uv` 進行極速依賴管理。
- 硬體加速：支援 CUDA 之 NVIDIA GPU（推論與訓練支援 Google Colab Tesla T4 15GB VRAM）。

### 依賴安裝
```powershell
# 使用 uv 同步虛擬環境與套件 (推薦)
uv sync

# 或使用標準 pip 安裝
pip install -r requirements.txt
```

---

## 3. 核心模型權重配置 (Core Weights)

本專案運行所需之 3 份核心模型權重皆放置於 `weights/` 目錄：

| 元件名稱 | 模型架構 | 權重檔案名稱 | 檔案大小標準 | 功用說明 |
| :--- | :--- | :--- | :--- | :--- |
| **1. 宏觀筆跡分割器** | ResNet34-UNet | `weights/best_exam_remover.pth` | $\approx 93.4\text{ MB}$ ($> 80\text{ MB}$) | 150 DPI 全頁宏觀推論，輸出全局筆跡機率 $P_{hw}^{150}$ |
| **2. 版面結構剖析器** | DocLayout-YOLOv10 | `weights/doclayout_yolo_docstructbench_imgsz1024.pt` | $\approx 38.8\text{ MB}$ ($> 30\text{ MB}$) | 150 DPI 全頁佈局剖析，輸出圖表與文字結構遮罩 $M_{diag}^{150}$ |
| **3. 微觀深度展開修復器** | DocUnfoldV2 | `weights/best_docunfold_v2_dual_300dpi.pth` | $\approx 46.7\text{ MB}$ ($> 40\text{ MB}$) | 300 DPI 5 通道 7 階條件式物理深度展開網絡，執行無損修復 |

---

## 4. 使用指南 (Usage)

### 4.1 本地管線自我診斷 (Self-Diagnostic Suite)
在進行任何生產或訓練前，可於 5 秒內執行內建診斷套件，驗證幾何補邊不變量、5 通道模型實例化、四邊驗收演算法、核心權重完整性及損失函數抗 NaN 數值穩定性：

```powershell
uv run python -m tools.self_diagnostic
```
通過所有 12 項檢驗時將顯示：
```text
=== ★ Pipeline Self-Verification PASSED! Specification is 100% reproducible. ★ ===
```

### 4.2 雲端原生 300 DPI 微調訓練 (Cloud Training)
主調度器整合訓練素材打包、35MB 分卷上傳、遠端 T4 GPU 訓練與權重自動回傳：

```powershell
# 執行端到端完整管線 (微調訓練 1500 steps + 下載權重 + 啟動生產推論)
python tools/orchestrate_colab_300dpi.py
```
- 自動將 `assets/hw_layers_300dpi/`（54 份真實手寫圖層）、`assets/hw_marks_300dpi/`（429 份真實手寫記號）與 `scanned_pairs/*.pdf`（10 份高解析度底稿）打包為 `train_bundle_300dpi.zip` 上傳。
- 在 Colab 執行 1500 steps 原生 300 DPI 微調（有效批次大小 8，AdamW，CosineAnnealingLR）。
- 訓練完畢自動驗證並下載最新權重至 `weights/best_docunfold_v2_dual_300dpi.pth`。

### 4.3 雲端批次生產推論 (Cloud Batch Inference)
若本地核心權重已存在，可直接調度無人值守微批次推論（預設處理 243 份自然科段考卷）：

```powershell
# 略過訓練，直接進入批次推論生產階段
python tools/orchestrate_colab_300dpi.py --skip_train
```
- 自動掃描目標目錄，以 18 份試卷（約 40MB）為一微批次進行切片打包。
- 上傳至 Colab 執行 `tools/colab_batch_worker_300dpi.py` 雙尺度推論。
- 下載清洗成果壓縮包，透過 `zipfile.testzip()` 驗證 CRC32，並以原子方式覆蓋目標 PDF。
- 支援斷點續傳（進度持久化於 `colab_staging/colab_300dpi_production_progress.json`）。

### 4.4 300 DPI 原始手寫素材提取工具
若需從新的掃描樣本中擴充真實手寫圖層與記號：
```powershell
python tools/extract_handwriting_assets_300dpi.py
```

---

## 5. 專案結構概覽

```text
handwriteremove/
├── assets/                               # 原始訓練資產庫
│   ├── hw_layers_300dpi/                 # 54 份真實 300 DPI 全頁透明筆跡圖層
│   ├── hw_marks_300dpi/                  # 429 個真實手寫記號 (大勾、題號、算式) 與元數據
│   ├── exam_anchors_catalog.json         # 試卷錨點目錄
│   ├── exam_anchors_catalog_points.json  # 試卷錨點座標點目錄
│   └── scanned_catalog.json              # 166 頁乾淨掃描試卷索引
├── docs/
│   └── PRODUCTION_AND_TRAINING_PIPELINE.md # 工業級雙尺度 300 DPI 完整技術規範書
├── models/
│   ├── DocUnfold.py                      # 解耦之 DGUNet 核心主幹與 FP32 子空間投影模組
│   ├── DocUnfoldV2.py                    # 5 通道條件式 7 階深度展開網絡 (Zero-Conv 溫啟動)
│   └── __init__.py
├── scanned_pairs/                        # 10 份高解析度背景考卷掃描檔 (PDF)
├── tools/
│   ├── colab_batch_worker_300dpi.py      # 雲端雙尺度 300 DPI 批次推論核心節點
│   ├── extract_handwriting_assets_300dpi.py # 300 DPI 手寫圖層與記號提取工具
│   ├── orchestrate_colab_300dpi.py       # 雲端生產與訓練端到端總調度器 (微批次/分卷傳輸)
│   ├── self_diagnostic.py                # 快速自動化自我診斷驗證套件 (12 項檢驗)
│   └── train_colab_docunfold_dual_300dpi.py # 雲端雙尺度 300 DPI 原生微調訓練器
├── weights/                              # 核心權重存放目錄 (.gitignore 管理)
├── .gitignore                            # 規範快取、分卷暫存與大型權重忽略規則
├── pyproject.toml                        # 專案套件配置 (uv)
└── README.md
```

---

## 6. 技術規範手冊參照 (Technical Manual Reference)

關於詳細的數學推導、損失函數配比（$0.5\mathcal{L}_{\text{global}} + 10.0\mathcal{L}_{\text{hw}} + 2.0\mathcal{L}_{\text{diag}} + 0.05\mathcal{P}_{\text{vgg}}$）、四重物理光學增廣引擎、抗 NaN 數值守衛與 Colab T4 顯存最佳化等技術細節，請參閱：
👉 **[`docs/PRODUCTION_AND_TRAINING_PIPELINE.md`](docs/PRODUCTION_AND_TRAINING_PIPELINE.md)**
