# 考卷手寫筆跡去除工具 (DIS IS-Net)

基於 Hugging Face [`Inoob/DIS-Handwriting-Remover`](https://huggingface.co/Inoob/DIS-Handwriting-Remover) 的高精度考卷筆跡消除工具。

本專案採用 **DIS (Dichotomous Image Segmentation) IS-Net** 深度學習模型，能精準區分「印刷字體/幾何圖形」與「手寫算式/筆跡筆痕」，並將手寫筆跡乾淨消除。

---

## 特性與優勢
- **雙尺度融合推論 (Dual-scale Fusion)**：結合「全域巨觀視野」（捕捉大字體與跨題筆劃）與「局部切塊微觀細節」（捕捉邊緣超細算式與註記）。
- **完整 PDF 工作流**：直接輸入考卷 PDF，自動逐頁高解析度渲染、深度學習消除筆跡、並自動合併輸出全新乾淨的 PDF。
- **GPU CUDA 加速**：對齊 PyTorch 2.13.0 + CUDA 12.6，推論極速（NVIDIA GeForce GTX 1650 SUPER 實測單頁約 1~2 秒）。
- **零磁碟額外佔用與純淨邊界**：
  - 快取統一保留於 G 槽（`G:\.cache\uv` 與 `G:\.cache\huggingface`）。
  - 模型權重放置於專案內部目錄 `weights/isnet.pth`。

---

## 快速開始

### 1. 安裝與同步環境
專案使用 `uv` 管理：

```powershell
uv sync
```

### 2. 執行去筆跡

#### 處理 PDF 考卷（推薦）
```powershell
# 自動逐頁去筆跡並匯出成全新的 cleaned PDF
uv run python remover.py -i 20251025101835564.pdf -o cleaned_exam.pdf
```

#### 處理單張圖片
```powershell
# 處理單張圖片並同時儲存手寫遮罩以供檢視
uv run python remover.py -i output_images/page_001.png -o output/page_001.png --save-mask
```

#### 批次處理整個資料夾
```powershell
uv run python remover.py -i output_images/ -o output/
```

---

## 常用參數說明

| 參數 | 預設值 | 說明 |
| :--- | :--- | :--- |
| `-i, --input` | `20251025101835564.pdf` | 輸入 PDF 檔名、圖片檔名或圖片資料夾 |
| `-o, --output` | `output` | 輸出 PDF 檔名或輸出目錄 |
| `--mode` | `dual` | 推論模式：`dual` (推薦，全域+切塊雙尺度)、`fast` (全圖快速推論)、`tile` (純切塊) |
| `-t, --threshold`| `0.30` | 筆跡二值化門檻（0.1 ~ 0.9，數值越小消除越徹底） |
| `-d, --dilate` | `1` | 遮罩膨脹半徑次數（預設 1，消除抗鋸齒邊緣殘留） |
| `--dpi` | `150` | PDF 逐頁渲染解析度 DPI（建議 150 ~ 200） |
| `--save-mask` | `False` | 是否儲存手寫筆跡的二值遮罩 (PNG) |
| `--device` | 自動偵測 | 推論運算裝置 (`cuda` 或 `cpu`) |