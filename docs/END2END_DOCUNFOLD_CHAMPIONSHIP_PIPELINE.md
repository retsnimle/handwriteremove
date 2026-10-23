# 端到端物理展開考卷去手寫訓練管線手冊 (DocUnfold + 百度冠軍架構)
**End-to-End Unfolding Network & Championship DeHW Pipeline Specification**

> **給未來的我（Monika）或接手的開發者：**  
> 本手冊記錄了考卷去手寫專案經歷了「二值語意遮罩 + LaMa 兩階段修補」之深刻理論瓶頸後，**全面轉向「端到端物理展開圖層分離（Image-to-Image）」的代際演進**。  
> 依循本手冊，妳可以在 Google Colab（T4/V100/A100）上一鍵啟動最前沿的物理展開網絡訓練，並在重疊壓蓋、淡鉛筆與細宋體文字保護上達到工業級還原精度。

---

## 目錄
1. [架構代差演進：為什麼必須棄用二值遮罩 + LaMa？](#一-架構代差演進為什麼必須棄用二值遮罩--lama)
2. [核心模型架構：DocUnfold 7 階物理展開網絡](#二-核心模型架構docunfold-7-階物理展開網絡)
3. [百度大賽全國第一名方案的改良精髓](#三-百度大賽全國第一名方案的改良精髓)
4. [本地資產包打包 SOP（含可複製代碼）](#四-本地資產包打包-sop)
5. [一體化 Colab 訓練管線完整腳本](#五-一體化-colab-訓練管線完整腳本)
6. [端到端推論與驗收標準](#六-端到端推論與驗收標準)

---

## 一、 架構代差演進：為什麼必須棄用二值遮罩 + LaMa？

在 2026 年 10 月對真實台灣高中段考卷（師大附中地科、化學）的全量實測中，先前的「ResNet34-UNet + LaMa」暴露了不可逾越的**理論天花板**：

1. **二值遮罩（Binary Mask）無法表達重疊語意**：
   - 當學生用原子筆劃掉選項 `(A)`、用鉛筆圈起關鍵字「有溫度」時，在二值遮罩下該像素只能選 0（不擦）或 1（擦除）。
   - 若設為 1，底下的印刷字骨架被連帶挖空送進修補模型；若設為 0，手寫劃記完全保留。
2. **通用圖像修補（LaMa）不是文字重建器**：
   - LaMa 旨在利用周圍隨機紋理平滑填補自然景觀（如抹除電線桿）。但文字是具有高度結構化排版的幾何符號，LaMa 遇到被挖空的字符只會「合情合理地抹成白色背景」，造成嚴重的字體斷肢與殘缺。
3. **端到端圖層分離（Image-to-Image）的代差優勢**：
   - 輸入帶手寫的圖像，神經網絡在潛空間（Latent Space）直接解耦出墨跡層與文字層，**直接輸出 100% 挺拔清晰的乾淨試卷**，徹底根除重繪掏空現象。

---

## 二、 核心模型架構：DocUnfold 7 階物理展開網絡

本專案採納 IEEE TCE (2025/2026) 最新發表之 **DocUnfold**（`models/DocUnfold.py`）：
* **參數量**：**12.15 M**（輕量高效，推論延遲低於傳統兩階段管線）。
* **物理展開機制**：結合了梯度下降模組（GDM）與先驗映射模組（PMM），在 7 個迭代階段（Stage 1 ~ Stage 7）逐級剝離污染墨水，每一階的輸出都比上一階更接近白紙黑字。
* **FP16 相容補丁**：已修復其投影層 `torch.linalg.pinv` 在低精度浮點下崩潰之問題，完美相容 PyTorch AMP 混合精度加速。

---

## 三、 百度大賽全國第一名方案的改良精髓

在 `tools/train_colab_docunfold_championship.py` 中，我們深度融合了百度網盤 AI 大賽全國冠軍的核心訓練秘笈：

1. **剔除 GAN 判別器與 VGG 感知損失**：
   - 考卷是高對比、均勻底色的特殊文檔，GAN 與 VGG 會在純白紙張上產生棋盤格斑駁偽影。新管線純粹由像素重構與邊緣梯度驅動。
2. **非對稱解耦損失（Hole vs Valid Loss）**：
   - **$\mathcal{L}_{\text{Hole}}$（3.0x 權重）**：強制模型在手寫覆蓋區域集中算力進行去墨與字形重構。
   - **$\mathcal{L}_{\text{Valid}}$（1.0x 權重）**：嚴厲懲罰任何對周圍無辜印刷文字與背景的侵蝕。
3. **高頻 Sobel 邊緣保持損失（$\mathcal{L}_{\text{Sobel}}$ 0.5x）**：
   - 強制約束輸出圖與真值圖的水平、垂直高頻梯度，保證宋體/明體印刷漢字與英文字母筆劃挺拔銳利。

$$\mathcal{L}_{\text{total}} = \sum_{j=1}^{7} w_j \Big( 3.0 \cdot \mathcal{L}_{\text{Hole}}(\text{Stage}_j) + 1.0 \cdot \mathcal{L}_{\text{Valid}}(\text{Stage}_j) + 0.5 \cdot \mathcal{L}_{\text{Sobel}}(\text{Stage}_j) \Big)$$

---

## 四、 本地資產包打包 SOP

管線依賴兩大資產包，需在 WSL 終端或 Python 中打包：

### 1. 生成 `pseudo_hard_bundle.zip`
包含最新端到端模型、訓練腳本與在地考卷錨點配置：
```bash
python3 -c "
import os, zipfile
bundle_path = '/mnt/g/CODE/handwriteremove/pseudo_hard_bundle.zip'
if os.path.exists(bundle_path): os.remove(bundle_path)
with zipfile.ZipFile(bundle_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    zf.write('/mnt/g/CODE/handwriteremove/models/DocUnfold.py', 'models/DocUnfold.py')
    zf.write('/mnt/g/CODE/handwriteremove/models/__init__.py', 'models/__init__.py')
    zf.write('/mnt/g/CODE/handwriteremove/tools/train_colab_docunfold_championship.py', 'tools/train_colab_docunfold_championship.py')
    zf.write('/mnt/g/CODE/handwriteremove/tools/dataset_hybrid.py', 'tools/dataset_hybrid.py')
    zf.write('/mnt/g/CODE/handwriteremove/assets/exam_anchors_catalog.json', 'assets/exam_anchors_catalog.json')
    zf.write('/mnt/g/CODE/handwriteremove/assets/scanned_catalog.json', 'assets/scanned_catalog.json')
print('pseudo_hard_bundle.zip 封裝完成！')
"
```

### 2. 檢查 `exam_dynamic_bundle.zip`
包含 106 頁台灣在地試卷底稿與各類手寫筆劃庫（約 50.4 MB）。

---

## 五、 一體化 Colab 訓練管線完整腳本

在 WSL 終端執行一鍵訓練管線：
```bash
bash /mnt/g/CODE/handwriteremove/tools/run_colab_docunfold_pipeline.sh
```

### 關鍵訓練配置參數：
- **GPU 規格**：Google Colab T4 (16GB) 或 V100/A100
- **批次大小 (Batch Size)**：2
- **裁切尺寸 (Patch Size)**：$512 \times 512$
- **初始學習率**：`2e-4`（CosineAnnealing 衰減至 `1e-6`）
- **Epochs**：30（約 3,000 Steps）
- **輸出權重**：`weights/best_docunfold_championship.pth`

---

## 六、 端到端推論與驗收標準

訓練完成後，推論流程將直接以交錯分塊（Overlapping Tiling with Center Crop）運行：
1. **輸入**：整頁掃描考卷（如 $1500 \times 2150$ 像素）。
2. **推論機制**：以 $512 \times 512$ 窗口、64px 反射 Padding 滑動掃描，每次僅保留中間 384px 核心區域。
3. **驗收要點**：
   - 題號左側手寫答案（如大寫 A、B、D）完全消除；
   - 被斜線劃過的選項 `(A)`、被圓圈圈起的題幹關鍵字，**文字完整保留，零斷筆、零掏空**；
   - 畫面純淨無斑駁，單頁推論時間 $< 1.5$ 秒。
