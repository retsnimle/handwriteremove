#!/usr/bin/env python3
"""
考卷手寫筆跡去除與結構重繪生產工具 (Dual-Scale ResNet34-UNet + LaMa Inpainting)
Production-Ready Handwriting Removal and Structural Inpainting Pipeline

架構特點：
1. 第一階段（分割）：ResNet34-UNet 雙尺度金字塔 (1.0x 原生 + 0.5x 全域宏觀)
   - 捕捉跨距 250px 巨型手寫筆跡，杜絕機率坍塌
   - 內建印刷文字抗鋸齒護盾 (Printed Shield)，嚴密保護題號與選擇題作答括號
2. 第二階段（重繪）：LaMa (Fast Fourier Convolution Inpainting) 結構重繪
   - 解決粗暴塗白的死斑斷線問題，自動沿著幾何結構修復表格線、坐標軸與紙張自然底色
3. 端到端 PDF / 圖片流式處理：
   - 支援單張圖片、批次目錄與多頁完整考卷 PDF 輸出
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

import segmentation_models_pytorch as smp


class ExamHandwritingRemover:
    def __init__(
        self,
        seg_weights: str = "weights/best_exam_remover_hybrid.pth",
        lama_weights: str = "weights/big-lama.pt",
        device: Optional[str] = None,
        use_lama: bool = True
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[ExamRemover] 初始化運算裝置: {self.device}")

        # 1. 載入三軌微調 ResNet34-UNet 筆跡分割模型
        if not os.path.exists(seg_weights):
            fallback_weights = "weights/best_exam_remover.pth"
            if os.path.exists(fallback_weights):
                seg_weights = fallback_weights
            else:
                raise FileNotFoundError(f"未找到分割模型權重: {seg_weights}")

        print(f"[ExamRemover] 載入 ResNet34-UNet 筆跡分割權重: {seg_weights}")
        self.seg_model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=1
        )
        self.seg_model.load_state_dict(
            torch.load(seg_weights, map_location=self.device, weights_only=False)
        )
        self.seg_model.to(self.device).eval()

        # 2. 載入 LaMa 結構感知重繪 JIT 模型
        self.use_lama = use_lama
        self.lama_model = None
        if self.use_lama:
            if os.path.exists(lama_weights):
                print(f"[ExamRemover] 載入 LaMa 結構重繪模型: {lama_weights}")
                self.lama_model = torch.jit.load(lama_weights, map_location="cpu").to(self.device).eval()
            else:
                print(f"[ExamRemover] ⚠️ 未找到 LaMa 權重 {lama_weights}，將退回形態學紙色修復模式。")
                self.use_lama = False

        self.MEAN = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.STD = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def _infer_single_scale(self, bgr_img: np.ndarray) -> np.ndarray:
        H, W, _ = bgr_img.shape
        t_raw = torch.from_numpy(bgr_img).to(self.device)
        t_rgb = t_raw[..., [2, 1, 0]].permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
        t_norm = (t_rgb - self.MEAN) / self.STD
        pad_h = ((H + 31) // 32) * 32 - H
        pad_w = ((W + 31) // 32) * 32 - W
        t_pad = F.pad(t_norm, (0, pad_w, 0, pad_h), mode='constant', value=0.0)
        with torch.inference_mode():
            logits = self.seg_model(t_pad)
            probs = torch.sigmoid(logits)[0, 0, :H, :W].cpu().numpy()
        return probs

    @staticmethod
    def _guided_filter_gray(I_norm: np.ndarray, p: np.ndarray, r: int = 5, eps: float = 5e-4) -> np.ndarray:
        ksize = (2 * r + 1, 2 * r + 1)
        mean_I = cv2.boxFilter(I_norm, cv2.CV_32F, ksize)
        mean_p = cv2.boxFilter(p, cv2.CV_32F, ksize)
        mean_Ip = cv2.boxFilter(I_norm * p, cv2.CV_32F, ksize)
        cov_Ip = mean_Ip - mean_I * mean_p

        mean_II = cv2.boxFilter(I_norm * I_norm, cv2.CV_32F, ksize)
        var_I = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        mean_a = cv2.boxFilter(a, cv2.CV_32F, ksize)
        mean_b = cv2.boxFilter(b, cv2.CV_32F, ksize)

        q = mean_a * I_norm + mean_b
        return np.clip(q, 0.0, 1.0)

    def inpaint_lama_patch(self, rgb_patch: np.ndarray, mask_patch: np.ndarray) -> np.ndarray:
        H, W, _ = rgb_patch.shape
        pad_h = ((H + 31) // 32) * 32 - H
        pad_w = ((W + 31) // 32) * 32 - W

        t_img = torch.from_numpy(rgb_patch).to(self.device).permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
        t_mask = torch.from_numpy(mask_patch).to(self.device).unsqueeze(0).unsqueeze(0).float().div_(255.0)

        t_img_pad = F.pad(t_img, (0, pad_w, 0, pad_h), mode="reflect")
        t_mask_pad = F.pad(t_mask, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

        with torch.inference_mode():
            out_pad = self.lama_model(t_img_pad, t_mask_pad)

        out_cropped = out_pad[0, :, :H, :W].permute(1, 2, 0).clamp(0, 1).mul(255).byte().cpu().numpy()
        return out_cropped

    def _build_printed_shield(self, gray: np.ndarray, p_10: np.ndarray, p_fused: Optional[np.ndarray] = None) -> np.ndarray:
        """
        幾何字符尺度與鄰域密度感知印刷護盾 (Geometric Density-Aware Character Shield):
        1. 印刷文字 (中文字、英文字母、數字題號、括號) 在 150 DPI 下高度在 12px ~ 36px 之間
        2. 手寫大字、豎線劃記 (如大 D 豎筆長達 56px，長寬比 > 3.0) 絕非印刷字符，不可保護
        3. 鄰域密度檢驗：漢字筆劃周圍一定有大量其他筆劃成群緊挨，而作答括號/留白區中的手寫細豎線周圍是孤立白紙
        4. 相交處機率約束：若神經網絡高置信度判定為手寫 (p_fused > 0.70)，禁止誤吞保護
        """
        H, W = gray.shape[:2]
        bw_dark = (gray < 165) & (p_10 < 0.25)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bw_dark.astype(np.uint8))
        char_shield = np.zeros((H, W), dtype=bool)

        for i in range(1, num_labels):
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            cx, cy = centroids[i]

            # 1. 幾何尺寸過濾
            if h > 38 and (h / max(w, 1)) > 3.0:
                continue  # 細長手寫豎線，剔除！
            if w > 60 and (w / max(h, 1)) > 5.0:
                continue  # 細長手寫橫線/劃記，剔除！

            # 2. 鄰域密度檢驗
            roi_y0, roi_y1 = max(0, int(cy - 12)), min(H, int(cy + 12))
            roi_x0, roi_x1 = max(0, int(cx - 20)), min(W, int(cx + 20))
            neighbor_dark = np.sum((gray[roi_y0:roi_y1, roi_x0:roi_x1] < 165) & (labels[roi_y0:roi_y1, roi_x0:roi_x1] != i))

            if neighbor_dark < 15 and w < 6:
                continue

            char_shield[labels == i] = True

        # 稍微膨脹 1px 保護字體抗鋸齒邊緣
        k_char = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated_char_shield = cv2.dilate(char_shield.astype(np.uint8), k_char) > 0

        # 最終字符護盾：限制在深色且神經網絡沒有以高機率認定為手寫的地方
        if p_fused is not None:
            final_shield = dilated_char_shield & (gray < 215) & (p_fused < 0.70)
        else:
            final_shield = dilated_char_shield & (gray < 215) & (p_10 < 0.35)

        return final_shield

    @staticmethod
    def _merge_proximate_boxes(boxes: np.ndarray, proximity_thresh: int = 48) -> List[List[int]]:
        """
        邊界框空間聚類聚合 (Sweep-line O(N log N))
        將膨脹 proximity_thresh 後相交的相鄰連通塊合成單一 Cluster
        """
        if len(boxes) == 0:
            return []

        expanded = boxes.copy()
        expanded[:, 0] = np.maximum(0, expanded[:, 0] - proximity_thresh)
        expanded[:, 1] = np.maximum(0, expanded[:, 1] - proximity_thresh)
        expanded[:, 2] += proximity_thresh
        expanded[:, 3] += proximity_thresh

        order = np.argsort(expanded[:, 0])
        sorted_boxes = expanded[order]

        clusters = []
        cur = sorted_boxes[0].tolist()

        for box in sorted_boxes[1:]:
            if box[0] <= cur[2]:
                if not (box[3] < cur[1] or box[1] > cur[3]):
                    cur[0] = min(cur[0], box[0])
                    cur[1] = min(cur[1], box[1])
                    cur[2] = max(cur[2], box[2])
                    cur[3] = max(cur[3], box[3])
                    continue
            clusters.append(cur)
            cur = box.tolist()
        clusters.append(cur)
        return clusters

    def _cluster_and_inpaint(self, rgb_img: np.ndarray, erase_mask: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        空間聚合批次重繪：將鄰近小連通塊聚合為連續結構，保證 LaMa FFC 具備足夠全域感受野
        """
        H, W = rgb_img.shape[:2]
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            (erase_mask > 0).astype(np.uint8)
        )

        if num_labels <= 1:
            return rgb_img.copy(), 0

        boxes = []
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < 12:
                continue
            x, y, w, h = (
                stats[i, cv2.CC_STAT_LEFT],
                stats[i, cv2.CC_STAT_TOP],
                stats[i, cv2.CC_STAT_WIDTH],
                stats[i, cv2.CC_STAT_HEIGHT]
            )
            boxes.append([x, y, x + w, y + h])

        if not boxes:
            return rgb_img.copy(), 0

        boxes_arr = np.array(boxes, dtype=np.int32)
        clusters = self._merge_proximate_boxes(boxes_arr, proximity_thresh=48)

        inpainted = rgb_img.copy()
        pad = 32

        for cluster in clusters:
            x0 = max(0, cluster[0] - pad)
            y0 = max(0, cluster[1] - pad)
            x1 = min(W, cluster[2] + pad)
            y1 = min(H, cluster[3] + pad)

            # 確保 FFC 感受野至少 192x192
            crop_w, crop_h = x1 - x0, y1 - y0
            if crop_w < 192 or crop_h < 192:
                cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
                half_w = max(96, crop_w // 2)
                half_h = max(96, crop_h // 2)
                x0 = max(0, cx - half_w)
                y0 = max(0, cy - half_h)
                x1 = min(W, cx + half_w)
                y1 = min(H, cy + half_h)

            crop_rgb = rgb_img[y0:y1, x0:x1]
            crop_mask = erase_mask[y0:y1, x0:x1]

            if not np.any(crop_mask > 0):
                continue

            crop_inpainted = self.inpaint_lama_patch(crop_rgb, crop_mask)
            inner_mask = (crop_mask > 0)[..., None]
            inpainted[y0:y1, x0:x1] = np.where(inner_mask, crop_inpainted, inpainted[y0:y1, x0:x1])

        return inpainted, len(clusters)

    @staticmethod
    def _extract_structural_lines(gray: np.ndarray, min_len: int = 35) -> np.ndarray:
        """
        自適應提取頁面上的印刷表格線與幾何直線
        """
        bw = (gray < 160).astype(np.uint8) * 255
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_len, 1))
        h_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel)
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_len))
        v_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kernel)
        structural = ((h_lines > 0) | (v_lines > 0)).astype(np.uint8) * 255
        k_hit = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        return cv2.dilate(structural, k_hit) > 0

    @staticmethod
    def _extract_diagram_shield(gray: np.ndarray) -> np.ndarray:
        """
        印刷灰階插圖護盾 (Grayscale Diagram and Halftone Illustration Shield):
        基於兩大物理與幾何準則：
        1. 幾何尺度：「灰階圖片的特徵是寬度比較寬」
           - 單個圖元寬度顯著 (w >= 40px, area >= 300px)，且具備二維厚度 (抗鋸齒細線僅 1-2px)
           - 水平掃描線聚類 (Horizontal Sweep-Line Grouping)：並排子圖元 (如 4 個電荷半圓組) 合併後總寬度 >= 120px
        2. 光度均勻度：「印刷灰階的特徵是比較均勻」
           - 中間調範圍：75 <= gray <= 228 (排除紙白 >230 與手寫深墨 <60)
           - 深黑像元拒絕：ratio(gray < 60) <= 4% (手寫塗鴉與鉛筆算式必定包含高反差深黑核心)
           - 局部均勻度：標準差 std <= 42.0 (微融合後印刷網點過渡平滑，手寫雜線方差極高)
        """
        H, W = gray.shape[:2]
        shield = np.zeros((H, W), dtype=bool)

        # 1. 中間調帶提取 + 網點微融合
        midtone_band = (gray >= 75) & (gray <= 228)
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fused = cv2.morphologyEx(midtone_band.astype(np.uint8), cv2.MORPH_CLOSE, k_close)

        # 消除文字抗鋸齒細線 (厚度 < 3px 消除)
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thick_areas = cv2.morphologyEx(fused, cv2.MORPH_OPEN, k_open)

        # 2. 連通分量初篩 (幾何 + 光度雙約束)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thick_areas, connectivity=8)

        candidate_boxes = []
        candidate_labels = []

        for i in range(1, num_labels):
            x, y, w, h, area = stats[i, 0], stats[i, 1], stats[i, 2], stats[i, 3], stats[i, 4]
            if w < 40 or h < 25 or area < 300:
                continue

            roi = gray[y:y+h, x:x+w]
            mask_roi = (labels[y:y+h, x:x+w] == i)
            if not np.any(mask_roi):
                continue

            valid_pixels = roi[mask_roi]

            # 光度準則 A：深黑拒絕
            if np.mean(valid_pixels < 60) > 0.04:
                continue

            # 光度準則 B：印刷均勻度
            if float(np.std(valid_pixels)) > 42.0:
                continue

            candidate_boxes.append([x, y, x + w, y + h])
            candidate_labels.append(i)

            # 單獨達到寬度標準的獨立圖元 (如蛋形導體、螺線管、表格陰影)
            if w >= 55 and area >= 1000:
                shield[labels == i] = True

        # 3. 水平掃描線聚類 (針對並排子圖元，如四個灰階半圓組)
        if len(candidate_boxes) >= 2:
            boxes = np.array(candidate_boxes)
            y_centers = (boxes[:, 1] + boxes[:, 3]) / 2.0
            used = np.zeros(len(boxes), dtype=bool)

            for i in range(len(boxes)):
                if used[i]:
                    continue
                cluster_indices = [i]
                used[i] = True
                for j in range(i + 1, len(boxes)):
                    if used[j]:
                        continue
                    if abs(y_centers[i] - y_centers[j]) < 40:
                        gap = max(0, max(boxes[i, 0], boxes[j, 0]) - min(boxes[i, 2], boxes[j, 2]))
                        if gap < 85:
                            cluster_indices.append(j)
                            used[j] = True

                if len(cluster_indices) >= 2:
                    c_boxes = boxes[cluster_indices]
                    cluster_w = np.max(c_boxes[:, 2]) - np.min(c_boxes[:, 0])
                    if cluster_w >= 120:
                        for idx in cluster_indices:
                            lbl = candidate_labels[idx]
                            shield[labels == lbl] = True

        # 4. 微羽化膨脹 3px
        k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        return cv2.dilate(shield.astype(np.uint8), k_dilate).astype(bool)

    @staticmethod
    def _extract_color_and_vector_shield(bgr_img: np.ndarray, gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        1. 彩色墨水檢測 (Color Ink Mask):
           考卷為黑白印刷，紅筆/藍筆等彩色墨跡 100% 為手寫作答或老師批改。
           提取色相差並膨脹 2px 吞噬邊界羽化粉邊，強制標記擦除，絕不容許護盾保護。
        2. 選項圖示精確護盾 (Option Diagram Shield):
           僅在檢測到 >=4 個等距並排選項圖形塊 (如第 4 題 (A)~(E) 電場線) 的 ROI 內部保護細箭頭，
           徹底杜絕護盾擴散至試卷其餘文字與草稿區。
        """
        H, W = gray.shape[:2]

        # ---------------------------------------------------------
        # A. 彩色墨水提取 (紅筆 / 藍筆)
        # ---------------------------------------------------------
        b = bgr_img[:, :, 0].astype(int)
        g = bgr_img[:, :, 1].astype(int)
        r = bgr_img[:, :, 2].astype(int)

        red_ink = ((r - g > 12) & (r - b > 12) & (r > 60))
        blue_ink = ((b - r > 12) & (b - g > 8) & (b > 60))
        color_ink_raw = (red_ink | blue_ink)
        # 膨脹 2px 徹底吞噬彩色筆跡邊緣淡化輪廓
        k_ink = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        color_ink = cv2.dilate(color_ink_raw.astype(np.uint8), k_ink).astype(bool)

        # ---------------------------------------------------------
        # B. 選項圖示與斜向箭頭精確提取
        # ---------------------------------------------------------
        vector_shield = np.zeros((H, W), dtype=bool)

        # 二值化並以小核聚合成獨立單元
        _, bw = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        blocks = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k)

        num, labels, stats, _ = cv2.connectedComponentsWithStats(blocks)

        comps = []
        for i in range(1, num):
            x, y, w, h, area = stats[i, 0], stats[i, 1], stats[i, 2], stats[i, 3], stats[i, 4]
            if 60 <= w <= 250 and 25 <= h <= 120 and area >= 1000:
                comps.append((x, y, w, h))

        comps.sort(key=lambda c: c[1])

        # 水平聚合：同行且至少 4 個選項單元並排 (如 (A), (B), (C), (D), (E))
        grouped_rows = []
        for c in comps:
            matched = False
            for r_row in grouped_rows:
                if abs(np.mean([it[1] for it in r_row]) - c[1]) < 25:
                    r_row.append(c)
                    matched = True
                    break
            if not matched:
                grouped_rows.append([c])

        for r_row in grouped_rows:
            if len(r_row) >= 4:
                for x, y, w, h in r_row:
                    roi_gray = gray[y:y+h, x:x+w]
                    _, bw_roi = cv2.threshold(roi_gray, 180, 255, cv2.THRESH_BINARY_INV)
                    dist_roi = cv2.distanceTransform(bw_roi, cv2.DIST_L2, 3)
                    thin_roi = (dist_roi > 0) & (dist_roi <= 2.2) & (roi_gray < 180)
                    k_arrow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    thin_dilated = cv2.dilate(thin_roi.astype(np.uint8), k_arrow).astype(bool)
                    vector_shield[y:y+h, x:x+w] |= thin_dilated

        vector_shield[color_ink] = False
        return color_ink, vector_shield



    def clean_page(
        self,
        bgr_img: np.ndarray,
        mode: str = "dual",
        prob_threshold: float = 0.92,
        save_mask: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
        t0 = time.time()
        H, W, _ = bgr_img.shape
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

        # ----------------------------------------------------
        # Stage 1: 分割手寫筆跡 (支援 dual 雙尺度與 fast 單尺度)
        # ----------------------------------------------------
        p_10 = self._infer_single_scale(bgr_img)

        if mode == "dual":
            # 0.5x 全域金字塔推論 (宏觀視野捕獲 250px 大字)
            bgr_half = cv2.resize(bgr_img, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
            p_half = self._infer_single_scale(bgr_half)
            p_05_raw = cv2.resize(p_half, (W, H), interpolation=cv2.INTER_LINEAR)

            # 引導濾波邊緣對齊
            I_darkness = 1.0 - (gray.astype(np.float32) / 255.0)
            p_05_guided = self._guided_filter_gray(I_darkness, p_05_raw, r=5, eps=5e-4)
            p_05_refined = np.where(gray < 242, p_05_guided, 0.0)
        else:
            p_05_refined = np.zeros_like(p_10)

        # 1. 彩色墨水 (紅筆/藍筆) 與 選項向量箭頭護盾
        color_ink, vector_shield = self._extract_color_and_vector_shield(bgr_img, gray)

        # 2. 自適應幾何與密度感知印刷文字護盾 + 灰階插圖護盾
        p_fused = np.maximum(p_10, p_05_refined)
        printed_shield = self._build_printed_shield(gray, p_10, p_fused=p_fused)
        diagram_shield = self._extract_diagram_shield(gray)
        # 印刷灰階插圖絕非彩色筆墨，強制剔除彩色像素
        diagram_shield[color_ink] = False

        # 統一總護盾 (印刷文字 + 灰階插圖 + 選項向量箭頭)
        master_protect_shield = printed_shield | diagram_shield | vector_shield
        # 嚴格禁止護盾保護任何彩色筆跡
        master_protect_shield[color_ink] = False

        # 保護式機率融合：任何被判定為印刷文字或插圖的像素，強制機率歸零，物理阻斷 UNet 幻覺
        p_fused[master_protect_shield] = 0.0

        # 手寫二值擦除遮罩提取：高機率筆跡 OR 彩色墨水 (紅筆/藍筆)，且絕不受護盾阻擋
        raw_erase = ((p_fused > prob_threshold) | color_ink) & (~master_protect_shield) & (gray < 242)
        erase_mask = raw_erase.astype(np.uint8) * 255
        k_3x3_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        erase_mask = cv2.dilate(erase_mask, k_3x3_rect)
        erase_mask[master_protect_shield] = 0

        # 空遮罩保護
        if not np.any(erase_mask > 0):
            elapsed = time.time() - t0
            return bgr_img.copy(), (erase_mask if save_mask else None), {
                "elapsed_total_sec": elapsed,
                "inpaint_time_sec": 0.0,
                "reconstructed_patches": 0,
                "mode": mode
            }

        # ----------------------------------------------------
        # Stage 2: 結構感知分流重繪 (Selective Inpainting)
        # 1. 結構線碰撞筆跡 -> LaMa 空間聚類結構貫通 (Mask 預先膨脹 3px 吞噬殘墨)
        # 2. 純白紙與作答空隙筆跡 -> 自適應紙色羽化填充 (徹底杜絕黑線毛邊腦補)
        # ----------------------------------------------------
        t_inpaint_start = time.time()

        # 自適應紙底色估算
        k_bg = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))
        paper_bg = cv2.morphologyEx(bgr_img, cv2.MORPH_CLOSE, k_bg)
        valid_paper = (gray > 180) & (p_10 < 0.10)
        if np.any(valid_paper):
            paper_median = np.median(bgr_img[valid_paper], axis=0)
            paper_bg = np.maximum(paper_bg, paper_median.astype(np.uint8))
        else:
            paper_bg = np.maximum(paper_bg, 240)

        reconstructed_count = 0

        if self.use_lama and self.lama_model is not None:
            # 偵測版面上的真正印刷結構線 (表格、框線、坐標軸)
            struct_hit_zone = self._extract_structural_lines(gray, min_len=35)

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((erase_mask > 0).astype(np.uint8))
            lama_mask = np.zeros_like(erase_mask)
            paper_mask = np.zeros_like(erase_mask)

            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] < 8:
                    continue
                comp = (labels == i)
                if np.any(comp & struct_hit_zone):
                    lama_mask[comp] = 255
                elif np.any(comp & diagram_shield) and not np.any(comp & color_ink):
                    # 觸及灰階插圖護盾且非彩色墨水：完全不擦除、不重繪，實現 100% 原圖無損輸出
                    continue
                else:
                    paper_mask[comp] = 255

            # 1. 純白紙 / 作答區極速羽化填充 (消除毛邊，不產生任何神經網絡幻覺)
            k_p = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            paper_mask_dilated = cv2.dilate(paper_mask, k_p)
            paper_mask_dilated[master_protect_shield] = 0

            alpha = (paper_mask_dilated > 0).astype(np.float32)
            alpha_blur = cv2.GaussianBlur(alpha, (3, 3), 0)[..., None]
            stage1_bgr = (bgr_img.astype(np.float32) * (1.0 - alpha_blur) + paper_bg.astype(np.float32) * alpha_blur).astype(np.uint8)

            # 2. 結構相交區調用 LaMa 聚類重繪 (預先膨脹吞噬殘墨)
            if np.any(lama_mask > 0):
                k_l = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                lama_mask_dilated = cv2.dilate(lama_mask, k_l)
                lama_mask_dilated[master_protect_shield] = 0

                rgb_stage1 = cv2.cvtColor(stage1_bgr, cv2.COLOR_BGR2RGB)
                inpainted_rgb, reconstructed_count = self._cluster_and_inpaint(rgb_stage1, lama_mask_dilated)
                cleaned_bgr = cv2.cvtColor(inpainted_rgb, cv2.COLOR_RGB2BGR)
            else:
                cleaned_bgr = stage1_bgr
        else:
            # 純形態學自然紙色修復 fallback
            k_p = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            erase_mask_dilated = cv2.dilate(erase_mask, k_p)
            erase_mask_dilated[master_protect_shield] = 0
            alpha = (erase_mask_dilated > 0).astype(np.float32)
            alpha_blur = cv2.GaussianBlur(alpha, (3, 3), 0)[..., None]
            cleaned_bgr = (bgr_img.astype(np.float32) * (1.0 - alpha_blur) + paper_bg.astype(np.float32) * alpha_blur).astype(np.uint8)

        elapsed = time.time() - t0
        inpaint_elapsed = time.time() - t_inpaint_start

        meta = {
            "elapsed_total_sec": elapsed,
            "inpaint_time_sec": inpaint_elapsed,
            "reconstructed_patches": reconstructed_count,
            "mode": mode
        }

        return cleaned_bgr, (erase_mask if save_mask else None), meta

    def process_file(
        self,
        input_path: str,
        output_path: str,
        mode: str = "dual",
        save_mask: bool = False
    ):
        img = cv2.imread(input_path)
        if img is None:
            raise ValueError(f"無法讀取影像：{input_path}")
        clean_img, mask, meta = self.clean_page(img, mode=mode, save_mask=save_mask)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        cv2.imwrite(output_path, clean_img)
        print(f"[ExamRemover] 已儲存乾淨試卷至：{output_path} (耗時 {meta['elapsed_total_sec']:.2f}s)")

        if save_mask and mask is not None:
            mask_path = os.path.splitext(output_path)[0] + "_mask.png"
            cv2.imwrite(mask_path, mask)
            print(f"[ExamRemover] 已儲存手寫遮罩至：{mask_path}")

    def process_pdf(
        self,
        input_pdf: str,
        output_pdf: str,
        dpi: int = 150,
        mode: str = "dual",
        save_mask: bool = False
    ):
        if fitz is None:
            raise ImportError("處理 PDF 需要安裝 PyMuPDF 套件。")

        print(f"[PDF] 正在處理整份 PDF: {input_pdf}")
        doc = fitz.open(input_pdf)
        total_pages = doc.page_count
        print(f"[PDF] 總頁數: {total_pages} 頁，渲染 DPI: {dpi}，模式: {mode}")

        dst_doc = fitz.open()
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)

        try:
            for i in range(total_pages):
                t_page = time.time()
                page = doc.load_page(i)
                pix = page.get_pixmap(matrix=mat, alpha=False)

                img_bytes = pix.tobytes("png")
                nparr = np.frombuffer(img_bytes, np.uint8)
                img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                cleaned_bgr, mask, meta = self.clean_page(img_bgr, mode=mode, save_mask=save_mask)
                h, w = cleaned_bgr.shape[:2]

                # 串流直接以壓縮圖片寫入 PyMuPDF (保持原 PDF 物理頁面尺寸 points)
                new_page = dst_doc.new_page(width=page.rect.width, height=page.rect.height)
                ok, img_buf = cv2.imencode(".png", cleaned_bgr)
                new_page.insert_image(new_page.rect, stream=img_buf.tobytes())

                if save_mask and mask is not None:
                    mask_dir = os.path.join(os.path.dirname(output_pdf) or ".", "masks")
                    os.makedirs(mask_dir, exist_ok=True)
                    m_path = os.path.join(mask_dir, f"page_{i+1:03d}_mask.png")
                    cv2.imwrite(m_path, mask)

                print(f"--> [PDF] 完成第 {i + 1} / {total_pages} 頁 (耗時 {time.time() - t_page:.2f}s, 重繪聚類群 {meta['reconstructed_patches']} 個)")

                del img_bgr, cleaned_bgr, img_buf, pix
                if i % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            os.makedirs(os.path.dirname(output_pdf) or ".", exist_ok=True)
            print(f"[PDF] 正在儲存壓縮 PDF: {output_pdf}...")
            dst_doc.save(output_pdf, garbage=4, deflate=True, clean=True)
            print(f"[PDF] 匯出完成！檔案大小: {os.path.getsize(output_pdf) / (1024*1024):.2f} MB")
        finally:
            doc.close()
            dst_doc.close()


def main():
    parser = argparse.ArgumentParser(
        description="考卷手寫筆跡去除與結構重繪生產工具 (Dual-Scale UNet + LaMa)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input", required=True, help="輸入 PDF 檔案、圖片路徑或圖片資料夾"
    )
    parser.add_argument("-o", "--output", required=True, help="輸出檔案或輸出目錄名稱")
    parser.add_argument(
        "--mode", choices=["dual", "fast"], default="dual",
        help="推論模式：dual (推薦，雙尺度金字塔 + 引導對齊), fast (單尺度極速模式)"
    )
    parser.add_argument(
        "--dpi", type=int, default=150, help="PDF 渲染解析度 DPI (預設 150)"
    )
    parser.add_argument(
        "--no-lama", action="store_true", help="停用 LaMa 重繪，改採形態學背景修復"
    )
    parser.add_argument(
        "--save-mask", action="store_true", help="是否儲存手寫筆跡的二值遮罩 (PNG)"
    )
    parser.add_argument(
        "--seg-weights", default="weights/best_exam_remover_hybrid.pth", help="分割模型權重路徑"
    )
    parser.add_argument(
        "--lama-weights", default="weights/big-lama.pt", help="LaMa 重繪模型權重路徑"
    )

    args = parser.parse_args()

    remover = ExamHandwritingRemover(
        seg_weights=args.seg_weights,
        lama_weights=args.lama_weights,
        use_lama=not args.no_lama
    )

    input_path = args.input
    output_path = args.output

    if input_path.lower().endswith(".pdf"):
        remover.process_pdf(input_path, output_path, dpi=args.dpi, mode=args.mode, save_mask=args.save_mask)
    elif os.path.isdir(input_path):
        os.makedirs(output_path, exist_ok=True)
        for fname in sorted(os.listdir(input_path)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                in_file = os.path.join(input_path, fname)
                out_file = os.path.join(output_path, fname)
                print(f"[Batch] 處理圖片: {fname}")
                remover.process_file(in_file, out_file, mode=args.mode, save_mask=args.save_mask)
    else:
        remover.process_file(input_path, output_path, mode=args.mode, save_mask=args.save_mask)


if __name__ == "__main__":
    main()
