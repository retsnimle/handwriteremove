#!/usr/bin/env python3
"""
高解析度全頁在線合成 PyTorch 資料集 (On-the-fly Full-Page Dataset)
專為 Colab T4 雲端極速訓練設計：
  1. 100% 原生全頁直出 (Zero Rescaling / Zero Cropping)：
     PDF 在 150 DPI 下渲染出多少像素 (W x H)，就輸出多少像素 (W x H)！
  2. 嚴格對齊題目卷版面語意：
     作答括號高覆蓋填答、選項圈選、題幹文字刪除塗改、留白演算大草稿。
  3. 動態 Batch-Level Padding：
     由 collate_fn 自動向右下補白至 32 倍數，確保長寬整除且完全不拉伸變形。
"""

import os
import random
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from synthetic_engine_v2 import ExamSynthesizer

class OnTheFlyFullPageDataset(Dataset):
    def __init__(
        self,
        assets_root="assets",
        clean_pdfs_root="clean_pdfs",
        total_steps=1500,
        dpi=150,
        is_train=True
    ):
        self.dpi = dpi
        self.engine = ExamSynthesizer(assets_root=assets_root, clean_pdfs_root=clean_pdfs_root, dpi=dpi)
        self.total_steps = total_steps
        self.is_train = is_train
        self.page_cache = {}  # 記憶體快取避免重複開啟 PDF
        
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return self.total_steps

    def get_clean_page(self, pdf_rel, page_num):
        key = (pdf_rel, page_num)
        if key in self.page_cache:
            return self.page_cache[key].copy()
        clean_bgr = self.engine.render_clean_page(pdf_rel, page_num)
        if len(self.page_cache) < 80:
            self.page_cache[key] = clean_bgr.copy()
        return clean_bgr

    def __getitem__(self, idx):
        item = random.choice(self.engine.catalog)
        raw_clean = self.get_clean_page(item["pdf"], item["page"])
        H, W = raw_clean.shape[:2]

        # 1. 印刷退化 (基於成對實測列印掃描試卷配準校準的高保真物理退化模型)
        target_bgr = self.engine.degrade_clean_print(raw_clean)
        synth_img = target_bgr.copy()
        synth_mask = np.zeros((H, W), dtype=np.uint8)
        sc = self.engine.scale_pt_to_px

        # 15% 機率作為硬性負樣本 (Hard Negative Mining):
        # 保持純淨印刷退化試卷，不疊加任何手寫，真值 Mask 恆為 0，強迫模型徹底杜絕印刷字誤判！
        is_hard_negative = self.is_train and (random.random() < 0.15)

        if not is_hard_negative:
            # 2. 括號作答 (85% ~ 100%)
            brackets = item.get("brackets", [])
            if brackets and self.engine.cat_letters:
                num_ans = max(1, int(len(brackets) * random.uniform(0.85, 1.0)))
                for box in random.sample(brackets, min(num_ans, len(brackets))):
                    fname = random.choice(self.engine.cat_letters)
                    stamp = cv2.imread(os.path.join(self.engine.marks_dir, fname), cv2.IMREAD_UNCHANGED)
                    if stamp is None: continue
                    bx = int(box[0] * sc) + random.randint(-5, 8)
                    by = int(box[1] * sc) + random.randint(-6, 6)
                    self.engine.overlay_rgba_stamp(synth_img, synth_mask, stamp, bx, by, random.uniform(0.85, 1.25), random.uniform(-10, 10))

            # 3. 選項圈選
            options = item.get("options", [])
            if options and self.engine.cat_circles:
                max_c = min(12, len(options))
                num_circles = random.randint(1, max_c)
                for opt_box in random.sample(options, num_circles):
                    fname = random.choice(self.engine.cat_circles)
                    stamp = cv2.imread(os.path.join(self.engine.marks_dir, fname), cv2.IMREAD_UNCHANGED)
                    if stamp is None: continue
                    cx = int((opt_box[0] + opt_box[2]) / 2.0 * sc) - stamp.shape[1] // 2 + random.randint(-4, 4)
                    cy = int((opt_box[1] + opt_box[3]) / 2.0 * sc) - stamp.shape[0] // 2 + random.randint(-4, 4)
                    self.engine.overlay_rgba_stamp(synth_img, synth_mask, stamp, cx, cy, random.uniform(0.85, 1.3), random.uniform(-12, 12))

            # 4. 散落解題註記
            if self.engine.cat_letters and random.random() < 0.9:
                for _ in range(random.randint(5, 15)):
                    fname = random.choice(self.engine.cat_letters)
                    stamp = cv2.imread(os.path.join(self.engine.marks_dir, fname), cv2.IMREAD_UNCHANGED)
                    if stamp is None: continue
                    rx = random.randint(int(W * 0.05), int(W * 0.92))
                    ry = random.randint(int(H * 0.08), int(H * 0.92))
                    self.engine.overlay_rgba_stamp(synth_img, synth_mask, stamp, rx, ry, random.uniform(0.75, 1.15), random.uniform(-15, 15))

            # 5. 題幹刪除線塗改
            text_lines = item.get("text_lines", [])
            if text_lines and self.engine.cat_strikes and random.random() < 0.75:
                for line_box in random.sample(text_lines, random.randint(1, min(5, len(text_lines)))):
                    fname = random.choice(self.engine.cat_strikes)
                    stamp = cv2.imread(os.path.join(self.engine.marks_dir, fname), cv2.IMREAD_UNCHANGED)
                    if stamp is None: continue
                    lx = int(line_box[0] * sc) + random.randint(0, 50)
                    ly = int(line_box[1] * sc) + random.randint(-4, 4)
                    self.engine.overlay_rgba_stamp(synth_img, synth_mask, stamp, lx, ly, random.uniform(0.8, 1.25), random.uniform(-4, 4))

            # 6. 批改紅筆標記
            if self.engine.cat_marks and random.random() < 0.6:
                for _ in range(random.randint(1, 4)):
                    fname = random.choice(self.engine.cat_marks)
                    stamp = cv2.imread(os.path.join(self.engine.marks_dir, fname), cv2.IMREAD_UNCHANGED)
                    if stamp is None: continue
                    rx = random.randint(int(W * 0.04), int(W * 0.88))
                    ry = random.randint(int(H * 0.06), int(H * 0.90))
                    self.engine.overlay_rgba_stamp(synth_img, synth_mask, stamp, rx, ry, random.uniform(0.7, 1.1), random.uniform(-15, 15))

            # 7. 演算推導與草稿
            if self.engine.layer_paths:
                for _ in range(random.randint(3, 6)):
                    l_path = random.choice(self.engine.layer_paths)
                    layer = cv2.imread(l_path, cv2.IMREAD_UNCHANGED)
                    if layer is None: continue
                    lh, lw = layer.shape[:2]
                    cw = random.randint(min(280, lw), min(800, lw))
                    ch = random.randint(min(220, lh), min(650, lh))
                    cx = random.randint(0, lw - cw)
                    cy = random.randint(0, lh - ch)
                    crop_stamp = layer[cy:cy+ch, cx:cx+cw]
                    if np.sum(crop_stamp[:, :, 3] > 0) > 250:
                        px = random.randint(15, max(16, W - cw - 15))
                        py = random.randint(35, max(36, H - ch - 35))
                        self.engine.overlay_rgba_stamp(synth_img, synth_mask, crop_stamp, px, py, random.uniform(0.8, 1.15), random.uniform(-8, 8))

        # 8. 輕微微歪斜
        if random.random() < 0.5:
            angle = random.uniform(-0.6, 0.6)
            M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
            synth_img = cv2.warpAffine(synth_img, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=(250, 250, 250))
            synth_mask = cv2.warpAffine(synth_mask, M, (W, H), flags=cv2.INTER_NEAREST, borderValue=0)

        # 訓練期微弱色彩抖動
        if self.is_train and random.random() < 0.4:
            alpha = random.uniform(0.95, 1.05)
            beta = random.uniform(-6, 6)
            synth_img = np.clip(synth_img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        # 轉 RGB 並標準化 (100% 原始尺寸，不 resize)
        im_rgb = cv2.cvtColor(synth_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        norm_img = (im_rgb - self.mean) / self.std

        tensor_img = torch.from_numpy(norm_img).permute(2, 0, 1).float()
        tensor_mask = torch.from_numpy((synth_mask > 128).astype(np.float32)).unsqueeze(0)

        return tensor_img, tensor_mask


def pad_to_multiple(x, pad_val, target_h, target_w):
    _, h, w = x.shape
    dh = target_h - h
    dw = target_w - w
    if dh > 0 or dw > 0:
        return F.pad(x, (0, dw, 0, dh), mode="constant", value=pad_val)
    return x


def fullpage_collate_fn(batch):
    """Batch-level padding: 動態向右下補白至 32 倍數"""
    max_h = max(item[0].shape[1] for item in batch)
    max_w = max(item[0].shape[2] for item in batch)
    pad_h = ((max_h + 31) // 32) * 32
    pad_w = ((max_w + 31) // 32) * 32

    batch_imgs = []
    batch_masks = []
    for img, mask in batch:
        batch_imgs.append(pad_to_multiple(img, 0.0, pad_h, pad_w))
        batch_masks.append(pad_to_multiple(mask, 0.0, pad_h, pad_w))

    return torch.stack(batch_imgs, dim=0), torch.stack(batch_masks, dim=0)
