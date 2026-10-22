#!/usr/bin/env python3
"""
真實掃描試卷在線合成資料集 (Real-Scanned Full-Page Dataset On-The-Fly)
直接以使用者列印掃描的 106 頁真實考卷為底圖：
  - 100% 真實紙張紋理、真實印表機碳粉擴散、真實掃描器高頻雜訊
  - 徹底拋棄向量 PDF，徹底終結 Domain Shift！
  - 包含 20% 硬性負樣本（純淨無筆跡真掃描試卷，Mask 恆為 0）
"""

import os
import json
import glob
import random
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import fitz

class ScannedFullPageDataset(Dataset):
    def __init__(
        self,
        scanned_root="scanned_pairs",
        assets_root="assets",
        catalog_path="assets/scanned_catalog.json",
        total_steps=1500,
        dpi=150,
        is_train=True
    ):
        self.dpi = dpi
        self.scanned_root = scanned_root
        self.total_steps = total_steps
        self.is_train = is_train
        self.page_cache = {}

        # 1. 載入 106 頁真實掃描目錄
        if os.path.exists(catalog_path):
            with open(catalog_path, "r", encoding="utf-8") as f:
                self.catalog = json.load(f)
        else:
            self.catalog = []
            for p in sorted(glob.glob(os.path.join(scanned_root, "*.pdf"))):
                doc = fitz.open(p)
                rel_p = os.path.basename(p)
                for pi in range(doc.page_count):
                    self.catalog.append({"pdf": rel_p, "page": pi})
                doc.close()

        # 2. 載入真實手寫記號與草稿層
        self.marks_dir = os.path.join(assets_root, "hw_marks")
        marks_meta_path = os.path.join(self.marks_dir, "marks_meta.json")
        self.marks = []
        if os.path.exists(marks_meta_path):
            with open(marks_meta_path, "r", encoding="utf-8") as f:
                self.marks = json.load(f)

        self.layer_paths = sorted(glob.glob(os.path.join(assets_root, "hw_layers", "*.png")))
        print(f"[ScannedDataset] 載入 {len(self.catalog)} 頁真實掃描試卷，{len(self.marks)} 個記號，{len(self.layer_paths)} 頁草稿層。")

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return self.total_steps

    def get_scanned_page(self, pdf_name, page_num):
        key = (pdf_name, page_num)
        if key in self.page_cache:
            return self.page_cache[key].copy()

        full_pdf_path = os.path.join(self.scanned_root, pdf_name)
        doc = fitz.open(full_pdf_path)
        page = doc[page_num]
        pix = page.get_pixmap(dpi=self.dpi)
        img_bytes = pix.tobytes("png")
        doc.close()

        bgr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if len(self.page_cache) < 80:
            self.page_cache[key] = bgr.copy()
        return bgr

    def overlay_stamp(self, bg_bgr, bg_mask, stamp_rgba, x, y, scale=1.0, angle=0.0):
        sh, sw = stamp_rgba.shape[:2]
        nw = max(6, int(sw * scale))
        nh = max(6, int(sh * scale))
        resized = cv2.resize(stamp_rgba, (nw, nh), interpolation=cv2.INTER_LINEAR)

        if abs(angle) > 1.0:
            M = cv2.getRotationMatrix2D((nw / 2, nh / 2), angle, 1.0)
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            bw = int((nh * sin) + (nw * cos))
            bh = int((nh * cos) + (nw * sin))
            M[0, 2] += (bw / 2) - nw / 2
            M[1, 2] += (bh / 2) - nh / 2
            resized = cv2.warpAffine(resized, M, (bw, bh), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255, 0))
            nw, nh = bw, bh

        H, W = bg_bgr.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(W, x + nw), min(H, y + nh)
        if x2 <= x1 or y2 <= y1:
            return

        sx1, sy1 = x1 - x, y1 - y
        sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)

        sub_stamp = resized[sy1:sy2, sx1:sx2]
        alpha = (sub_stamp[:, :, 3].astype(float) / 255.0)[:, :, None]
        ink_bgr = sub_stamp[:, :, :3].astype(float)

        bg_roi = bg_bgr[y1:y2, x1:x2].astype(float)
        blended = (bg_roi * ink_bgr) / 255.0

        bg_bgr[y1:y2, x1:x2] = np.clip(alpha * blended + (1.0 - alpha) * bg_roi, 0, 255).astype(np.uint8)
        patch_mask = (sub_stamp[:, :, 3] > 18).astype(np.uint8) * 255
        bg_mask[y1:y2, x1:x2] = np.maximum(bg_mask[y1:y2, x1:x2], patch_mask)

    def __getitem__(self, idx):
        item = random.choice(self.catalog)
        raw_bgr = self.get_scanned_page(item["pdf"], item["page"])
        H, W = raw_bgr.shape[:2]

        synth_img = raw_bgr.copy()
        synth_mask = np.zeros((H, W), dtype=np.uint8)

        # 20% 硬性負樣本 (Hard Negative Mining)：
        # 直接使用真實掃描純淨試卷，完全不貼筆跡，Mask 恆為 0，強烈懲罰印刷字誤判！
        is_hard_negative = self.is_train and (random.random() < 0.20)

        if not is_hard_negative:
            # 1. 貼上 3 ~ 6 個真實推導與草稿區塊
            if self.layer_paths:
                for _ in range(random.randint(3, 6)):
                    l_path = random.choice(self.layer_paths)
                    layer = cv2.imread(l_path, cv2.IMREAD_UNCHANGED)
                    if layer is None: continue
                    lh, lw = layer.shape[:2]
                    cw = random.randint(min(260, lw), min(750, lw))
                    ch = random.randint(min(200, lh), min(600, lh))
                    cx = random.randint(0, lw - cw)
                    cy = random.randint(0, lh - ch)
                    crop = layer[cy:cy+ch, cx:cx+cw]
                    if np.sum(crop[:, :, 3] > 0) > 200:
                        px = random.randint(10, max(11, W - cw - 10))
                        py = random.randint(20, max(21, H - ch - 20))
                        self.overlay_stamp(synth_img, synth_mask, crop, px, py,
                                           scale=random.uniform(0.85, 1.15),
                                           angle=random.uniform(-6, 6))

            # 2. 貼上 8 ~ 20 個真實選項圈選、作答、刪除線、批改記號
            if self.marks:
                for _ in range(random.randint(8, 22)):
                    m = random.choice(self.marks)
                    stamp = cv2.imread(os.path.join(self.marks_dir, m["file"]), cv2.IMREAD_UNCHANGED)
                    if stamp is None: continue
                    sh, sw = stamp.shape[:2]
                    rx = random.randint(10, max(11, W - sw - 10))
                    ry = random.randint(15, max(16, H - sh - 15))
                    self.overlay_stamp(synth_img, synth_mask, stamp, rx, ry,
                                       scale=random.uniform(0.80, 1.25),
                                       angle=random.uniform(-10, 10))

        # 訓練期微弱色彩/亮度微抖動
        if self.is_train and random.random() < 0.4:
            alpha = random.uniform(0.96, 1.04)
            beta = random.uniform(-4, 4)
            synth_img = np.clip(synth_img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        # 標準化 (100% 原始尺寸，不 resize)
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

def scanned_collate_fn(batch):
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
