#!/usr/bin/env python3
"""
雙軌混合資料集 (Hybrid Real & Dynamic Synthetic Dataset)
更新版：全面提高【手寫 A B C D E】在題號與作答框中的覆蓋比例！
結合兩大數據源：
1. 真實試卷庫 (ExamInk-Seg): 1,511 對真實學生手寫考卷 + 像素級手寫 Mask + 乾淨底稿
2. 在地全動態合成 (Full Dynamic Scanned): 106 頁臺灣學測/月考底圖 + 85% 高概率題號/作答框【手寫 ABCDE】蓋印 + 雙端微動態印章擾動 + 20% Hard Negative
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

class HybridExamDataset(Dataset):
    def __init__(
        self,
        examink_root="/content/ExamInk-Seg/data",
        scanned_root="/content/scanned_pairs",
        assets_root="/content/assets",
        catalog_path="/content/assets/scanned_catalog.json",
        anchors_path="/content/assets/exam_anchors_catalog.json",
        pseudo_root="output/pseudo_hard_dataset",
        total_steps=1600,
        dpi=150,
        is_train=True,
        real_ratio=0.20,
        pseudo_ratio=0.20
    ):
        self.total_steps = total_steps
        self.is_train = is_train
        self.real_ratio = real_ratio
        self.pseudo_ratio = pseudo_ratio
        self.dpi = dpi

        # ----------------------------------------------------
        # 軌道 3: 飛輪偽標籤數據 (Pseudo Hard Dataset)
        # ----------------------------------------------------
        self.pseudo_items = []
        p_img_dir = os.path.join(pseudo_root, "images")
        if os.path.exists(p_img_dir):
            for ip in sorted(glob.glob(os.path.join(p_img_dir, "*.png"))):
                stem = os.path.splitext(os.path.basename(ip))[0]
                self.pseudo_items.append({
                    "image": ip,
                    "clean": os.path.join(pseudo_root, "clean_gt", f"{stem}.png"),
                    "hw_mask": os.path.join(pseudo_root, "hw_masks", f"{stem}.png"),
                    "text_mask": os.path.join(pseudo_root, "text_masks", f"{stem}.png")
                })
        print(f"[HybridDataset] 載入飛輪偽標籤大字難例: {len(self.pseudo_items)} 組")

        # ImageNet 歸一化常數
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # ----------------------------------------------------
        # 軌道 1: ExamInk-Seg 真實考卷數據
        # ----------------------------------------------------
        split_dir = os.path.join(examink_root, "train" if is_train else "val")
        source_dir = os.path.join(split_dir, "source")
        mask_dir = os.path.join(split_dir, "mask")
        target_dir = os.path.join(split_dir, "target")

        self.real_items = []
        if os.path.exists(source_dir):
            for s_path in sorted(glob.glob(os.path.join(source_dir, "*.*"))):
                fname = os.path.basename(s_path)
                stem, _ = os.path.splitext(fname)
                m_path = os.path.join(mask_dir, f"{stem}.png")
                t_path = os.path.join(target_dir, fname)
                if not os.path.exists(t_path):
                    alt_ext = ".png" if fname.endswith(".jpg") else ".jpg"
                    t_path = os.path.join(target_dir, f"{stem}{alt_ext}")

                if os.path.exists(m_path):
                    self.real_items.append({
                        "source": s_path,
                        "mask": m_path,
                        "target": t_path if os.path.exists(t_path) else None
                    })

        print(f"[HybridDataset] 載入 ExamInk-Seg 真實試卷: {len(self.real_items)} 對 (split={split_dir})")

        # ----------------------------------------------------
        # 軌道 2: 在地動態掃描數據 (配合題號/作答框靶向蓋印)
        # ----------------------------------------------------
        self.scanned_root = scanned_root
        self.page_cache = {}

        if os.path.exists(catalog_path):
            with open(catalog_path, "r", encoding="utf-8") as f:
                self.catalog = json.load(f)
        else:
            self.catalog = []
            if os.path.exists(scanned_root):
                for p in sorted(glob.glob(os.path.join(scanned_root, "*.pdf"))):
                    doc = fitz.open(p)
                    rel_p = os.path.basename(p)
                    for pi in range(doc.page_count):
                        self.catalog.append({"pdf": rel_p, "page": pi})
                    doc.close()

        # 載入題號與作答框錨點
        if not os.path.exists(anchors_path):
            fallback_anchors = os.path.join(assets_root, "exam_anchors_catalog.json")
            if os.path.exists(fallback_anchors):
                anchors_path = fallback_anchors

        if os.path.exists(anchors_path):
            with open(anchors_path, "r", encoding="utf-8") as f:
                anchors_data = json.load(f)
            self.anchors_map = {(it["pdf"], it["page"]): it["anchors"] for it in anchors_data}
            total_anchors = sum(len(it["anchors"]) for it in anchors_data)
            print(f"[HybridDataset] 載入考卷錨點索引: {len(anchors_data)} 頁, 累計錨點 {total_anchors} 個。")
        else:
            self.anchors_map = {}
            print("[HybridDataset] ⚠️ 未找到 exam_anchors_catalog.json，退回全隨機散佈模式。")

        # 載入分類手寫素材：區分「手寫 A B C D E」與「一般記號」
        self.marks_dir = os.path.join(assets_root, "hw_marks")
        letter_marks_path = os.path.join(assets_root, "hw_letter_marks.json")
        symbol_marks_path = os.path.join(assets_root, "hw_symbol_marks.json")
        
        if os.path.exists(letter_marks_path) and os.path.exists(symbol_marks_path):
            with open(letter_marks_path, "r", encoding="utf-8") as f:
                self.letter_marks = json.load(f)
            with open(symbol_marks_path, "r", encoding="utf-8") as f:
                self.symbol_marks = json.load(f)
            self.all_marks = self.letter_marks + self.symbol_marks
            print(f"[HybridDataset] 載入專屬【手寫 A B C D E】庫: {len(self.letter_marks)} 個，記號庫: {len(self.symbol_marks)} 個。")
        else:
            marks_meta_path = os.path.join(self.marks_dir, "marks_meta.json")
            if os.path.exists(marks_meta_path):
                with open(marks_meta_path, "r", encoding="utf-8") as f:
                    self.all_marks = json.load(f)
            else:
                self.all_marks = []
            self.letter_marks = self.all_marks
            self.symbol_marks = self.all_marks

        self.layer_paths = sorted(glob.glob(os.path.join(assets_root, "hw_layers", "*.png")))
        print(f"[HybridDataset] 載入在地掃描底稿: {len(self.catalog)} 頁, 草稿層: {len(self.layer_paths)} 頁")

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

    def apply_background_dynamic_jitter(self, bg_bgr):
        h, w = bg_bgr.shape[:2]
        angle = random.uniform(-1.5, 1.5)
        scale = random.uniform(0.98, 1.02)
        tx = random.uniform(-8, 8)
        ty = random.uniform(-8, 8)

        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty
        bg_jittered = cv2.warpAffine(bg_bgr, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))

        r = random.random()
        if r < 0.35:
            sigma = random.uniform(0.3, 0.7)
            bg_jittered = cv2.GaussianBlur(bg_jittered, (0, 0), sigma)
        elif r < 0.65:
            k = random.uniform(0.2, 0.5)
            blurred = cv2.GaussianBlur(bg_jittered, (0, 0), 1.0)
            bg_jittered = np.clip((1.0 + k) * bg_jittered.astype(np.float32) - k * blurred.astype(np.float32), 0, 255).astype(np.uint8)

        if random.random() < 0.5:
            alpha = random.uniform(0.95, 1.05)
            beta = random.uniform(-5, 5)
            bg_jittered = np.clip(bg_jittered.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        return bg_jittered

    def apply_stamp_dynamic_jitter(self, stamp_rgba):
        sh, sw = stamp_rgba.shape[:2]
        if sh < 6 or sw < 6:
            return stamp_rgba

        res_rgba = stamp_rgba.copy()
        r_morph = random.random()
        k_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        if r_morph < 0.30:
            res_rgba[:, :, 3] = cv2.dilate(res_rgba[:, :, 3], k_morph, iterations=1)
        elif r_morph < 0.60:
            eroded = cv2.erode(res_rgba[:, :, 3], k_morph, iterations=1)
            if np.sum(eroded > 0) > 10:
                res_rgba[:, :, 3] = eroded

        if random.random() < 0.50:
            shear_x = random.uniform(-0.08, 0.08)
            M_shear = np.float32([[1, shear_x, 0], [0, 1, 0]])
            nw = int(sw + abs(shear_x) * sh)
            res_rgba = cv2.warpAffine(res_rgba, M_shear, (nw, sh), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255, 0))

        alpha_scale = random.uniform(0.70, 1.0)
        res_rgba[:, :, 3] = np.clip(res_rgba[:, :, 3].astype(np.float32) * alpha_scale, 0, 255).astype(np.uint8)

        if random.random() < 0.40:
            bgr = res_rgba[:, :, :3]
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-5, 5)) % 180
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.85, 1.15), 0, 255)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * random.uniform(0.90, 1.10), 0, 255)
            res_rgba[:, :, :3] = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        return res_rgba

    def overlay_stamp(self, bg_bgr, bg_mask, stamp_rgba, x, y, scale=1.0, angle=0.0):
        if self.is_train:
            stamp_rgba = self.apply_stamp_dynamic_jitter(stamp_rgba)

        sh, sw = stamp_rgba.shape[:2]
        nw = max(6, int(sw * scale))
        nh = max(6, int(sh * scale))
        resized = cv2.resize(stamp_rgba, (nw, nh), interpolation=cv2.INTER_LINEAR)

        if abs(angle) > 1.0:
            M = cv2.getRotationMatrix2D((nw / 2.0, nh / 2.0), angle, 1.0)
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            bw = int((nh * sin) + (nw * cos))
            bh = int((nh * cos) + (nw * sin))
            M[0, 2] += (bw / 2.0) - nw / 2.0
            M[1, 2] += (bh / 2.0) - nh / 2.0
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

    def sample_real_pair(self):
        item = random.choice(self.real_items)
        src_img = cv2.imread(item["source"], cv2.IMREAD_COLOR)
        raw_mask = cv2.imread(item["mask"], cv2.IMREAD_GRAYSCALE)

        if src_img is None or raw_mask is None:
            return None

        mask = (raw_mask > 100).astype(np.uint8) * 255

        if item["target"] and os.path.exists(item["target"]):
            tgt_img = cv2.imread(item["target"], cv2.IMREAD_COLOR)
            if tgt_img is not None and tgt_img.shape[:2] == src_img.shape[:2]:
                gray_tgt = cv2.cvtColor(tgt_img, cv2.COLOR_BGR2GRAY)
                pure_text_mask = ((gray_tgt < 85) & (mask == 0)).astype(np.uint8)
            else:
                gray_src = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
                pure_text_mask = ((gray_src < 85) & (mask == 0)).astype(np.uint8)
        else:
            gray_src = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
            pure_text_mask = ((gray_src < 85) & (mask == 0)).astype(np.uint8)

        if self.is_train:
            h, w = src_img.shape[:2]
            if random.random() < 0.5:
                angle = random.uniform(-1.0, 1.0)
                M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
                src_img = cv2.warpAffine(src_img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(255,255,255))
                mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
                pure_text_mask = cv2.warpAffine(pure_text_mask, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)

        im_rgb = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        norm_img = (im_rgb - self.mean) / self.std

        tensor_img = torch.from_numpy(norm_img).permute(2, 0, 1).float()
        tensor_mask = torch.from_numpy((mask > 128).astype(np.float32)).unsqueeze(0)
        tensor_pure_text = torch.from_numpy(pure_text_mask.astype(np.float32)).unsqueeze(0)

        return tensor_img, tensor_mask, tensor_pure_text

    def sample_synthetic_pair(self):
        item = random.choice(self.catalog)
        raw_bgr = self.get_scanned_page(item["pdf"], item["page"])

        if self.is_train:
            bg_bgr = self.apply_background_dynamic_jitter(raw_bgr)
        else:
            bg_bgr = raw_bgr.copy()

        H, W = bg_bgr.shape[:2]
        synth_img = bg_bgr.copy()
        synth_mask = np.zeros((H, W), dtype=np.uint8)

        gray_bg = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2GRAY)
        text_core_mask = (gray_bg < 85).astype(np.uint8)

        is_hard_negative = self.is_train and (random.random() < 0.20)

        if not is_hard_negative:
            # 1. 草稿層
            if self.layer_paths:
                for _ in range(random.randint(2, 4)):
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
                                           angle=random.uniform(-8, 8))

            # 2. ★ 靶向錨點蓋印：高概率覆蓋作答框與題號！
            page_anchors = self.anchors_map.get((item["pdf"], item["page"]), [])
            if page_anchors:
                brackets = [a for a in page_anchors if a["type"] == "bracket"]
                qnums = [a for a in page_anchors if a["type"] == "qnum"]
                
                # 對於作答括號：真實學生幾乎每題都會作答！抽取 85% ~ 100% 的括號！
                if brackets:
                    # ★ 學生真實考場行為：作答率 100%，所有作答框皆確實作答！
                    sampled_brackets = list(brackets)
                else:
                    sampled_brackets = []
                
                # 對於純題號：若有括號，偶爾圈選/筆記題號 (1~4 個)；若無括號 (純題號考卷)，抽 60%~85% 做答案
                if qnums:
                    if brackets:
                        q_count = min(len(qnums), random.randint(1, 4))
                    else:
                        q_count = min(len(qnums), max(4, int(len(qnums) * random.uniform(0.60, 0.85))))
                    sampled_qnums = random.sample(qnums, q_count)
                else:
                    sampled_qnums = []

                # --- 蓋印作答括號 ---
                for anc in sampled_brackets:
                    # 90% 概率蓋上真實手寫 A B C D E 字母
                    use_letter = (len(self.letter_marks) > 0) and (random.random() < 0.90)
                    if use_letter:
                        m = random.choice(self.letter_marks)
                    else:
                        m = random.choice(self.symbol_marks) if self.symbol_marks else random.choice(self.all_marks)

                    stamp = cv2.imread(os.path.join(self.marks_dir, m["file"]), cv2.IMREAD_UNCHANGED)
                    if stamp is None: continue
                    sh, sw = stamp.shape[:2]

                    # 括號蓋印：70% 正常尺度，30% 豪邁大字 (scale 1.20 ~ 1.70) 自然溢出覆蓋括號
                    if random.random() < 0.30:
                        scale = random.uniform(1.20, 1.70)
                    else:
                        scale = random.uniform(0.85, 1.15)
                    nw = int(sw * scale)
                    nh = int(sh * scale)

                    # 以括號中心為基準落筆
                    ix1, iy1, ix2, iy2 = anc["inner_bbox"]
                    cx = (ix1 + ix2) // 2
                    cy = (iy1 + iy2) // 2
                    rx = cx - nw // 2 + random.randint(-6, 6)
                    ry = cy - nh // 2 + random.randint(-6, 6)

                    self.overlay_stamp(synth_img, synth_mask, stamp, rx, ry,
                                       scale=scale, angle=random.uniform(-10, 10))

                # --- 蓋印題號 ---
                for anc in sampled_qnums:
                    # 題號落筆：若是字母答案，提供多種真實學生行為分佈
                    use_letter = (len(self.letter_marks) > 0) and (random.random() < (0.85 if not brackets else 0.50))
                    if use_letter:
                        m = random.choice(self.letter_marks)
                    else:
                        m = random.choice(self.symbol_marks) if self.symbol_marks else random.choice(self.all_marks)

                    stamp = cv2.imread(os.path.join(self.marks_dir, m["file"]), cv2.IMREAD_UNCHANGED)
                    if stamp is None: continue
                    sh, sw = stamp.shape[:2]

                    qx1, qy1, qx2, qy2 = anc["bbox"]
                    qw, qh = max(10, qx2 - qx1), max(10, qy2 - qy1)

                    if use_letter:
                        mode = random.random()
                        if mode < 0.40:
                            # 模式 A: 答案寫在題號左側空白 (正常尺寸)
                            scale = random.uniform(0.85, 1.15)
                            nw, nh = int(sw * scale), int(sh * scale)
                            rx = qx1 - nw - random.randint(4, 15)
                            ry = qy1 + (qh - nh) // 2 + random.randint(-4, 4)
                        elif mode < 0.80:
                            # 模式 B: 豪邁大字直接跨越覆蓋在題號與作答區之上 (scale 1.25 ~ 1.85)
                            scale = random.uniform(1.25, 1.85)
                            nw, nh = int(sw * scale), int(sh * scale)
                            rx = qx1 + (qw - nw) // 2 + random.randint(-10, 20)
                            ry = qy1 + (qh - nh) // 2 + random.randint(-15, 15)
                        else:
                            # 模式 C: 正常尺寸手寫字母直接覆蓋在題號上
                            scale = random.uniform(0.90, 1.20)
                            nw, nh = int(sw * scale), int(sh * scale)
                            rx = qx1 + (qw - nw) // 2 + random.randint(-4, 4)
                            ry = qy1 + (qh - nh) // 2 + random.randint(-4, 4)
                    else:
                        # 覆蓋/圈記在題號上 (可稍微偏大)
                        scale = random.uniform(0.90, 1.35)
                        nw, nh = int(sw * scale), int(sh * scale)
                        rx = qx1 + (qw - nw) // 2 + random.randint(-4, 4)
                        ry = qy1 + (qh - nh) // 2 + random.randint(-4, 4)

                    self.overlay_stamp(synth_img, synth_mask, stamp, rx, ry,
                                       scale=scale, angle=random.uniform(-12, 12))

            # 3. 散落印章
            if self.all_marks:
                for _ in range(random.randint(2, 4)):
                    m = random.choice(self.all_marks)
                    stamp = cv2.imread(os.path.join(self.marks_dir, m["file"]), cv2.IMREAD_UNCHANGED)
                    if stamp is None: continue
                    sh, sw = stamp.shape[:2]
                    rx = random.randint(10, max(11, W - sw - 10))
                    ry = random.randint(15, max(16, H - sh - 15))
                    self.overlay_stamp(synth_img, synth_mask, stamp, rx, ry,
                                       scale=random.uniform(0.80, 1.25),
                                       angle=random.uniform(-12, 12))

        # 稍微膨脹手寫 mask (kernel 3x3)，確保手寫筆劃邊緣的過渡像素不會被算作純印刷文字遭受 3.0x 誤殺罰分！
        synth_mask_dilated = cv2.dilate(synth_mask, np.ones((3, 3), np.uint8), iterations=1)
        pure_text_mask = (text_core_mask > 0) & (synth_mask_dilated == 0)

        im_rgb = cv2.cvtColor(synth_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        norm_img = (im_rgb - self.mean) / self.std

        tensor_img = torch.from_numpy(norm_img).permute(2, 0, 1).float()
        tensor_mask = torch.from_numpy((synth_mask > 128).astype(np.float32)).unsqueeze(0)
        tensor_pure_text = torch.from_numpy(pure_text_mask.astype(np.float32)).unsqueeze(0)

        return tensor_img, tensor_mask, tensor_pure_text

    def sample_pseudo_pair(self):
        item = random.choice(self.pseudo_items)
        img = cv2.imread(item["image"], cv2.IMREAD_COLOR)
        hw_m = cv2.imread(item["hw_mask"], cv2.IMREAD_GRAYSCALE)
        txt_m = cv2.imread(item["text_mask"], cv2.IMREAD_GRAYSCALE)

        if img is None or hw_m is None or txt_m is None:
            return None

        if self.is_train and random.random() < 0.5:
            h, w = img.shape[:2]
            angle = random.uniform(-1.0, 1.0)
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(255,255,255))
            hw_m = cv2.warpAffine(hw_m, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
            txt_m = cv2.warpAffine(txt_m, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)

        im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        norm_img = (im_rgb - self.mean) / self.std

        tensor_img = torch.from_numpy(norm_img).permute(2, 0, 1).float()
        tensor_mask = torch.from_numpy((hw_m > 128).astype(np.float32)).unsqueeze(0)
        tensor_pure_text = torch.from_numpy((txt_m > 128).astype(np.float32)).unsqueeze(0)
        return tensor_img, tensor_mask, tensor_pure_text

    def __getitem__(self, idx):
        r = random.random()
        # 1. 軌道 3: 20% 偽標籤大字難例
        if len(self.pseudo_items) > 0 and r < self.pseudo_ratio:
            res = self.sample_pseudo_pair()
            if res is not None:
                return res

        # 2. 軌道 1: 20% ExamInk-Seg 真實試卷
        if len(self.real_items) > 0 and r < (self.pseudo_ratio + self.real_ratio):
            res = self.sample_real_pair()
            if res is not None:
                return res

        # 3. 軌道 2: 60% 在地全動態包覆合成
        return self.sample_synthetic_pair()

def hybrid_collate_fn(batch):
    patch_size = 768
    crop_imgs, crop_masks, crop_texts = [], [], []

    for img, mask, text in batch:
        _, H, W = img.shape
        if H >= patch_size and W >= patch_size:
            hw_ys, hw_xs = torch.where(mask[0] > 0.5)
            
            # 決定垂直裁切中心
            if len(hw_ys) > 50 and random.random() < 0.85:
                c_idx = random.randint(0, len(hw_ys) - 1)
                cy, cx = hw_ys[c_idx].item(), hw_xs[c_idx].item()
                y1 = int(np.clip(cy - patch_size // 2, 0, H - patch_size))
            else:
                cx = random.randint(0, W)
                y1 = random.randint(0, H - patch_size)

            # ★ 欄位感知水平對齊 (保證題號與完整題幹納入 768px 視野)
            if W > 1600:
                if cx < W // 2:
                    x1 = random.randint(30, min(140, W - patch_size))
                else:
                    x1 = random.randint(min(W - patch_size, 1220), min(W - patch_size, 1320))
            else:
                x1 = random.randint(0, min(120, W - patch_size))
            
            crop_imgs.append(img[:, y1:y1+patch_size, x1:x1+patch_size])
            crop_masks.append(mask[:, y1:y1+patch_size, x1:x1+patch_size])
            crop_texts.append(text[:, y1:y1+patch_size, x1:x1+patch_size])
        else:
            pad_h = max(0, patch_size - H)
            pad_w = max(0, patch_size - W)
            crop_imgs.append(F.pad(img, (0, pad_w, 0, pad_h), value=1.0))
            crop_masks.append(F.pad(mask, (0, pad_w, 0, pad_h), value=0.0))
            crop_texts.append(F.pad(text, (0, pad_w, 0, pad_h), value=0.0))

    return torch.stack(crop_imgs), torch.stack(crop_masks), torch.stack(crop_texts)
