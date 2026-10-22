#!/usr/bin/env python3
"""
雙端全動態真實掃描考卷資料集 (Full Dynamic Dual-Augmentation Dataset On-The-Fly)
第二階段升級版：
1. 引入「跨距騎乘與包覆合成 (synthesize_enclosing_giant_stroke)」：
   - 模式 A (70%): 真實手寫字母印章放大 1.8~3.5 倍 (140~320px)，直接騎乘包覆題號與括號
   - 模式 B (30%): 具筆壓調變的貝茲曲線大圈選 (Pressure-Modulated Bézier Loop)，帶收筆尖角
2. 油墨物理疊印 (blend_ink_over_print): Multiply 色彩增值 + 墨色微噪聲 + 排除核心印刷骨架
3. 動態多尺度視野裁切 (Multi-Scale FoV Crop): 50% 採 1024~1536 大視野縮放，擴展感受野
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

class FullDynamicScannedDataset(Dataset):
    def __init__(
        self,
        scanned_root="scanned_pairs",
        assets_root="assets",
        catalog_path="assets/scanned_catalog.json",
        anchors_path="assets/exam_anchors_catalog.json",
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

        # 2. 載入題號與作答框錨點索引
        if not os.path.exists(anchors_path):
            fallback_anchors = os.path.join(assets_root, "exam_anchors_catalog.json")
            if os.path.exists(fallback_anchors):
                anchors_path = fallback_anchors

        if os.path.exists(anchors_path):
            with open(anchors_path, "r", encoding="utf-8") as f:
                anchors_data = json.load(f)
            self.anchors_map = {(it["pdf"], it["page"]): it["anchors"] for it in anchors_data}
            total_anchors = sum(len(it["anchors"]) for it in anchors_data)
            print(f"[FullDynamicDataset] 載入考卷錨點索引: {len(anchors_data)} 頁, 累計錨點 {total_anchors} 個。")
        else:
            self.anchors_map = {}
            print("[FullDynamicDataset] ⚠️ 未找到 exam_anchors_catalog.json，退回全隨機散佈模式。")

        # 3. 載入手寫印章與記號庫
        self.marks_dir = os.path.join(assets_root, "hw_marks")
        letter_marks_path = os.path.join(assets_root, "hw_letter_marks.json")
        symbol_marks_path = os.path.join(assets_root, "hw_symbol_marks.json")

        if os.path.exists(letter_marks_path) and os.path.exists(symbol_marks_path):
            with open(letter_marks_path, "r", encoding="utf-8") as f:
                self.letter_marks = json.load(f)
            with open(symbol_marks_path, "r", encoding="utf-8") as f:
                self.symbol_marks = json.load(f)
            self.all_marks = self.letter_marks + self.symbol_marks
            print(f"[FullDynamicDataset] 載入專屬【手寫 A B C D E】庫: {len(self.letter_marks)} 個，記號庫: {len(self.symbol_marks)} 個。")
        else:
            marks_meta_path = os.path.join(self.marks_dir, "marks_meta.json")
            if os.path.exists(marks_meta_path):
                with open(marks_meta_path, "r", encoding="utf-8") as f:
                    self.all_marks = json.load(f)
            else:
                self.all_marks = []
            self.letter_marks = self.all_marks
            self.symbol_marks = self.all_marks

        # 4. 載入全頁草稿層
        self.layer_paths = sorted(glob.glob(os.path.join(assets_root, "hw_layers", "*.png")))
        print(f"[FullDynamicDataset] 載入在地掃描底稿: {len(self.catalog)} 頁, 草稿層: {len(self.layer_paths)} 頁")

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return self.total_steps

    def get_page_image(self, pdf_rel_path, page_num):
        key = (pdf_rel_path, page_num)
        if key in self.page_cache:
            return self.page_cache[key].copy()
        
        full_pdf_path = os.path.join(self.scanned_root, pdf_rel_path)
        doc = fitz.open(full_pdf_path)
        page = doc[page_num]
        pix = page.get_pixmap(dpi=self.dpi)
        img_np = np.frombuffer(pix.tobytes("png"), dtype=np.uint8)
        bgr = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        doc.close()
        
        if len(self.page_cache) < 40:
            self.page_cache[key] = bgr.copy()
        return bgr

    def apply_background_dynamic_jitter(self, bg_bgr):
        h, w = bg_bgr.shape[:2]
        angle = random.uniform(-1.0, 1.0)
        scale = random.uniform(0.98, 1.02)
        tx = random.uniform(-6, 6)
        ty = random.uniform(-6, 6)
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty
        res_bg = cv2.warpAffine(bg_bgr, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        return res_bg

    def blend_ink_over_print(self, canvas_bgr, hw_mask_full, pure_text_mask, stamp_alpha, ink_color_bgr, x1, y1):
        """真實原子筆油墨疊印在印刷字上的物理混合"""
        H, W = canvas_bgr.shape[:2]
        sh, sw = stamp_alpha.shape[:2]
        x2, y2 = min(W, x1 + sw), min(H, y1 + sh)
        if x2 <= x1 or y2 <= y1:
            return

        roi_bgr = canvas_bgr[y1:y2, x1:x2].astype(np.float32)
        roi_alpha = stamp_alpha[:(y2 - y1), :(x2 - x1)].astype(np.float32)
        roi_alpha_3c = roi_alpha[..., None]

        noise = np.random.normal(0, 3.5, roi_bgr.shape).astype(np.float32)
        ink_layer = np.clip(np.array(ink_color_bgr, dtype=np.float32) + noise, 0, 255)

        multiply_bgr = (roi_bgr * ink_layer) / 255.0
        blended_roi = np.where(roi_bgr < 180, multiply_bgr, ink_layer)

        out_roi = roi_bgr * (1.0 - roi_alpha_3c) + blended_roi * roi_alpha_3c
        canvas_bgr[y1:y2, x1:x2] = np.clip(out_roi, 0, 255).astype(np.uint8)

        # 更新手寫 GT (排除最核心極黑骨架，避免切斷題號核心)
        roi_text_core = pure_text_mask[y1:y2, x1:x2] > 0
        valid_hw = (roi_alpha > 0.18) & (~roi_text_core)
        hw_mask_full[y1:y2, x1:x2][valid_hw] = 255

    def synthesize_enclosing_giant_stroke(self, canvas_bgr, hw_mask_full, pure_text_mask, anchor_box, letter_stamp_rgba=None):
        """
        模式 A (70%): 將手寫字母放大 1.8~3.5 倍 (140~320px)，直接騎乘包覆題號與括號
        模式 B (30%): 具筆壓調變的貝茲曲線大圈選
        """
        qx1, qy1, qx2, qy2 = anchor_box
        uw, uh = max(24, qx2 - qx1), max(24, qy2 - qy1)
        cx, cy = (qx1 + qx2) // 2, (qy1 + qy2) // 2

        ink_palette = [
            (random.randint(18, 45), random.randint(18, 45), random.randint(18, 45)),     # 黑筆
            (random.randint(95, 155), random.randint(25, 65), random.randint(18, 45)),    # 深藍筆
            (random.randint(70, 125), random.randint(70, 125), random.randint(70, 125)),  # 鉛筆
        ]
        ink_bgr = random.choice(ink_palette)

        if letter_stamp_rgba is not None and random.random() < 0.70:
            target_h = int(uh * random.uniform(1.8, 3.5))
            target_h = np.clip(target_h, 140, 320)
            target_w = int(target_h * random.uniform(0.65, 1.05))

            stamp_resized = cv2.resize(letter_stamp_rgba, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            stamp_alpha = stamp_resized[..., 3].astype(np.float32) / 255.0

            dx = int(random.uniform(-0.15, 0.15) * target_w)
            dy = int(random.uniform(-0.12, 0.12) * target_h)
            sx1 = max(0, cx + dx - target_w // 2)
            sy1 = max(0, cy + dy - target_h // 2)

            self.blend_ink_over_print(canvas_bgr, hw_mask_full, pure_text_mask, stamp_alpha, ink_bgr, sx1, sy1)
        else:
            pad_w = int(uw * random.uniform(1.3, 2.2))
            pad_h = int(uh * random.uniform(1.6, 3.0))
            patch_w, patch_h = pad_w * 2, pad_h * 2
            loop_alpha = np.zeros((patch_h, patch_w), dtype=np.float32)

            num_pts = 120
            total_angle = random.uniform(2.1 * np.pi, 2.6 * np.pi)
            start_angle = random.uniform(0, 2 * np.pi)
            base_thickness = random.uniform(2.5, 5.0)

            pts = []
            for i in range(num_pts):
                t = i / (num_pts - 1)
                theta = start_angle + t * total_angle
                rx = pad_w * (0.65 + 0.12 * np.sin(2 * theta) + 0.08 * t)
                ry = pad_h * (0.65 + 0.12 * np.cos(3 * theta) - 0.05 * t)
                px = int(patch_w // 2 + rx * np.cos(theta))
                py = int(patch_h // 2 + ry * np.sin(theta))
                pts.append((px, py, t))

            for i in range(len(pts) - 1):
                x_a, y_a, t_a = pts[i]
                x_b, y_b, _ = pts[i + 1]
                w_t = max(1, int(round(base_thickness * (0.8 + 0.45 * np.sin(np.pi * t_a)))))
                cv2.line(loop_alpha, (x_a, y_a), (x_b, y_b), 1.0, thickness=w_t, lineType=cv2.LINE_AA)

            loop_alpha = cv2.GaussianBlur(loop_alpha, (3, 3), 0.6)
            sx1 = max(0, cx - patch_w // 2)
            sy1 = max(0, cy - patch_h // 2)
            self.blend_ink_over_print(canvas_bgr, hw_mask_full, pure_text_mask, loop_alpha, ink_bgr, sx1, sy1)

    def multi_scale_random_crop(self, img, hw_mask, text_mask, target_size=768):
        H, W = img.shape[:2]
        if random.random() < 0.50:
            crop_sz = target_size
        else:
            crop_sz = random.choice([1024, 1280, 1536])
            crop_sz = min(crop_sz, H, W)

        y1 = random.randint(0, max(0, H - crop_sz))
        x1 = random.randint(0, max(0, W - crop_sz))

        img_c = img[y1:y1 + crop_sz, x1:x1 + crop_sz]
        hw_c = hw_mask[y1:y1 + crop_sz, x1:x1 + crop_sz]
        txt_c = text_mask[y1:y1 + crop_sz, x1:x1 + crop_sz]

        if crop_sz != target_size:
            img_c = cv2.resize(img_c, (target_size, target_size), interpolation=cv2.INTER_AREA)
            hw_c = cv2.resize(hw_c, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
            txt_c = cv2.resize(txt_c, (target_size, target_size), interpolation=cv2.INTER_NEAREST)

        return img_c, hw_c, txt_c

    def __getitem__(self, idx):
        item = random.choice(self.catalog)
        bg_bgr = self.get_page_image(item["pdf"], item["page"])
        bg_bgr = self.apply_background_dynamic_jitter(bg_bgr)
        H, W = bg_bgr.shape[:2]

        synth_img = bg_bgr.copy()
        synth_mask = np.zeros((H, W), dtype=np.uint8)

        gray = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2GRAY)
        text_core_mask = (gray < 85).astype(np.uint8) * 255

        anchors = self.anchors_map.get((item["pdf"], item["page"]), [])
        qnums = [a for a in anchors if a.get("type") == "qnum"]
        brackets = [a for a in anchors if a.get("type") == "bracket"]

        # 隨機抽取 3~6 個題號進行靶向合成
        if qnums:
            sampled_q = random.sample(qnums, min(len(qnums), random.randint(3, 6)))
            for q in sampled_q:
                # 尋找同一行相鄰的作答括號
                qx1, qy1, qx2, qy2 = q["bbox"]
                adj_bracket = None
                for b in brackets:
                    bx1, by1, bx2, by2 = b["bbox"]
                    if abs(by1 - qy1) < 25 and 0 < (bx1 - qx1) < 140:
                        adj_bracket = b
                        break
                
                # 計算聯合外接框
                if adj_bracket:
                    bx1, by1, bx2, by2 = adj_bracket["bbox"]
                    union_box = (min(qx1, bx1), min(qy1, by1), max(qx2, bx2), max(qy2, by2))
                else:
                    union_box = (qx1, qy1, qx2 + 45, qy2)

                # 35% 機率啟動「跨距騎乘與包覆合成」
                if random.random() < 0.35:
                    stamp_rgba = None
                    if self.letter_marks:
                        m = random.choice(self.letter_marks)
                        stamp_rgba = cv2.imread(os.path.join(self.marks_dir, m["file"]), cv2.IMREAD_UNCHANGED)
                    self.synthesize_enclosing_giant_stroke(synth_img, synth_mask, text_core_mask, union_box, stamp_rgba)
                else:
                    # 原有正常作答與批改
                    if self.all_marks:
                        m = random.choice(self.all_marks)
                        stamp = cv2.imread(os.path.join(self.marks_dir, m["file"]), cv2.IMREAD_UNCHANGED)
                        if stamp is not None:
                            sh, sw = stamp.shape[:2]
                            scale = random.uniform(0.9, 1.25)
                            nw, nh = int(sw * scale), int(sh * scale)
                            rx = qx1 - nw - random.randint(4, 15)
                            ry = qy1 + (qy2 - qy1 - nh) // 2
                            alpha = stamp[..., 3].astype(np.float32) / 255.0
                            stamp_res = cv2.resize(alpha, (nw, nh))
                            self.blend_ink_over_print(synth_img, synth_mask, text_core_mask, stamp_res, (30, 30, 30), rx, ry)

        # 散落草稿筆跡
        if self.layer_paths and random.random() < 0.35:
            lp = random.choice(self.layer_paths)
            layer = cv2.imread(lp, cv2.IMREAD_UNCHANGED)
            if layer is not None:
                lh, lw = layer.shape[:2]
                ly = random.randint(0, max(0, H - lh))
                lx = random.randint(0, max(0, W - lw))
                l_alpha = (layer[..., 3].astype(np.float32) / 255.0) * random.uniform(0.4, 0.8)
                self.blend_ink_over_print(synth_img, synth_mask, text_core_mask, l_alpha, (40, 40, 40), lx, ly)

        # 多尺度動態視野裁切
        crop_img, crop_hw, crop_txt = self.multi_scale_random_crop(synth_img, synth_mask, text_core_mask, target_size=768)

        # 形態學安全緩衝區排除 (3x3 膨脹避免受罰)
        hw_dil = cv2.dilate(crop_hw, np.ones((3, 3), np.uint8))
        pure_text = (crop_txt > 0) & (hw_dil == 0)

        im_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        norm_img = (im_rgb - self.mean) / self.std

        tensor_img = torch.from_numpy(norm_img).permute(2, 0, 1).float()
        tensor_mask = torch.from_numpy((crop_hw > 128).astype(np.float32)).unsqueeze(0)
        tensor_pure_text = torch.from_numpy(pure_text.astype(np.float32)).unsqueeze(0)

        return tensor_img, tensor_mask, tensor_pure_text
