#!/usr/bin/env python3
"""
真實手寫試卷合成引擎 v2 (Synthetic Engine v2)

特點：
  1. 零偽造：100% 採用使用者提供的真實手寫掃描資產 (42頁手寫層 + 391個元素表記號)，絕無任何程式繪製假線條。
  2. 兩段式退化：
     - 5a 印刷先退化：純淨考卷先施加掃描雜訊、紙色、輕微模糊，對齊手寫素材畫質，並產出乾淨目標 (target)。
     - 5b 合成後微退化：整頁輕微共同歪斜與微弱壓縮，遮罩同步幾何變換。
  3. 版面語意對齊：
     - 作答字母：貼於 brackets 括號內或周圍（模擬壓線題號）。
     - 選項圈選：貼於 options (A)~(E) 標籤上。
     - 刪除線塗改：直接壓在 text_lines 題幹文字行上（困難樣本）。
     - 演算草稿：從 42 頁真實手寫層截取大區塊，貼在留白區或題幹間隙。
"""

import os
import json
import glob
import random
import cv2
import numpy as np
import fitz

class ExamSynthesizer:
    def __init__(self, assets_root="assets", clean_pdfs_root="clean_pdfs", dpi=150):
        self.dpi = dpi
        self.scale_pt_to_px = dpi / 72.0
        self.assets_root = assets_root
        self.clean_pdfs_root = clean_pdfs_root
        
        # 1. 載入版面目錄
        catalog_path = os.path.join(assets_root, "layout_catalog.json")
        with open(catalog_path, "r", encoding="utf-8") as f:
            self.catalog = json.load(f)
            
        # 2. 載入元素表記號中繼資料並分類
        marks_meta_path = os.path.join(assets_root, "hw_marks", "marks_meta.json")
        with open(marks_meta_path, "r", encoding="utf-8") as f:
            self.marks_meta = json.load(f)
            
        self.marks_dir = os.path.join(assets_root, "hw_marks")
        self.categorize_marks()
        
        # 3. 載入整頁手寫層清單
        self.layer_paths = sorted(glob.glob(os.path.join(assets_root, "hw_layers", "*.png")))
        print(f"[Synthesizer v2] 載入 {len(self.catalog)} 頁純淨考卷，"
              f"{len(self.marks_meta)} 個真實手寫記號，{len(self.layer_paths)} 頁整頁草稿層。")

    def categorize_marks(self):
        """依長寬比、顏色、尺寸自動劃分記號類型"""
        self.cat_letters = []    # 字母與數字
        self.cat_strikes = []    # 長條刪除線、大型塗改
        self.cat_circles = []    # 圈選標記
        self.cat_marks = []      # 勾、叉、訂正記號
        
        for m in self.marks_meta:
            w, h = m["w"], m["h"]
            ratio = w / max(1, h)
            mean_b, mean_g, mean_r = m["mean_bgr"]
            is_red = (mean_r > mean_b + 18) and (mean_r > mean_g + 18)
            
            # 刪除線與塗改：長條形 (ratio > 2.2 且寬度 > 70) 或 大型密集塗抹
            if (ratio > 2.2 and w > 70) or (w > 120 and h > 50 and m["ink_px"] > 2500):
                self.cat_strikes.append(m["file"])
            # 勾與叉 (特別是第1頁的大勾大叉，或紅筆大標記)
            elif (w > 130 and h > 130) or (is_red and (w > 80 or h > 80)):
                self.cat_marks.append(m["file"])
            # 圈選 (長寬比 0.7~1.4，尺寸 35~110)
            elif 0.7 <= ratio <= 1.4 and 35 <= w <= 110:
                self.cat_circles.append(m["file"])
            # 作答字母與數字 (尺寸 20~130)
            else:
                self.cat_letters.append(m["file"])
                
        print(f"  分類統計: 字母/數字={len(self.cat_letters)}, 圈選={len(self.cat_circles)}, "
              f"刪除線/塗抹={len(self.cat_strikes)}, 批改勾叉={len(self.cat_marks)}")

    def render_clean_page(self, pdf_rel_path, page_num):
        """以指定 DPI 渲染純淨 PDF 頁面"""
        full_pdf_path = os.path.join(self.clean_pdfs_root, os.path.basename(pdf_rel_path))
        doc = fitz.open(full_pdf_path)
        page = doc[page_num]
        mat = fitz.Matrix(self.scale_pt_to_px, self.scale_pt_to_px)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_bytes = pix.tobytes("png")
        doc.close()
        nparr = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    def degrade_clean_print(self, clean_bgr):
        """基於成對實測列印掃描試卷配準校準的高保真物理退化模型"""
        h, w = clean_bgr.shape[:2]
        
        # 1. 碳粉著墨擴散 (模擬印表機點陣化著墨：墨水面積擴張約 1.41 倍)
        ink = (255.0 - clean_bgr.astype(np.float32)) / 255.0
        power = random.uniform(0.38, 0.50)
        boosted_ink = np.power(ink, power)
        
        # 2. 墨色與紙張階調轉移 (實測 Clean 0~50 -> Scan 80~125)
        min_ink = random.uniform(75.0, 125.0)
        paper = np.array([
            random.uniform(248.0, 254.0),
            random.uniform(249.0, 254.0),
            random.uniform(250.0, 255.0)
        ], dtype=np.float32)
        toned = paper - boosted_ink * (paper - min_ink)
        
        # 3. 掃描鏡頭 PSF 微弱光學模糊
        sigma = random.uniform(0.4, 0.7)
        blurred = cv2.GaussianBlur(toned, (3, 3), sigma)
        
        # 4. 感光元件底噪與碳粉微觀顆粒感
        noise = np.random.normal(0, random.uniform(3.5, 6.0), (h, w, 3)).astype(np.float32)
        degraded = np.clip(blurred + noise, 0, 255).astype(np.uint8)
        return degraded

    def overlay_rgba_stamp(self, bg_bgr, bg_mask, stamp_rgba, x, y, scale=1.0, angle=0.0):
        """正片疊底疊加真實手寫印章，並同步更新真值遮罩"""
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
            resized = cv2.warpAffine(resized, M, (bw, bh), flags=cv2.INTER_LINEAR, borderValue=(255,255,255,0))
            nw, nh = bw, bh

        H, W = bg_bgr.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(W, x + nw), min(H, y + nh)
        if x2 <= x1 or y2 <= y1:
            return
            
        sx1, sy1 = x1 - x, y1 - y
        sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)
        
        sub_stamp = resized[sy1:sy2, sx1:sx2]
        alpha = sub_stamp[:, :, 3].astype(float) / 255.0
        ink_bgr = sub_stamp[:, :, :3].astype(float)
        
        # 正片疊底 (Multiply Blending) 物理吸收
        bg_roi = bg_bgr[y1:y2, x1:x2].astype(float)
        blended = (bg_roi * ink_bgr) / 255.0
        
        alpha_3 = alpha[:, :, None]
        bg_bgr[y1:y2, x1:x2] = np.clip(
            alpha_3 * blended + (1.0 - alpha_3) * bg_roi, 0, 255
        ).astype(np.uint8)
        
        # 真值手寫遮罩：不透明度 > 15 處標記為手寫 (255)
        patch_mask = (sub_stamp[:, :, 3] > 18).astype(np.uint8) * 255
        bg_mask[y1:y2, x1:x2] = np.maximum(bg_mask[y1:y2, x1:x2], patch_mask)

    def generate_sample(self):
        """合成一頁試卷的完整流程"""
        item = random.choice(self.catalog)
        raw_clean = self.render_clean_page(item["pdf"], item["page"])
        
        # 5a. 印刷頁先退化 -> 作為修復目標 target
        target_bgr = self.degrade_clean_print(raw_clean)
        synth_img = target_bgr.copy()
        
        H, W = synth_img.shape[:2]
        synth_mask = np.zeros((H, W), dtype=np.uint8)
        
        sc = self.scale_pt_to_px
        
        # 1. 題號與作答框：幾乎每題都填寫答案 (覆蓋率 85% ~ 100%)
        brackets = item.get("brackets", [])
        if brackets and self.cat_letters:
            # 85% ~ 100% 的作答框都填入答案
            num_ans = max(1, int(len(brackets) * random.uniform(0.85, 1.0)))
            sampled_brackets = random.sample(brackets, min(num_ans, len(brackets)))
            for box in sampled_brackets:
                fname = random.choice(self.cat_letters)
                stamp = cv2.imread(os.path.join(self.marks_dir, fname), cv2.IMREAD_UNCHANGED)
                if stamp is None: continue
                
                # 坐標轉換並模擬手寫偏出括號、壓到題號
                bx = int(box[0] * sc) + random.randint(-5, 8)
                by = int(box[1] * sc) + random.randint(-6, 6)
                scale = random.uniform(0.85, 1.25)
                angle = random.uniform(-10, 10)
                self.overlay_rgba_stamp(synth_img, synth_mask, stamp, bx, by, scale, angle)
                
        # 2. 選項圈選 (依 options，每個題目周圍常有圈選或刪去標記)
        options = item.get("options", [])
        if options and self.cat_circles:
            # 較高比例的選項被圈選 (4 ~ 12 個選項)
            num_circles = random.randint(3, min(12, len(options)))
            sampled_opts = random.sample(options, num_circles)
            for opt_box in sampled_opts:
                fname = random.choice(self.cat_circles)
                stamp = cv2.imread(os.path.join(self.marks_dir, fname), cv2.IMREAD_UNCHANGED)
                if stamp is None: continue
                # 置中套在 (A) ~ (E) 上
                cx = int((opt_box[0] + opt_box[2]) / 2.0 * sc) - stamp.shape[1] // 2 + random.randint(-4, 4)
                cy = int((opt_box[1] + opt_box[3]) / 2.0 * sc) - stamp.shape[0] // 2 + random.randint(-4, 4)
                self.overlay_rgba_stamp(synth_img, synth_mask, stamp, cx, cy, random.uniform(0.85, 1.3), random.uniform(-12, 12))

        # 3. 散落各處的隨機字母或筆記註記 (模擬在題目旁寫 A/B/C、打問號、寫推導註解)
        if self.cat_letters and random.random() < 0.9:
            num_scattered = random.randint(5, 15)
            for _ in range(num_scattered):
                fname = random.choice(self.cat_letters)
                stamp = cv2.imread(os.path.join(self.marks_dir, fname), cv2.IMREAD_UNCHANGED)
                if stamp is None: continue
                # 隨機落在版面左右邊界、或題幹周圍
                rx = random.randint(int(W * 0.05), int(W * 0.92))
                ry = random.randint(int(H * 0.08), int(H * 0.92))
                self.overlay_rgba_stamp(synth_img, synth_mask, stamp, rx, ry, random.uniform(0.75, 1.15), random.uniform(-15, 15))

        # 4. 刪除線與塗抹 (直接壓在 text_lines 題幹上，模擬解題塗改)
        text_lines = item.get("text_lines", [])
        if text_lines and self.cat_strikes and random.random() < 0.7:
            num_strikes = random.randint(1, min(5, len(text_lines)))
            sampled_lines = random.sample(text_lines, num_strikes)
            for line_box in sampled_lines:
                fname = random.choice(self.cat_strikes)
                stamp = cv2.imread(os.path.join(self.marks_dir, fname), cv2.IMREAD_UNCHANGED)
                if stamp is None: continue
                lx = int(line_box[0] * sc) + random.randint(0, 50)
                ly = int(line_box[1] * sc) + random.randint(-4, 4)
                self.overlay_rgba_stamp(synth_img, synth_mask, stamp, lx, ly, random.uniform(0.8, 1.25), random.uniform(-4, 4))

        # 5. 批改勾叉記號 (老師批改紅筆標記)
        if self.cat_marks and random.random() < 0.6:
            num_m = random.randint(1, 4)
            for _ in range(num_m):
                fname = random.choice(self.cat_marks)
                stamp = cv2.imread(os.path.join(self.marks_dir, fname), cv2.IMREAD_UNCHANGED)
                if stamp is None: continue
                rx = random.randint(int(W * 0.04), int(W * 0.88))
                ry = random.randint(int(H * 0.06), int(H * 0.90))
                self.overlay_rgba_stamp(synth_img, synth_mask, stamp, rx, ry, random.uniform(0.7, 1.1), random.uniform(-15, 15))

        # 6. 豐富的計算過程與草稿 (從 42 頁草稿層截取，出現 3 ~ 6 塊，覆蓋空白處與隙縫)
        if self.layer_paths:
            num_scratches = random.randint(3, 6)
            for _ in range(num_scratches):
                l_path = random.choice(self.layer_paths)
                layer = cv2.imread(l_path, cv2.IMREAD_UNCHANGED)
                if layer is None: continue
                lh, lw = layer.shape[:2]
                cw = random.randint(280, min(800, lw))
                ch = random.randint(220, min(650, lh))
                cx = random.randint(0, lw - cw)
                cy = random.randint(0, lh - ch)
                crop_stamp = layer[cy:cy+ch, cx:cx+cw]
                # 只要有墨水就貼
                if np.sum(crop_stamp[:, :, 3] > 0) > 250:
                    px = random.randint(15, max(16, W - cw - 15))
                    py = random.randint(35, max(36, H - ch - 35))
                    self.overlay_rgba_stamp(synth_img, synth_mask, crop_stamp, px, py, random.uniform(0.8, 1.15), random.uniform(-8, 8))

        # 5b. 合成後整張輕微共同退化 (極微微歪斜與輕微壓縮)
        if random.random() < 0.5:
            angle = random.uniform(-1.0, 1.0)
            M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
            synth_img = cv2.warpAffine(synth_img, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=(250, 250, 250))
            synth_mask = cv2.warpAffine(synth_mask, M, (W, H), flags=cv2.INTER_NEAREST, borderValue=0)
            target_bgr = cv2.warpAffine(target_bgr, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=(250, 250, 250))

        return synth_img, synth_mask, target_bgr

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_preview", type=int, default=5, help="生成預覽圖數量")
    args = parser.parse_args()
    
    out_dir = "output/v2_preview"
    os.makedirs(out_dir, exist_ok=True)
    
    engine = ExamSynthesizer()
    print(f"正在生成 {args.num_preview} 張 v2 合成對照預覽圖...")
    
    for i in range(args.num_preview):
        synth, mask, target = engine.generate_sample()
        h, w = synth.shape[:2]
        
        # 產生三聯預覽：輸入圖 | 遮罩標註 (紅色半透明) | 退化後乾淨底稿
        vis_mask = synth.copy()
        vis_mask[mask > 128] = [0, 0, 255]
        
        # 縮放到寬度 700 供檢核
        s = 700.0 / w
        nh, nw = int(h * s), 700
        r_synth = cv2.resize(synth, (nw, nh))
        r_vis = cv2.resize(vis_mask, (nw, nh))
        r_target = cv2.resize(target, (nw, nh))
        
        trio = np.hstack([r_synth, r_vis, r_target])
        save_path = os.path.join(out_dir, f"sample_v2_{i+1:02d}.png")
        cv2.imwrite(save_path, trio)
        print(f"  預覽圖 {i+1} 已輸出: {save_path}")
        
    print(f"🎉 預覽圖已全數儲存至: {out_dir}")

if __name__ == "__main__":
    main()
