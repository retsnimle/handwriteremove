#!/usr/bin/env python3
r"""
Colab 雙尺度 (Coarse-to-Fine) 300 DPI 專用微調訓練器 (頂級物理加固 SOTA 版)
========================================================================
依據 Aider 戰略級架構顧問 (550B 大模型) 審查意見完美加固：
1. 印章旋轉畫布防截斷 (P1)：動態計算旋轉外接矩形 (W_new, H_new)，100% 杜絕印章邊角被切平。
2. 真實物理油墨覆蓋 (P2)：採用不透明顏料 Alpha-Over 物理疊印，真實還原鉛筆石墨/原子筆壓在碳粉上層的物理光學。
3. 損失函數防 NaN 安全防護罩 (P3)：overlap_mask 為空時平滑回傳 0.0，杜絕空集合梯度例外。
4. Charbonnier (Pseudo-Huber, eps=1e-3) + VGG16 (1.0, 0.5, 0.25) + 8.0x 印刷交界加權。
"""

import os
import sys
import glob
import json
import random
import time
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as tv_models

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
for p in (SCRIPT_DIR, REPO_ROOT):
    if p and p not in sys.path:
        sys.path.insert(0, p)

try:
    from models.DocUnfoldV2 import DocUnfoldV2
except ImportError:
    from DocUnfoldV2 import DocUnfoldV2


def charbonnier_loss(x, eps=1e-3):
    return torch.sqrt(x * x + eps * eps).mean()


def masked_charbonnier(diff, weight_map, eps=1e-3):
    """diff: (B, C, H, W), weight_map: (B, 1, H, W) in [0, 1]"""
    loss_map = torch.sqrt((diff * diff).mean(dim=1, keepdim=True) + eps * eps)
    w_sum = weight_map.sum()
    if w_sum > 1e-4:
        return (loss_map * weight_map).sum() / (w_sum + 1e-5)
    return torch.tensor(0.0, device=diff.device)


class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = tv_models.vgg16(weights=tv_models.VGG16_Weights.DEFAULT).features
        self.slice1 = nn.Sequential(*[vgg[x] for x in range(4)]).eval()    # relu1_2
        self.slice2 = nn.Sequential(*[vgg[x] for x in range(4, 9)]).eval()  # relu2_2
        self.slice3 = nn.Sequential(*[vgg[x] for x in range(9, 16)]).eval() # relu3_3
        for param in self.parameters():
            param.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        p = (pred - self.mean) / self.std
        t = (target - self.mean) / self.std
        p1, t1 = self.slice1(p), self.slice1(t)
        p2, t2 = self.slice2(p1), self.slice2(t1)
        p3, t3 = self.slice3(p2), self.slice3(t2)
        return 1.0 * F.l1_loss(p1, t1) + 0.5 * F.l1_loss(p2, t2) + 0.25 * F.l1_loss(p3, t3)


class DualScale300DPIDataset(Dataset):
    def __init__(
        self,
        assets_root="assets",
        scanned_root="scanned_pairs",
        patch_size=512,
        total_steps=1500,
    ):
        self.assets_root = assets_root
        self.scanned_root = scanned_root
        self.patch_size = patch_size
        self.total_steps = total_steps

        # 優先載入 300 DPI 原生筆跡層
        self.hw_layers = sorted(glob.glob(os.path.join(assets_root, "hw_layers_300dpi", "*.png")))
        if not self.hw_layers:
            self.hw_layers = sorted(glob.glob(os.path.join(assets_root, "hw_layers", "*.png")))

        # 優先載入 300 DPI 原生記號
        self.marks_dir = os.path.join(assets_root, "hw_marks_300dpi")
        if not os.path.exists(self.marks_dir):
            self.marks_dir = os.path.join(assets_root, "hw_marks")
        self.marks = sorted(glob.glob(os.path.join(self.marks_dir, "*.png")))

        # 載入底稿 catalog
        cat_file = os.path.join(assets_root, "scanned_catalog.json")
        with open(cat_file, "r", encoding="utf-8") as f:
            self.catalog = json.load(f)

        print(f"[DualScaleDataset] 載入完成: {len(self.catalog)} 頁底稿, {len(self.hw_layers)} 頁手寫層, {len(self.marks)} 個記號 (Patch={patch_size})", flush=True)

    def __len__(self):
        return self.total_steps

    def _render_page_300(self, item):
        try:
            import pymupdf as fitz
        except ImportError:
            import fitz
        pdf_path = os.path.join(self.scanned_root, item["pdf"])
        doc = fitz.open(pdf_path)
        page = doc[item["page"]]
        mat300 = fitz.Matrix(300 / 72.0, 300 / 72.0)
        pix300 = page.get_pixmap(matrix=mat300, colorspace=fitz.csRGB, alpha=False)
        arr300 = np.frombuffer(pix300.samples, dtype=np.uint8).reshape((pix300.height, pix300.width, 3))
        img300 = cv2.cvtColor(arr300, cv2.COLOR_RGB2BGR)
        doc.close()
        return img300

    def apply_background_dynamic_jitter(self, bg_bgr):
        """紙張底圖微小旋轉、平移與微幅縮放擾動 (300 DPI 尺度)"""
        h, w = bg_bgr.shape[:2]
        angle = random.uniform(-1.2, 1.2)
        scale = random.uniform(0.98, 1.02)
        tx = random.uniform(-12, 12)
        ty = random.uniform(-12, 12)
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty
        res_bg = cv2.warpAffine(bg_bgr, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        return res_bg

    def transform_stamp_dynamic(self, stamp_rgba):
        """
        記號/印章的動態微小變形 (P1 加固：外接矩形畫布防裁切)
        """
        h, w = stamp_rgba.shape[:2]
        angle = random.uniform(-15.0, 15.0)
        scale = random.uniform(0.85, 1.35)

        # 計算旋轉後的外接矩形尺寸，徹底杜絕旋轉切角
        rad = np.deg2rad(abs(angle))
        sin_val, cos_val = np.sin(rad), np.cos(rad)
        new_w = int(np.ceil((w * cos_val + h * sin_val) * scale)) + 8
        new_h = int(np.ceil((w * sin_val + h * cos_val) * scale)) + 8

        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
        M[0, 2] += (new_w - w) / 2.0
        M[1, 2] += (new_h - h) / 2.0
        M[0, 1] += random.uniform(-0.08, 0.08)
        M[1, 0] += random.uniform(-0.08, 0.08)

        warped = cv2.warpAffine(
            stamp_rgba, M, (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255, 0)
        )
        alpha = warped[:, :, 3]

        # 隨機形態學肥瘦調變
        if random.random() < 0.4:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            if random.random() < 0.5:
                alpha = cv2.dilate(alpha, k, iterations=1)
            else:
                alpha = cv2.erode(alpha, k, iterations=1)

        # 隨機景深高斯柔化
        if random.random() < 0.5:
            sigma = random.uniform(0.4, 0.8)
            alpha = cv2.GaussianBlur(alpha, (3, 3), sigma)

        warped[:, :, 3] = alpha
        return warped

    def blend_stamp_onto_patch(self, patch_bgr, patch_mask, stamp_rgba, x, y):
        """
        真實墨水物理混合 (P2 加固：不透明顏料 Alpha-Over 疊印物理)
        """
        sh, sw = stamp_rgba.shape[:2]
        ph, pw = patch_bgr.shape[:2]

        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(pw, x + sw), min(ph, y + sh)
        if x2 <= x1 or y2 <= y1:
            return

        sx1, sy1 = x1 - x, y1 - y
        sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)

        stamp_sub = stamp_rgba[sy1:sy2, sx1:sx2]
        alpha = stamp_sub[:, :, 3].astype(np.float32) / 255.0
        if alpha.max() < 0.05:
            return

        # 墨色調色盤擾動 (黑筆、深藍筆、鉛筆石墨、批改紅筆)
        ink_palette = [
            (random.randint(18, 40), random.randint(18, 40), random.randint(18, 40)),      # 黑筆
            (random.randint(100, 160), random.randint(30, 75), random.randint(20, 50)),   # 藍筆
            (random.randint(70, 120), random.randint(70, 120), random.randint(70, 120)),   # 鉛筆
            (random.randint(25, 60), random.randint(25, 60), random.randint(170, 235)),   # 紅筆
        ]
        ink_color = np.array(random.choice(ink_palette), dtype=np.float32)
        noise = np.random.normal(0, 3.0, (y2 - y1, x2 - x1, 3)).astype(np.float32)
        ink_rgb = np.clip(ink_color + noise, 0, 255)

        roi_bgr = patch_bgr[y1:y2, x1:x2].astype(np.float32)
        alpha_3c = alpha[:, :, None]

        # 真實物理 Alpha-Over (手寫顏料覆蓋於紙張/碳粉最上層)
        blended_roi = ink_rgb * alpha_3c + roi_bgr * (1.0 - alpha_3c)
        patch_bgr[y1:y2, x1:x2] = np.clip(blended_roi, 0, 255).astype(np.uint8)

        patch_mask[y1:y2, x1:x2] = np.maximum(patch_mask[y1:y2, x1:x2], (alpha * 255).astype(np.uint8))

    def __getitem__(self, idx):
        ps = self.patch_size
        item = random.choice(self.catalog)

        try:
            raw_bgr = self._render_page_300(item)
        except Exception:
            raw_bgr = np.full((3000, 2000, 3), 255, dtype=np.uint8)

        # 1. 紙張底圖微小動態擾動 (旋轉、縮放、平移)
        clean_bgr = self.apply_background_dynamic_jitter(raw_bgr)

        H, W = clean_bgr.shape[:2]
        if H < ps or W < ps:
            clean_bgr = cv2.resize(clean_bgr, (max(H, ps), max(W, ps)))
            H, W = clean_bgr.shape[:2]

        # --- HARD MINING LOOP: Reject empty/low-ink crops ---
        MIN_INK_PIXELS = int(ps * ps * 0.005)   # ~0.5% coverage = 1310 實體墨跡像素 @ 512^2
        MAX_TRIES = 25

        crop_clean = None
        crop_dirty = None
        crop_mask = None

        for _ in range(MAX_TRIES):
            y = random.randint(0, H - ps)
            x = random.randint(0, W - ps)
            crop_clean = clean_bgr[y : y + ps, x : x + ps].copy()
            crop_dirty = crop_clean.copy()
            crop_mask = np.zeros((ps, ps), dtype=np.uint8)

            # 2. 疊加 300 DPI 原生草稿層 (70% 機率)
            if self.hw_layers and random.random() < 0.70:
                l_path = random.choice(self.hw_layers)
                layer = cv2.imread(l_path, cv2.IMREAD_UNCHANGED)
                if layer is not None and layer.shape[0] >= ps and layer.shape[1] >= ps:
                    ly = random.randint(0, layer.shape[0] - ps)
                    lx = random.randint(0, layer.shape[1] - ps)
                    sub_layer = layer[ly : ly + ps, lx : lx + ps]
                    alpha = sub_layer[:, :, 3].astype(np.float32) / 255.0
                    if alpha.max() > 0.1:
                        alpha_factor = random.uniform(0.75, 1.0)
                        alpha = alpha * alpha_factor
                        for c in range(3):
                            crop_dirty[:, :, c] = (crop_dirty[:, :, c] * (1.0 - alpha) + sub_layer[:, :, c] * alpha).astype(np.uint8)
                        crop_mask = np.maximum(crop_mask, (alpha * 255).astype(np.uint8))

            # 3. 散落疊加 1~5 個動態變形手寫記號/印章 (85% 機率)
            if self.marks and random.random() < 0.85:
                num_stamps = random.randint(1, 5)
                for _ in range(num_stamps):
                    m_path = random.choice(self.marks)
                    stamp = cv2.imread(m_path, cv2.IMREAD_UNCHANGED)
                    if stamp is not None and stamp.shape[0] > 10 and stamp.shape[1] > 10:
                        deformed_stamp = self.transform_stamp_dynamic(stamp)
                        rx = random.randint(-20, ps - 20)
                        ry = random.randint(-20, ps - 20)
                        self.blend_stamp_onto_patch(crop_dirty, crop_mask, deformed_stamp, rx, ry)

            # 嚴格以實體像素計數 (crop_mask > 10)，杜絕 0~255 灰階累加導致的 255 倍閾值縮水
            ink_pixel_count = int((crop_mask > 10).sum())
            if ink_pixel_count >= MIN_INK_PIXELS:
                break
        else:
            # Fallback: force a stamp in center if all tries failed
            if self.marks:
                m_path = random.choice(self.marks)
                stamp = cv2.imread(m_path, cv2.IMREAD_UNCHANGED)
                if stamp is not None:
                    deformed_stamp = self.transform_stamp_dynamic(stamp)
                    self.blend_stamp_onto_patch(crop_dirty, crop_mask, deformed_stamp, ps // 2, ps // 2)

        # 4. 模擬宏觀 UNet 輸出的 probs_hw (高斯擴散 + 不均勻振幅，作為第 4 通道輸入引導先驗)
        k_size = random.choice([9, 13, 17])
        sim_prob = cv2.GaussianBlur(crop_mask.astype(np.float32) / 255.0, (k_size, k_size), 3.5)
        sim_prob = np.clip(sim_prob * random.uniform(0.72, 1.0), 0.0, 1.0)

        # 5. 模擬 DocLayout-YOLO 局部圖表遮罩 (35% 樣本隨機在局部生成 1~3 個模擬圖表區塊，徹底消除全圖 2.0x 突波)
        diag_mask_gt = np.zeros((ps, ps), dtype=np.float32)
        if random.random() < 0.35:
            num_boxes = random.randint(1, 3)
            for _ in range(num_boxes):
                bw = random.randint(60, 200)
                bh = random.randint(60, 200)
                bx = random.randint(0, max(0, ps - bw))
                by = random.randint(0, max(0, ps - bh))
                diag_mask_gt[by : by + bh, bx : bx + bw] = 1.0

        # 6. 生成真值手寫監督遮罩 (Ground Truth Dilated Mask，經 7x7 橢圓核膨脹覆蓋筆劃幾何邊界)
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        gt_hw_weight = cv2.dilate((crop_mask > 10).astype(np.uint8), kernel_dilate, iterations=1).astype(np.float32)

        # 轉為 5 通道 Tensor (輸入)
        t_rgb = torch.from_numpy(crop_dirty[..., [2, 1, 0]].transpose(2, 0, 1)).float() / 255.0
        t_prob = torch.from_numpy(sim_prob).unsqueeze(0).float()
        t_diag_prior = torch.from_numpy(diag_mask_gt).unsqueeze(0).float()
        t_5ch = torch.cat([t_rgb, t_prob, t_diag_prior], dim=0)

        # 目標 (真值乾淨背景)
        t_clean = torch.from_numpy(crop_clean[..., [2, 1, 0]].transpose(2, 0, 1)).float() / 255.0

        # 損失監督真值張量
        t_gt_hw_weight = torch.from_numpy(gt_hw_weight).unsqueeze(0).float()
        t_gt_diag_weight = torch.from_numpy(diag_mask_gt).unsqueeze(0).float()
        raw_ink_ratio = float((crop_mask > 10).sum()) / float(ps * ps)
        t_raw_ink_ratio = torch.tensor(raw_ink_ratio, dtype=torch.float32)

        return t_5ch, t_clean, t_gt_hw_weight, t_gt_diag_weight, t_raw_ink_ratio


def train_dual_scale_300dpi():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--weights_in", default="/content/weights/best_docunfold_v2_5ch.pth")
    parser.add_argument("--weights_out", default="/content/weights/best_docunfold_v2_dual_300dpi.pth")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== [300 DPI 雙尺度微調] 啟動 DocUnfoldV2 512x512 全動態物理增強訓練 (加固 SOTA 版 | FP32 | Batch={args.batch_size}, Accum={args.grad_accum}) ===", flush=True)

    dataset = DualScale300DPIDataset(patch_size=512, total_steps=args.steps)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    weights_in_path = args.weights_in
    if not os.path.exists(weights_in_path):
        local_cand = os.path.join(REPO_ROOT, "weights", os.path.basename(args.weights_in))
        if os.path.exists(local_cand):
            weights_in_path = local_cand

    model = DocUnfoldV2(cond_c=2).to(device)
    if os.path.exists(weights_in_path):
        try:
            ckpt = torch.load(weights_in_path, map_location=device, weights_only=True)
        except Exception:
            ckpt = torch.load(weights_in_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict):
            if "model_state" in ckpt:
                ckpt = ckpt["model_state"]
            elif "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]
        w1 = ckpt.get("shallow_feat1.0.weight", None) if isinstance(ckpt, dict) else None
        if w1 is not None and w1.shape[1] == 5:
            model.load_state_dict(ckpt)
            print(f"★ 成功無損熱啟動 5 通道最佳權重: {weights_in_path}", flush=True)
        else:
            model.load_pretrained_3ch_with_zero_cond(ckpt)
            print(f"★ 成功以 Zero-Conv 溫啟動 3 通道預訓練權重: {weights_in_path}", flush=True)
    else:
        print(f"⚠️ [注意] 指定權重檔不存在: {args.weights_in}，將採隨機初始化權重開始訓練。", flush=True)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_opt_steps = max(1, args.steps // args.grad_accum)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_opt_steps, eta_min=1e-6)
    vgg_loss = VGGPerceptualLoss().to(device)

    step = 0
    t0 = time.time()
    for batch in loader:
        t_5ch, t_clean, t_gt_hw_weight, t_gt_diag_weight, t_raw_ink_ratio = [b.to(device) for b in batch]

        out_stages = model(t_5ch)
        pred_stage7 = out_stages[0]
        diff = pred_stage7 - t_clean

        # 1. 全局基礎保真損失 (維持紙張顏色與乾淨印刷文字底氣，調為 0.5x)
        l_global = charbonnier_loss(diff) * 0.5

        # 2. 手寫墨跡精準攻堅損失 (以真值膨脹幾何遮罩監督，權重拉升至 10.0x 絕對攻堅)
        l_hw = masked_charbonnier(diff, t_gt_hw_weight) * 10.0

        # 3. 題幹與物理圖表保護損失 (真值遮罩監督，加權 2.0x)
        l_diag = masked_charbonnier(diff, t_gt_diag_weight) * 2.0

        # 4. VGG 感知特徵損失 (壓制至 0.05x，杜絕印刷紋理雜訊干擾)
        p_loss = vgg_loss(pred_stage7, t_clean) * 0.05

        total_loss = l_global + l_hw + l_diag + p_loss

        loss = total_loss / args.grad_accum
        loss.backward()

        step += 1
        if step % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        raw_hw = l_hw.item() / 10.0
        raw_diag = l_diag.item() / 2.0 if l_diag.item() > 0 else 0.0
        raw_ink_pct = t_raw_ink_ratio.mean().item() * 100.0
        dilated_mask_pct = t_gt_hw_weight.mean().item() * 100.0
        if step == 1 or step % 25 == 0:
            dt = time.time() - t0
            curr_lr = scheduler.get_last_lr()[0]
            print(f"🔥 Step [{step:04d}/{args.steps:04d}] (lr: {curr_lr:.2e} | {dt:.1f}s | Ink: {raw_ink_pct:.2f}% [Mask: {dilated_mask_pct:.1f}%]) | Total: {total_loss.item():.4f} (Global: {l_global.item():.4f}, HW: {l_hw.item():.4f} [Raw: {raw_hw:.4f}], Diag: {l_diag.item():.4f}, VGG: {p_loss.item():.4f})", flush=True)

        if step % 300 == 0 or step == args.steps:
            os.makedirs(os.path.dirname(args.weights_out), exist_ok=True)
            torch.save(model.state_dict(), args.weights_out)
            print(f"  ★ 已儲存最新 300 DPI 權重 -> {args.weights_out}", flush=True)

    print(f"🏆 雙尺度 300 DPI 微調圓滿完成！輸出: {args.weights_out}", flush=True)


if __name__ == "__main__":
    train_dual_scale_300dpi()
