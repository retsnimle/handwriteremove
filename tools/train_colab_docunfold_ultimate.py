#!/usr/bin/env python3
"""
🏆 終極端到端試卷手寫擦除訓練器 (Ultimate DocUnfold + 百度冠軍架構)
Ultimate Championship End-to-End Exam Handwriting Remover for Google Colab

三軌真實數據融合規範：
1. 軌道 1（真實世界數據流 70%）：
   - HW5K 官方核心集 (5,000 組真實成對試卷)
   - ExamInk-Seg 試卷庫 (1,511 組真實成對試卷)
   - 合計 6,511 組真實世界手寫樣本，動態計算差值 Mask，學透真實紙質與石墨擴散
2. 軌道 2（台灣在地動態合成流 25%）：
   - 106 頁在地學測/段考底稿 + 354 個題號括號錨點蓋印 (守住宋體明體與台灣題型排版)
3. 軌道 3（飛輪難例偽標籤流 5%）：
   - 70 組大字騎乘遮蓋題號難例四元組

模型與損失：
- DocUnfold 7 階物理展開網絡 (12.15M, 官方 256x256 黃金尺度)
- 百度冠軍非對稱損失 (Hole 3.0x + Valid 1.0x + Sobel 0.5x)
- 梯度累積 4 步 (等效 Batch Size=4，全精度 Float32 顯存 < 2.5 GB)
"""

import os
import sys
import time
import math
import glob
import random
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# 載入 DocUnfold 模型
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from models.DocUnfold import DocUnfold
except ImportError:
    try:
        from DocUnfold import DocUnfold
    except ImportError:
        sys.path.append(r"g:\CODE\agent\_trash_archive\DocUnfold")
        from models.DocUnfold import DocUnfold

try:
    from tools.dataset_hybrid import HybridExamDataset
except ImportError:
    try:
        from dataset_hybrid import HybridExamDataset
    except ImportError:
        HybridExamDataset = None


# ---------------------------------------------------------------------------
# 1. 百度冠軍級混合損失函數 (Championship DeHW Loss)
# ---------------------------------------------------------------------------
class SobelEdge(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def forward(self, x):
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        gx = F.conv2d(gray, self.kx, padding=1)
        gy = F.conv2d(gray, self.ky, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)


class ChampionshipDocLoss(nn.Module):
    def __init__(self, hole_weight=3.0, valid_weight=1.0, edge_weight=0.5):
        super().__init__()
        self.hole_weight = hole_weight
        self.valid_weight = valid_weight
        self.edge_weight = edge_weight
        self.l1 = nn.L1Loss()
        self.sobel = SobelEdge()

    def forward(self, stages_out, target_clean, hw_mask):
        total_loss = 0.0
        # 逐階衰減權重 (Stage 7 權重最大)
        stage_weights = [1.0, 0.4, 0.25, 0.15, 0.1, 0.05, 0.05]

        target_edge = self.sobel(target_clean)
        mask_3c = hw_mask.repeat(1, 3, 1, 1)

        for idx, pred in enumerate(stages_out):
            w = stage_weights[idx] if idx < len(stage_weights) else 0.05
            pred = torch.clamp(pred, 0.0, 1.0)

            # 百度冠軍分區 L1 (Hole vs Valid)
            loss_hole = self.l1(pred * mask_3c, target_clean * mask_3c)
            loss_valid = self.l1(pred * (1.0 - mask_3c), target_clean * (1.0 - mask_3c))

            # 高頻文字邊緣保持
            pred_edge = self.sobel(pred)
            loss_edge = self.l1(pred_edge, target_edge)

            stage_loss = (self.hole_weight * loss_hole + 
                          self.valid_weight * loss_valid + 
                          self.edge_weight * loss_edge)
            total_loss += w * stage_loss

        return total_loss


# ---------------------------------------------------------------------------
# 2. 終極三軌真實+在地混合資料集 (Ultimate Tri-Track Dataset)
# ---------------------------------------------------------------------------
class UltimateEnd2EndDataset(Dataset):
    def __init__(
        self,
        hybrid_ds,
        hw5k_dir="/content/dataset/HW5K/train",
        examink_dir="/content/ExamInk-Seg/data/train",
        patch_size=256,
        total_steps=1000,
        real_ratio=0.70,
        local_ratio=0.25,
        flywheel_ratio=0.05
    ):
        self.hybrid_ds = hybrid_ds
        self.patch_size = patch_size
        self.total_steps = total_steps
        self.real_ratio = real_ratio
        self.local_ratio = local_ratio
        self.flywheel_ratio = flywheel_ratio

        # ------------------------------------------------------------------
        # 建立軌道 1：真實世界數據庫 (HW5K + ExamInk-Seg)
        # ------------------------------------------------------------------
        self.real_pairs = []

        # A. 載入 HW5K (5,000 組)
        if os.path.exists(hw5k_dir):
            hw_inp_dir = os.path.join(hw5k_dir, "input")
            hw_tar_dir = os.path.join(hw5k_dir, "target")
            if os.path.exists(hw_inp_dir) and os.path.exists(hw_tar_dir):
                for fname in sorted(os.listdir(hw_inp_dir)):
                    if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                        inp_p = os.path.join(hw_inp_dir, fname)
                        tar_p = os.path.join(hw_tar_dir, fname)
                        if os.path.exists(tar_p):
                            self.real_pairs.append({
                                "type": "hw5k",
                                "input": inp_p,
                                "target": tar_p,
                                "mask": None # 差值動態計算
                            })

        # B. 載入 ExamInk-Seg (1,511 組)
        if os.path.exists(examink_dir):
            e_src_dir = os.path.join(examink_dir, "source")
            e_tgt_dir = os.path.join(examink_dir, "target")
            e_msk_dir = os.path.join(examink_dir, "mask")
            if os.path.exists(e_src_dir) and os.path.exists(e_tgt_dir):
                for fname in sorted(os.listdir(e_src_dir)):
                    stem, ext = os.path.splitext(fname)
                    src_p = os.path.join(e_src_dir, fname)
                    tgt_p = os.path.join(e_tgt_dir, fname)
                    if not os.path.exists(tgt_p):
                        alt_ext = ".png" if ext == ".jpg" else ".jpg"
                        tgt_p = os.path.join(e_tgt_dir, stem + alt_ext)
                    msk_p = os.path.join(e_msk_dir, stem + ".png")
                    if os.path.exists(tgt_p):
                        self.real_pairs.append({
                            "type": "examink",
                            "input": src_p,
                            "target": tgt_p,
                            "mask": msk_p if os.path.exists(msk_p) else None
                        })

        print(f"[UltimateDataset] ★ 軌道 1 (真實世界試卷庫) 成功載入: {len(self.real_pairs)} 組！")

        # ------------------------------------------------------------------
        # 建立軌道 3：飛輪偽標籤大字難例
        # ------------------------------------------------------------------
        self.pseudo_items = self.hybrid_ds.pseudo_items if self.hybrid_ds else []
        print(f"[UltimateDataset] ★ 軌道 3 (飛輪大字難例) 成功載入: {len(self.pseudo_items)} 組！")
        print(f"[UltimateDataset] ★ 軌道 2 (台灣在地考卷底圖) 成功載入: {len(self.hybrid_ds.catalog) if self.hybrid_ds else 0} 頁！")
        print(f"[UltimateDataset] 目標取樣權重: 真實試卷={self.real_ratio*100:.0f}%, 在地考卷={self.local_ratio*100:.0f}%, 飛輪難例={self.flywheel_ratio*100:.0f}%")

    def __len__(self):
        return self.total_steps

    def sample_real_pair(self):
        pair = random.choice(self.real_pairs)
        img_in = cv2.imread(pair["input"], cv2.IMREAD_COLOR)
        img_tar = cv2.imread(pair["target"], cv2.IMREAD_COLOR)
        if img_in is None or img_tar is None:
            return None

        H, W = img_in.shape[:2]
        if img_tar.shape[:2] != (H, W):
            img_tar = cv2.resize(img_tar, (W, H))

        # 獲取或動態生成手寫 Mask (百度冠軍 compute_mask 差值法)
        if pair["mask"] and os.path.exists(pair["mask"]):
            mask = cv2.imread(pair["mask"], cv2.IMREAD_GRAYSCALE)
            if mask is not None and mask.shape[:2] != (H, W):
                mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        else:
            diff = np.abs(img_in.astype(np.float32) - img_tar.astype(np.float32))
            mean_diff = np.mean(diff, axis=-1)
            mask = (mean_diff > 18).astype(np.uint8) * 255

        # 隨機裁切 256x256 Patch (優先鎖定手寫筆跡區)
        ps = self.patch_size
        if H >= ps and W >= ps:
            hw_ys, hw_xs = np.where(mask > 50)
            if len(hw_ys) > 30 and random.random() < 0.85:
                idx = random.randint(0, len(hw_ys) - 1)
                cy, cx = hw_ys[idx], hw_xs[idx]
                y1 = int(np.clip(cy - ps // 2, 0, H - ps))
                x1 = int(np.clip(cx - ps // 2, 0, W - ps))
            else:
                y1 = random.randint(0, H - ps)
                x1 = random.randint(0, W - ps)

            crop_in = img_in[y1:y1+ps, x1:x1+ps]
            crop_tar = img_tar[y1:y1+ps, x1:x1+ps]
            crop_msk = mask[y1:y1+ps, x1:x1+ps]
        else:
            crop_in = cv2.resize(img_in, (ps, ps))
            crop_tar = cv2.resize(img_tar, (ps, ps))
            crop_msk = cv2.resize(mask, (ps, ps))

        # 輕微資料增強 (微小旋轉與水平翻轉，保持文字幾何先驗)
        if random.random() < 0.30:
            crop_in = cv2.flip(crop_in, 1)
            crop_tar = cv2.flip(crop_tar, 1)
            crop_msk = cv2.flip(crop_msk, 1)

        t_in = torch.from_numpy(cv2.cvtColor(crop_in, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0).permute(2, 0, 1)
        t_tar = torch.from_numpy(cv2.cvtColor(crop_tar, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0).permute(2, 0, 1)
        t_msk = torch.from_numpy((crop_msk > 50).astype(np.float32)).unsqueeze(0)

        return t_in, t_tar, t_msk

    def sample_local_pair(self):
        item = random.choice(self.hybrid_ds.catalog)
        raw_bgr = self.hybrid_ds.get_scanned_page(item["pdf"], item["page"])
        H, W = raw_bgr.shape[:2]

        bg_bgr = self.hybrid_ds.apply_background_dynamic_jitter(raw_bgr)
        synth_img = bg_bgr.copy()
        synth_mask = np.zeros((H, W), dtype=np.uint8)

        # 蓋印草稿與印章
        if self.hybrid_ds.layer_paths:
            for _ in range(random.randint(1, 2)):
                l_path = random.choice(self.hybrid_ds.layer_paths)
                layer = cv2.imread(l_path, cv2.IMREAD_UNCHANGED)
                if layer is not None:
                    lh, lw = layer.shape[:2]
                    cw = random.randint(min(260, lw), min(600, lw))
                    ch = random.randint(min(200, lh), min(500, lh))
                    cx = random.randint(0, lw - cw)
                    cy = random.randint(0, lh - ch)
                    crop_layer = layer[cy:cy+ch, cx:cx+cw]
                    rx = random.randint(0, max(1, W - cw))
                    ry = random.randint(0, max(1, H - ch))
                    self.hybrid_ds.overlay_stamp(synth_img, synth_mask, crop_layer, rx, ry, scale=random.uniform(0.7, 1.2))

        if self.hybrid_ds.letter_marks or self.hybrid_ds.symbol_marks:
            all_m = self.hybrid_ds.letter_marks + self.hybrid_ds.symbol_marks
            for _ in range(random.randint(3, 6)):
                m = random.choice(all_m)
                stamp = cv2.imread(os.path.join(self.hybrid_ds.marks_dir, m["file"]), cv2.IMREAD_UNCHANGED)
                if stamp is not None:
                    sh, sw = stamp.shape[:2]
                    rx = random.randint(20, max(21, W - sw - 20))
                    ry = random.randint(20, max(21, H - sh - 20))
                    self.hybrid_ds.overlay_stamp(synth_img, synth_mask, stamp, rx, ry, scale=random.uniform(0.8, 1.3), angle=random.uniform(-15, 15))

        ps = self.patch_size
        if H >= ps and W >= ps:
            hw_ys, hw_xs = np.where(synth_mask > 100)
            if len(hw_ys) > 30 and random.random() < 0.85:
                idx = random.randint(0, len(hw_ys) - 1)
                cy, cx = hw_ys[idx], hw_xs[idx]
                y1 = int(np.clip(cy - ps // 2, 0, H - ps))
                x1 = int(np.clip(cx - ps // 2, 0, W - ps))
            else:
                y1 = random.randint(0, H - ps)
                x1 = random.randint(0, W - ps)

            crop_in = synth_img[y1:y1+ps, x1:x1+ps]
            crop_tar = bg_bgr[y1:y1+ps, x1:x1+ps]
            crop_msk = synth_mask[y1:y1+ps, x1:x1+ps]
        else:
            crop_in = cv2.resize(synth_img, (ps, ps))
            crop_tar = cv2.resize(bg_bgr, (ps, ps))
            crop_msk = cv2.resize(synth_mask, (ps, ps))

        t_in = torch.from_numpy(cv2.cvtColor(crop_in, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0).permute(2, 0, 1)
        t_tar = torch.from_numpy(cv2.cvtColor(crop_tar, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0).permute(2, 0, 1)
        t_msk = torch.from_numpy((crop_msk > 100).astype(np.float32)).unsqueeze(0)

        return t_in, t_tar, t_msk

    def sample_flywheel_pair(self):
        item = random.choice(self.pseudo_items)
        img_in = cv2.imread(item["image"], cv2.IMREAD_COLOR)
        img_tar = cv2.imread(item["clean"], cv2.IMREAD_COLOR)
        mask = cv2.imread(item["hw_mask"], cv2.IMREAD_GRAYSCALE)
        if img_in is None or img_tar is None:
            return None

        ps = self.patch_size
        H, W = img_in.shape[:2]
        if H >= ps and W >= ps:
            y1 = random.randint(0, H - ps)
            x1 = random.randint(0, W - ps)
            crop_in = img_in[y1:y1+ps, x1:x1+ps]
            crop_tar = img_tar[y1:y1+ps, x1:x1+ps]
            crop_msk = mask[y1:y1+ps, x1:x1+ps] if mask is not None else np.zeros((ps, ps), dtype=np.uint8)
        else:
            crop_in = cv2.resize(img_in, (ps, ps))
            crop_tar = cv2.resize(img_tar, (ps, ps))
            crop_msk = cv2.resize(mask, (ps, ps)) if mask is not None else np.zeros((ps, ps), dtype=np.uint8)

        t_in = torch.from_numpy(cv2.cvtColor(crop_in, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0).permute(2, 0, 1)
        t_tar = torch.from_numpy(cv2.cvtColor(crop_tar, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0).permute(2, 0, 1)
        t_msk = torch.from_numpy((crop_msk > 100).astype(np.float32)).unsqueeze(0)

        return t_in, t_tar, t_msk

    def __getitem__(self, idx):
        # 依據黃金比例分流：70% 真實試卷流, 25% 在地考卷蓋印流, 5% 難例流
        r = random.random()
        try:
            if len(self.real_pairs) > 0 and r < self.real_ratio:
                res = self.sample_real_pair()
                if res is not None: return res

            if len(self.pseudo_items) > 0 and r < (self.real_ratio + self.flywheel_ratio):
                res = self.sample_flywheel_pair()
                if res is not None: return res

            return self.sample_local_pair()
        except Exception:
            ps = self.patch_size
            dummy = torch.ones(3, ps, ps, dtype=torch.float32)
            dmask = torch.zeros(1, ps, ps, dtype=torch.float32)
            return dummy, dummy, dmask


# ---------------------------------------------------------------------------
# 3. 主訓練流程
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Ultimate DocUnfold + 百度冠軍端到端試卷訓練器")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--steps_per_epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--hw5k_dir", type=str, default="/content/dataset/HW5K/train")
    parser.add_argument("--examink_dir", type=str, default="/content/ExamInk-Seg/data/train")
    parser.add_argument("--save_path", type=str, default="weights/best_docunfold_championship.pth")
    parser.add_argument("--resume", type=str, default="")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"============================================================")
    print(f"🏆 啟動 Ultimate DocUnfold + 百度冠軍端到端試卷訓練器")
    print(f"運算裝置: {device} ({torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'})")
    print(f"配置: Epochs={args.epochs}, Steps/Epoch={args.steps_per_epoch}, Batch={args.batch_size}, GradAccum={args.grad_accum}, PatchSize={args.patch_size}")
    print(f"============================================================")

    # 1. 初始化在地基礎底稿
    base_hybrid = HybridExamDataset(
        catalog_path="assets/scanned_catalog.json",
        scanned_root="scanned_pairs",
        assets_root="assets",
        anchors_path="assets/exam_anchors_catalog.json",
        pseudo_root="output/pseudo_hard_dataset",
        total_steps=args.steps_per_epoch * args.batch_size * args.grad_accum,
        dpi=150,
        is_train=True
    )

    # 2. 構建三軌真實融合 Dataset
    train_ds = UltimateEnd2EndDataset(
        hybrid_ds=base_hybrid,
        hw5k_dir=args.hw5k_dir,
        examink_dir=args.examink_dir,
        patch_size=args.patch_size,
        total_steps=args.steps_per_epoch * args.batch_size * args.grad_accum,
        real_ratio=0.70,
        local_ratio=0.25,
        flywheel_ratio=0.05
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True if device=="cuda" else False)

    # 3. 實例化 DocUnfold 模型
    model = DocUnfold().to(device)
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[Model] DocUnfold 7 階物理展開網絡載入成功，參數量: {params_m:.2f} M")

    if args.resume and os.path.exists(args.resume):
        print(f"[Model] 正在熱啟動權重: {args.resume}")
        model.load_state_dict(torch.load(args.resume, map_location=device), strict=False)
        print("[Model] 權重熱啟動完成！")

    criterion = ChampionshipDocLoss(hole_weight=3.0, valid_weight=1.0, edge_weight=0.5).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    best_loss = float("inf")

    # 4. 訓練迴圈
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        step_count = 0
        optimizer.zero_grad()

        for dirty, clean, mask in train_loader:
            dirty = dirty.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            stages_out = model(dirty)
            loss = criterion(stages_out, clean, mask)
            
            scaled_loss = loss / args.grad_accum
            scaled_loss.backward()

            step_count += 1
            running_loss += loss.item()

            if step_count % args.grad_accum == 0 or step_count == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            effective_step = step_count // args.grad_accum
            if (step_count % (25 * args.grad_accum) == 0):
                print(f"[Epoch {epoch:02d}/{args.epochs:02d}] Step {effective_step:03d}/{args.steps_per_epoch:03d} - Loss: {loss.item():.4f}")

        scheduler.step()
        epoch_loss = running_loss / max(1, step_count)
        epoch_time = time.time() - t0
        print(f"--> [Epoch {epoch:02d} 完成] 平均 Loss: {epoch_loss:.5f} | 耗時: {epoch_time:.1f}s | LR: {scheduler.get_last_lr()[0]:.2e}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), args.save_path)
            print(f"★ 最優權重已更新並保存至: {args.save_path} (Loss: {best_loss:.5f})")

        latest_path = args.save_path.replace(".pth", "_latest.pth")
        torch.save(model.state_dict(), latest_path)

    print("\n============================================================")
    print(f"🎉 全部訓練完畢！最優權重路徑: {args.save_path}")
    print("============================================================")

if __name__ == "__main__":
    main()
