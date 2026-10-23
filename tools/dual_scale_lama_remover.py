import os
import sys
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import fitz
import segmentation_models_pytorch as smp


class DualScaleLaMaRemover:
    def __init__(
        self,
        hybrid_weights="weights/best_exam_remover_hybrid.pth",
        lama_weights="weights/big-lama.pt",
        device="cuda"
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"[DualScaleLaMaRemover] 裝置: {self.device}")

        # 1. 載入三軌微調之 ResNet34-UNet 雙尺度分割模型
        print(f"[Model 1] 載入三軌微調去手寫模型: {hybrid_weights}")
        self.seg_model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
        self.seg_model.load_state_dict(torch.load(hybrid_weights, map_location=self.device, weights_only=True))
        self.seg_model.to(self.device).eval()

        # 2. 載入 LaMa 結構感知重繪 JIT 模型
        print(f"[Model 2] 載入 LaMa 快速傅立葉重繪模型: {lama_weights}")
        self.lama_model = torch.jit.load(lama_weights, map_location=self.device).eval()

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
        with torch.no_grad():
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
        """
        以 LaMa 重繪單個 Patch 或局部區域
        rgb_patch: (H, W, 3) uint8 RGB
        mask_patch: (H, W) uint8 binary (0 or 255)
        """
        H, W, _ = rgb_patch.shape
        pad_h = ((H + 31) // 32) * 32 - H
        pad_w = ((W + 31) // 32) * 32 - W

        t_img = torch.from_numpy(rgb_patch).to(self.device).permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
        t_mask = torch.from_numpy(mask_patch).to(self.device).unsqueeze(0).unsqueeze(0).float().div_(255.0)

        t_img_pad = F.pad(t_img, (0, pad_w, 0, pad_h), mode="reflect")
        t_mask_pad = F.pad(t_mask, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

        with torch.no_grad():
            out_pad = self.lama_model(t_img_pad, t_mask_pad)

        out_cropped = out_pad[0, :, :H, :W].permute(1, 2, 0).clamp(0, 1).mul(255).byte().cpu().numpy()
        return out_cropped

    def clean_page(self, bgr_img: np.ndarray):
        t0 = time.time()
        H, W, _ = bgr_img.shape
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

        # ----------------------------------------------------
        # Stage 1: 雙尺度金字塔分割手寫筆跡
        # ----------------------------------------------------
        # 1. 1.0x 原尺度推論
        p_10 = self._infer_single_scale(bgr_img)

        # 2. 0.5x 全域金字塔推論 (抓取跨距超大字結構)
        bgr_half = cv2.resize(bgr_img, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
        p_half = self._infer_single_scale(bgr_half)
        p_05_raw = cv2.resize(p_half, (W, H), interpolation=cv2.INTER_LINEAR)

        # 3. 引導濾波對齊原圖邊緣
        I_darkness = 1.0 - (gray.astype(np.float32) / 255.0)
        p_05_guided = self._guided_filter_gray(I_darkness, p_05_raw, r=5, eps=5e-4)
        p_05_refined = np.where(gray < 242, p_05_guided, 0.0)

        # 4. 建立雙尺度印刷文字護盾 (保護題號骨架)
        printed_skeleton = (gray < 95) & (p_10 < 0.06) & (p_05_raw < 0.22)
        k_3x3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        printed_shield = (
            (cv2.dilate(printed_skeleton.astype(np.uint8), k_3x3) > 0)
            & (gray < 210)
            & (p_10 < 0.15)
            & (p_05_guided < 0.55)
        )

        # 5. 保護式機率融合
        p_fused = np.maximum(p_10, p_05_refined)
        p_fused[printed_shield] = p_10[printed_shield]

        # 6. 連通域動態膨脹取得手寫擦除遮罩
        seed_mask = (p_fused > 0.28) & (~printed_shield)
        k_5x5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        halo_tier1 = (cv2.dilate(seed_mask.astype(np.uint8), k_5x5) > 0) & (gray < 225) & (~printed_shield)
        core_and_tier1 = seed_mask | halo_tier1
        halo_tier2 = (cv2.dilate(core_and_tier1.astype(np.uint8), k_5x5) > 0) & (gray < 244) & (~printed_shield)
        full_erase_mask = (core_and_tier1 | halo_tier2).astype(np.uint8) * 255

        # ----------------------------------------------------
        # Stage 2: LaMa 結構感知重繪
        # ----------------------------------------------------
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        
        # 尋找所有手寫連通區域，採用帶 Padding 的局部 Bounding Box 重繪
        # 優勢：比全圖重繪快 10 倍以上，且保證未塗寫區域 100% 位元級原生保留！
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats((full_erase_mask > 0).astype(np.uint8))
        
        inpainted_rgb = rgb_img.copy()
        pad = 32

        t_lama_start = time.time()
        reconstructed_count = 0
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 10:  # 忽略微小噪點
                continue

            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            # 加上 32px 邊緣 Context Padding，讓 LaMa 能看見橫截的表格線與周圍紙張背景色
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(W, x + w + pad)
            y1 = min(H, y + h + pad)

            crop_rgb = rgb_img[y0:y1, x0:x1]
            crop_mask = full_erase_mask[y0:y1, x0:x1]

            # 如果裁切塊太小，墊到至少 128x128
            crop_h, crop_w = crop_rgb.shape[:2]
            if crop_h < 64 or crop_w < 64:
                # 擴展到至少 64
                cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
                x0 = max(0, cx - 64)
                y0 = max(0, cy - 64)
                x1 = min(W, cx + 64)
                y1 = min(H, cy + 64)
                crop_rgb = rgb_img[y0:y1, x0:x1]
                crop_mask = full_erase_mask[y0:y1, x0:x1]

            # 調用 LaMa 局部重繪
            crop_inpainted = self.inpaint_lama_patch(crop_rgb, crop_mask)

            # 僅在遮罩內部填入 LaMa 重繪結果（雙尺度護盾處維持原圖文字）
            inner_mask = (crop_mask > 0)[..., None]
            orig_crop = inpainted_rgb[y0:y1, x0:x1]
            inpainted_rgb[y0:y1, x0:x1] = np.where(inner_mask, crop_inpainted, orig_crop)
            reconstructed_count += 1

        t_lama_elapsed = time.time() - t_lama_start
        cleaned_bgr = cv2.cvtColor(inpainted_rgb, cv2.COLOR_RGB2BGR)

        # 傳統塗白基準（用於效果對照）
        white_filled_bgr = bgr_img.copy()
        white_filled_bgr[full_erase_mask > 0] = 255

        total_elapsed = time.time() - t0
        return {
            "cleaned_bgr": cleaned_bgr,
            "white_filled_bgr": white_filled_bgr,
            "erase_mask": full_erase_mask,
            "p_10": p_10,
            "p_05": p_05_raw,
            "p_fused": p_fused,
            "reconstructed_count": reconstructed_count,
            "lama_time_sec": t_lama_elapsed,
            "total_time_sec": total_elapsed
        }


def run_verification():
    remover = DualScaleLaMaRemover(
        hybrid_weights="weights/best_exam_remover_hybrid.pth",
        lama_weights="weights/big-lama.pt"
    )

    doc = fitz.open("20251020202938305.pdf")
    pix = doc[0].get_pixmap(dpi=150)
    bgr = cv2.imdecode(np.frombuffer(pix.tobytes("png"), np.uint8), cv2.IMREAD_COLOR)
    doc.close()

    print(f"考卷原圖尺寸: {bgr.shape[1]}x{bgr.shape[0]}")
    res = remover.clean_page(bgr)
    print(f"★ 雙階段去手寫與 LaMa 重繪完成！")
    print(f"  - 筆跡區塊修復數: {res['reconstructed_count']} 個")
    print(f"  - LaMa 重繪耗時: {res['lama_time_sec']:.3f} 秒")
    print(f"  - 雙尺度總耗時: {res['total_time_sec']:.3f} 秒")

    os.makedirs("output/qa", exist_ok=True)
    # 儲存全頁結果
    cv2.imwrite("output/qa/sat_p1_lama_cleaned_full.png", res["cleaned_bgr"])

    # 截取第 1 題大 D 關鍵區域 (y: 360~660, x: 20~300)
    crop_orig = bgr[360:660, 20:300]
    crop_white = res["white_filled_bgr"][360:660, 20:300]
    crop_lama = res["cleaned_bgr"][360:660, 20:300]
    crop_mask = cv2.cvtColor(res["erase_mask"][360:660, 20:300], cv2.COLOR_GRAY2BGR)

    # 四聯對照切片: [原圖 (大D 騎乘題號), 提取之擦除遮罩, 粗暴塗白死斑, LaMa 結構感知重繪]
    compare = np.hstack([crop_orig, crop_mask, crop_white, crop_lama])
    out_path = "output/qa/sat_q1_lama_inpaint_comparison.png"
    cv2.imwrite(out_path, compare)
    print(f"★ 已產出四聯對照圖: {out_path}")


if __name__ == "__main__":
    run_verification()
