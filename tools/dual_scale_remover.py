import os
import sys
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import fitz
import segmentation_models_pytorch as smp


class DualScaleRemover:
    def __init__(self, weights_path="weights/best_exam_remover_hybrid.pth", device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"[DualScaleRemover] 裝置: {self.device}")
        self.model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device, weights_only=True))
        self.model.to(self.device).eval()
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
            logits = self.model(t_pad)
            probs = torch.sigmoid(logits)[0, 0, :H, :W].cpu().numpy()
        return probs

    @staticmethod
    def _guided_filter_gray(I_norm: np.ndarray, p: np.ndarray, r: int = 6, eps: float = 1e-3) -> np.ndarray:
        """
        純 OpenCV 高速邊緣引導濾波 (O(N) 複雜度，不依賴 opencv-contrib)
        利用原圖高解析度灰階邊緣 I_norm，消除 p_05 雙線性插值產生的邊緣模糊與空間偏移
        """
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

    def predict_dual_scale(self, bgr_img: np.ndarray):
        t0 = time.time()
        H, W, _ = bgr_img.shape
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

        # 1. 原尺度推論 P_1.0 (2481x1754)
        p_10 = self._infer_single_scale(bgr_img)

        # 2. 半尺度推論 P_0.5 (1240x877) 並放大回原尺寸
        bgr_half = cv2.resize(bgr_img, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
        p_half = self._infer_single_scale(bgr_half)
        p_05_raw = cv2.resize(p_half, (W, H), interpolation=cv2.INTER_LINEAR)

        # 3. 原圖梯度引導對齊 (消除 0.5x 放大的 2~4px 插值暈開，讓機率緊貼真實筆劃邊界)
        I_darkness = 1.0 - (gray.astype(np.float32) / 255.0)
        p_05_guided = self._guided_filter_gray(I_darkness, p_05_raw, r=5, eps=5e-4)

        # 過濾純白背景上的微弱插值雜訊 (放寬至 242 以保留原子筆淡灰抗鋸齒尾韻)
        p_05_refined = np.where(gray < 242, p_05_guided, 0.0)

        # 4. 建立「雙尺度印刷字體護盾 (Printed Anti-Aliasing Shield)」
        # 唯有在 1.0x 與 0.5x 同時呈現低機率的深色像素，才認定為純印刷核心骨架
        printed_skeleton = (gray < 95) & (p_10 < 0.06) & (p_05_raw < 0.22)
        k_3x3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # 向外保護 1px 的印刷抗鋸齒外殼 (gray < 210)，但排除高確信手寫重疊處，徹底防止題號「1.」產生毛刺
        printed_shield = (
            (cv2.dilate(printed_skeleton.astype(np.uint8), k_3x3) > 0)
            & (gray < 210)
            & (p_10 < 0.15)
            & (p_05_guided < 0.55)
        )

        # 5. 保護式機率融合
        p_fused = np.maximum(p_10, p_05_refined)
        p_fused[printed_shield] = p_10[printed_shield]

        # 6. 測地線連通域邊緣追蹤 (Geodesic Halo Growth) — 徹底吃掉 210~230 空心細輪廓
        # Step 6a: 高確信手寫主幹
        seed_mask = (p_fused > 0.28) & (~printed_shield)
        # Step 6b: 第一層擴展 (抓取緊貼主幹的次深色邊緣, gray < 225)
        k_5x5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        halo_tier1 = (cv2.dilate(seed_mask.astype(np.uint8), k_5x5) > 0) & (gray < 225) & (~printed_shield)
        core_and_tier1 = seed_mask | halo_tier1
        # Step 6c: 第二層擴展 (抓取最外圍 210~244 的淡灰抗鋸齒虛影)
        halo_tier2 = (cv2.dilate(core_and_tier1.astype(np.uint8), k_5x5) > 0) & (gray < 244) & (~printed_shield)
        full_erase_mask = core_and_tier1 | halo_tier2

        # 7. 局部乾淨紙色基底估計與軟邊緣融合 (避免死白 255 造成視覺斷層)
        k_bg = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))
        paper_bg = cv2.morphologyEx(bgr_img, cv2.MORPH_CLOSE, k_bg)
        paper_bg = np.maximum(paper_bg, 248)  # 確保底色乾淨明亮

        # 計算軟邊緣 Alpha (主幹 100% 替換，最外緣 1~2px 平滑過渡)
        alpha = full_erase_mask.astype(np.float32)
        alpha_blur = cv2.GaussianBlur(alpha, (3, 3), 0)
        alpha_final = np.where(core_and_tier1, 1.0, alpha_blur)
        alpha_final[printed_shield] = 0.0
        alpha_3c = alpha_final[..., None]

        cleaned_bgr = (bgr_img.astype(np.float32) * (1.0 - alpha_3c) + paper_bg.astype(np.float32) * alpha_3c).astype(np.uint8)
        elapsed = time.time() - t0
        return {
            "p_10": p_10,
            "p_05": p_05_raw,
            "p_fused": p_fused,
            "cleaned_bgr": cleaned_bgr,
            "elapsed_sec": elapsed
        }

def test_sat_p1_upgrade():
    remover = DualScaleRemover("weights/best_exam_remover_hybrid.pth")
    doc = fitz.open("20251020202938305.pdf")
    pix = doc[0].get_pixmap(dpi=150)
    bgr = cv2.imdecode(np.frombuffer(pix.tobytes("png"), np.uint8), cv2.IMREAD_COLOR)
    doc.close()

    print(f"考卷原圖尺寸: {bgr.shape[1]}x{bgr.shape[0]}")
    res = remover.predict_dual_scale(bgr)
    print(f"升級版引導濾波雙尺度推論完成！耗時: {res['elapsed_sec']:.2f} 秒")

    os.makedirs("output/qa", exist_ok=True)
    # 儲存全頁結果
    cv2.imwrite("output/qa/sat_p1_guided_cleaned_full.png", res["cleaned_bgr"])

    # 截取第 1 題大 D 關鍵區域 (y: 360~660, x: 20~300)
    crop_orig = bgr[360:660, 20:300]
    crop_p10 = bgr[360:660, 20:300].copy()
    crop_p10[res["p_10"][360:660, 20:300] > 0.45] = 255

    crop_clean = res["cleaned_bgr"][360:660, 20:300]

    # 對比切片: [原圖 (大D), 舊版 1.0x 單尺度 (失敗), 升級版引導濾波雙尺度 (刀鋒級清空)]
    compare = np.hstack([crop_orig, crop_p10, crop_clean])
    cv2.imwrite("output/qa/sat_q1_guided_filter_comparison.png", compare)
    print("★ 已產出升級版對照切片: output/qa/sat_q1_guided_filter_comparison.png")

if __name__ == "__main__":
    test_sat_p1_upgrade()
