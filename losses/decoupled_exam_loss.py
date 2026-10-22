import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoupledExamLoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 1.0,
        text_fp_weight: float = 3.0,
        overlap_recall_boost: float = 4.0,
        buffer_kernel: int = 5,
        proximity_kernel: int = 15,
    ):
        """
        解決 3.0x Text_FP_Penalty 導致跨題大字輸出 0.0001 的解耦損失函數：
        1. buffer_kernel (5x5): 解除手寫筆跡周圍 5px 內的印刷字誤殺懲罰，消除交界恐懼梯度
        2. proximity_kernel (15x15): 對緊鄰/橫跨印刷字 15px 內的手寫筆劃施加 4.0x 正向召回激勵
        """
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.text_fp_weight = text_fp_weight
        self.overlap_recall_boost = overlap_recall_boost
        self.buffer_k = buffer_kernel
        self.prox_k = proximity_kernel

    @staticmethod
    def _gpu_dilate(mask: torch.Tensor, ksize: int) -> torch.Tensor:
        pad = ksize // 2
        return F.max_pool2d(mask.float(), kernel_size=ksize, stride=1, padding=pad)

    def forward(
        self,
        logits: torch.Tensor,          # (B, 1, H, W) 模型原始輸出
        hw_targets: torch.Tensor,      # (B, 1, H, W) 手寫 Ground Truth {0, 1}
        pure_text_masks: torch.Tensor  # (B, 1, H, W) 模板印刷核心像素 {0, 1} (gray < 85)
    ):
        probs = torch.sigmoid(logits)
        hw_gt = (hw_targets > 0.5).float()
        text_gt = (pure_text_masks > 0.5).float()

        with torch.no_grad():
            # 1. 形態學安全緩衝區：從受保護印刷遮罩中，扣除手寫筆劃及其 5x5 鄰域
            hw_dilated_buffer = self._gpu_dilate(hw_gt, self.buffer_k)
            strict_text_mask = text_gt * (1.0 - hw_dilated_buffer)

            # 2. 跨字手寫強迫召回區：僅約束在與印刷字真實接觸/交界 5px 內的手寫像素
            # 避免激勵權重洩漏到純白紙背景上的手寫筆劃
            text_dilated_contact = self._gpu_dilate(text_gt, 5)
            hard_hw_mask = hw_gt * text_dilated_contact

            # 3. 構建空間自適應像素權重 (對真實交界處手寫給予 1 + 4.0 = 5.0 倍的正樣本權重)
            pixel_weights = 1.0 + self.overlap_recall_boost * hard_hw_mask

        # --- A. 空間加權 BCE Loss ---
        bce_raw = F.binary_cross_entropy_with_logits(logits, hw_gt, reduction="none")
        loss_bce = (bce_raw * pixel_weights).mean()

        # --- B. Soft Dice Loss ---
        smooth = 1.0
        intersection = (probs * hw_gt).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + hw_gt.sum(dim=(2, 3))
        loss_dice = (1.0 - (2.0 * intersection + smooth) / (union + smooth)).mean()

        # --- C. 解耦後的 Text_FP_Penalty (數值穩定分母 clamp >= 1.0) ---
        denom = strict_text_mask.sum(dim=(2, 3)).clamp(min=1.0)
        fp_on_strict_text = (probs * strict_text_mask).sum(dim=(2, 3)) / denom
        loss_text_fp = fp_on_strict_text.mean()

        total_loss = (
            self.bce_weight * loss_bce
            + self.dice_weight * loss_dice
            + self.text_fp_weight * loss_text_fp
        )
        return total_loss, loss_bce.item(), loss_dice.item(), loss_text_fp.item()
