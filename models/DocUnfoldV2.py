#!/usr/bin/env python3
"""
DocUnfoldV2: 5-Channel Mask & Layout-Conditioned 7-Stage Deep Unfolding Network
===============================================================================
Resolves the "Blind Unfolder" bottleneck identified in architectural review:
- Input tensor x: [B, 5, H, W] (or img: [B, 3, H, W], cond: [B, 2, H, W])
  - Channels 0..2: RGB image in [0, 1]
  - Channel 3: Handwriting probability map `probs_hw` in [0, 1] (from ResNet34-UNet)
  - Channel 4: Diagram / Layout structural prior mask `layout_mask` in [0, 1]
    (1.0 inside figure/table/coordinate-axis regions, 0.0 in general text)
- At all 7 unfolding stages (Stage 1 .. Stage 7), the intermediate 3-channel
  gradient-descent image estimate `x_k_img` is concatenated with `cond` ([B, 2, H, W])
  before entering `shallow_featK`, so every proximal mapping stage has explicit
  pixel-level visibility of handwriting location and diagram context.
- Supports ControlNet-style Zero-Convolution warm-starting from 3-channel
  `best_docunfold_targeted.pth` / `best_docunfold_calibrated.pth`.
"""

import torch
import torch.nn as nn
try:
    from models.DocUnfold import DocUnfold, conv
except ImportError:
    from DocUnfold import DocUnfold, conv


class DocUnfoldV2(DocUnfold):
    def __init__(
        self,
        n_feat=40,
        scale_unetfeats=20,
        kernel_size=3,
        reduction=4,
        bias=False,
        act=None,
        cond_c=2,
    ):
        super().__init__(
            n_feat=n_feat,
            scale_unetfeats=scale_unetfeats,
            kernel_size=kernel_size,
            reduction=reduction,
            bias=bias,
            act=act,
        )
        self.cond_c = cond_c
        in_c = 3 + cond_c
        if cond_c > 0:
            self.shallow_feat1[0] = conv(in_c, n_feat, kernel_size, bias=bias)
            self.shallow_feat2[0] = conv(in_c, n_feat, kernel_size, bias=bias)
            self.shallow_feat3[0] = conv(in_c, n_feat, kernel_size, bias=bias)
            self.shallow_feat4[0] = conv(in_c, n_feat, kernel_size, bias=bias)
            self.shallow_feat5[0] = conv(in_c, n_feat, kernel_size, bias=bias)
            self.shallow_feat6[0] = conv(in_c, n_feat, kernel_size, bias=bias)
            self.shallow_feat7[0] = conv(in_c, n_feat, kernel_size, bias=bias)

    def load_pretrained_3ch_with_zero_cond(self, state_dict_3ch):
        """
        ControlNet-style Zero-Conv Warm-Start:
        Copies [40, 3, 3, 3] RGB weights from pretrained 3-channel DocUnfold into
        channels 0..2 of shallow_feat{1..7}.0.weight, and initializes channels 3..4
        to 0.0 so Step-0 behavior is 100% identical to the pretrained specialist.
        """
        own_state = self.state_dict()
        loaded_keys = 0
        zero_padded_keys = 0
        for name, param in state_dict_3ch.items():
            if name not in own_state:
                continue
            target_param = own_state[name]
            if "shallow_feat" in name and name.endswith(".0.weight"):
                if param.shape[0] != target_param.shape[0]:
                    raise ValueError(f"n_feat mismatch: ckpt={param.shape[0]} vs model={target_param.shape[0]}")
                if param.shape[1] == 3 and target_param.shape[1] == 3 + self.cond_c:
                    with torch.no_grad():
                        own_state[name].zero_()
                        own_state[name][:, :3, :, :].copy_(param)
                    zero_padded_keys += 1
                    continue

            if param.shape == target_param.shape:
                own_state[name].copy_(param)
                loaded_keys += 1
        self.load_state_dict(own_state)
        return loaded_keys, zero_padded_keys

    def _cat_cond(self, x_img, cond):
        if self.cond_c <= 0 or cond is None:
            return x_img
        return torch.cat([x_img, cond], dim=1)

    def forward(self, x, cond=None):
        if x.shape[1] > 3 and cond is None:
            img = x[:, :3, :, :]
            cond = x[:, 3 : 3 + self.cond_c, :, :]
        else:
            img = x
            if self.cond_c > 0 and cond is None:
                raise ValueError(
                    f"[DocUnfoldV2] 錯誤: 本模型需要 5 通道輸入或提供 cond 張量 (cond_c={self.cond_c})，"
                    f"但收到 3 通道輸入且 cond=None。請傳入 [B, 5, H, W] 或指定 cond。"
                )

        # Stage 1
        phixsy_1 = self.phi_0(img) - img
        x1_img = img - self.r0 * self.phit_0(phixsy_1)
        x1 = self.shallow_feat1(self._cat_cond(x1_img, cond))
        feat1, feat_fin1 = self.stage1_encoder(x1)
        res1 = self.stage1_decoder(feat_fin1, feat1)
        x2_samfeats, stage1_img = self.sam12(res1[-1], x1_img)

        # Stage 2
        phixsy_2 = self.phi_1(stage1_img) - img
        x2_img = stage1_img - self.r1 * self.phit_1(phixsy_2)
        x2 = self.shallow_feat2(self._cat_cond(x2_img, cond))
        x2_cat = self.merge12(x2, x2_samfeats)
        feat2, feat_fin2 = self.stage2_encoder(x2_cat, feat1, res1)
        res2 = self.stage2_decoder(feat_fin2, feat2)
        x3_samfeats, stage2_img = self.sam23(res2[-1], x2_img)

        # Stage 3
        phixsy_3 = self.phi_2(stage2_img) - img
        x3_img = stage2_img - self.r2 * self.phit_2(phixsy_3)
        x3 = self.shallow_feat3(self._cat_cond(x3_img, cond))
        x3_cat = self.merge23(x3, x3_samfeats)
        feat3, feat_fin3 = self.stage3_encoder(x3_cat, feat2, res2)
        res3 = self.stage3_decoder(feat_fin3, feat3)
        x4_samfeats, stage3_img = self.sam34(res3[-1], x3_img)

        # Stage 4
        phixsy_4 = self.phi_3(stage3_img) - img
        x4_img = stage3_img - self.r3 * self.phit_3(phixsy_4)
        x4 = self.shallow_feat4(self._cat_cond(x4_img, cond))
        x4_cat = self.merge34(x4, x4_samfeats)
        feat4, feat_fin4 = self.stage4_encoder(x4_cat, feat3, res3)
        res4 = self.stage4_decoder(feat_fin4, feat4)
        x5_samfeats, stage4_img = self.sam45(res4[-1], x4_img)

        # Stage 5
        phixsy_5 = self.phi_4(stage4_img) - img
        x5_img = stage4_img - self.r4 * self.phit_4(phixsy_5)
        x5 = self.shallow_feat5(self._cat_cond(x5_img, cond))
        x5_cat = self.merge45(x5, x5_samfeats)
        feat5, feat_fin5 = self.stage5_encoder(x5_cat, feat4, res4)
        res5 = self.stage5_decoder(feat_fin5, feat5)
        x6_samfeats, stage5_img = self.sam56(res5[-1], x5_img)

        # Stage 6
        phixsy_6 = self.phi_5(stage5_img) - img
        x6_img = stage5_img - self.r5 * self.phit_5(phixsy_6)
        x6 = self.shallow_feat6(self._cat_cond(x6_img, cond))
        x6_cat = self.merge56(x6, x6_samfeats)
        feat6, feat_fin6 = self.stage6_encoder(x6_cat, feat5, res5)
        res6 = self.stage6_decoder(feat_fin6, feat6)
        x7_samfeats, stage6_img = self.sam67(res6[-1], x6_img)

        # Stage 7
        phixsy_7 = self.phi_6(stage6_img) - img
        x7_img = stage6_img - self.r6 * self.phit_6(phixsy_7)
        x7 = self.shallow_feat7(self._cat_cond(x7_img, cond))
        x7_cat = self.merge67(x7, x7_samfeats)
        stage7_img = self.tail(x7_cat) + img

        return [stage7_img, stage6_img, stage5_img, stage4_img, stage3_img, stage2_img, stage1_img]
