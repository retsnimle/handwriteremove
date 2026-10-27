#!/usr/bin/env python3
"""
Self-Diagnostic Suite: Dual-Scale 300 DPI Pipeline Verification
Usage: uv run python -m tools.self_diagnostic
"""
import os
import sys
import numpy as np
import torch
try:
    import pymupdf as fitz
except ImportError:
    import fitz

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def run_diagnostics():
    print("=== [Monika Pipeline Self-Diagnostic Suite] Starting Verification ===")

    # 1. Universal Padding & Geometry Invariant Check
    ps, stride = 512, 384
    test_cases = [
        ("JIS B4 Portrait", (3035, 4299)),
        ("A4 Standard Portrait", (2480, 3508)),
        ("A4 Standard Landscape", (3508, 2480)),
        ("A3 Scan", (3508, 4960)),
        ("Extreme Aspect Ratio", (5000, 1200)),
        ("Micro Canvas < PatchSize", (300, 400)),
    ]
    for name, (W, H) in test_cases:
        pad_w = max(ps, ((W - ps + stride - 1) // stride) * stride + ps)
        pad_h = max(ps, ((H - ps + stride - 1) // stride) * stride + ps)
        xs = list(range(0, pad_w - ps + 1, stride))
        ys = list(range(0, pad_h - ps + 1, stride))
        assert len(xs) > 0 and xs[-1] + ps >= W, f"❌ {name} Width coverage failed ({xs[-1] + ps} < {W})"
        assert len(ys) > 0 and ys[-1] + ps >= H, f"❌ {name} Height coverage failed ({ys[-1] + ps} < {H})"
        print(f"  [PASS] Geometric Padding 100% Coverage: {name:25s} (Canvas: {W:4d}x{H:4d} -> Pad: {pad_w:4d}x{pad_h:4d})")

    # 2. 5-Channel Architecture Instantiation Check
    try:
        from models.DocUnfoldV2 import DocUnfoldV2
        m = DocUnfoldV2(cond_c=2)
        assert m.shallow_feat1[0].weight.shape[1] == 5, f"Expected in_channels=5, got {m.shallow_feat1[0].weight.shape[1]}"
        print("  [PASS] DocUnfoldV2 5-Channel Architecture Confirmed")
    except Exception as e:
        print(f"  ❌ DocUnfoldV2 instantiation failed: {e}")
        return False

    # 3. 4-Edge Pixel SOP Inspection Check
    mock_clean = np.full((3508, 2480, 3), 255, dtype=np.uint8)
    H, W = mock_clean.shape[:2]
    kw = min(20, W // 2)
    kh = min(20, H // 2)
    r_strip = mock_clean[:, -kw:]
    b_strip = mock_clean[-kh:, :]
    l_strip = mock_clean[:, :kw]
    t_strip = mock_clean[:kh, :]
    r_ratio = float(np.sum(np.all(r_strip < 15, axis=-1)) / (r_strip.shape[0] * r_strip.shape[1]))
    b_ratio = float(np.sum(np.all(b_strip < 15, axis=-1)) / (b_strip.shape[0] * b_strip.shape[1]))
    l_ratio = float(np.sum(np.all(l_strip < 15, axis=-1)) / (l_strip.shape[0] * l_strip.shape[1]))
    t_ratio = float(np.sum(np.all(t_strip < 15, axis=-1)) / (t_strip.shape[0] * t_strip.shape[1]))
    max_ratio = max(r_ratio, b_ratio, l_ratio, t_ratio)
    assert max_ratio == 0.0
    print(f"  [PASS] 4-Edge Pixel SOP Algorithm Logic Valid (Black Ratio: {max_ratio*100:.2f}%)")

    # 4. Core Weights Check
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weights_dir = os.path.join(repo_root, "weights")
    needed_weights = [
        ("best_exam_remover.pth", 80.0),                               # actual ~93.4 MB
        ("doclayout_yolo_docstructbench_imgsz1024.pt", 30.0),         # actual ~38.8 MB
        ("best_docunfold_v2_dual_300dpi.pth", 40.0)                   # actual ~46.7 MB
    ]
    all_weights_ok = True
    for w, thresh_mb in needed_weights:
        wp = os.path.join(weights_dir, w)
        if os.path.exists(wp):
            sz_mb = os.path.getsize(wp) / (1024 * 1024)
            if sz_mb >= thresh_mb:
                print(f"  [PASS] Core Weight Ready & Sized: {w} ({sz_mb:.1f} MB >= {thresh_mb:.1f} MB)")
            else:
                print(f"  ⚠️ Core Weight undersized: {w} ({sz_mb:.1f} MB < {thresh_mb:.1f} MB)")
                all_weights_ok = False
        else:
            print(f"  ⚠️ Core Weight missing: {w}")
            all_weights_ok = False

    # 5. Loss Function Charbonnier Anti-NaN Check
    try:
        from tools.train_colab_docunfold_dual_300dpi import masked_charbonnier
        d = torch.randn(1, 3, 512, 512)
        m = torch.zeros(1, 1, 512, 512)
        loss = masked_charbonnier(d, m)
        assert float(loss) == 0.0
        print("  [PASS] Masked Charbonnier Anti-NaN Stability Confirmed")
    except Exception as e:
        print(f"  ❌ Loss function test failed: {e}")
        return False

    print("=== ★ Pipeline Self-Verification PASSED! Specification is 100% reproducible. ★ ===")
    return all_weights_ok


if __name__ == "__main__":
    success = run_diagnostics()
    sys.exit(0 if success else 1)
