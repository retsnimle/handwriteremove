import torch
import torch.nn.functional as F
import numpy as np
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"LaMa benchmark on device: {device}")
model = torch.jit.load("weights/big-lama.pt", map_location=device).eval()

# 測試整頁考卷尺寸 (1754 x 2481)
H, W = 1754, 2481
pad_h = ((H + 31) // 32) * 32 - H
pad_w = ((W + 31) // 32) * 32 - W

# 建立假影像與遮罩
dummy_img = torch.rand(1, 3, H, W, device=device)
dummy_mask = torch.zeros(1, 1, H, W, device=device)
# 假手寫筆跡遮罩
dummy_mask[:, :, 300:500, 100:400] = 1.0

img_pad = F.pad(dummy_img, (0, pad_w, 0, pad_h), mode="reflect")
mask_pad = F.pad(dummy_mask, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

print(f"Input padded shape: {img_pad.shape}")

# 預熱
with torch.no_grad():
    _ = model(img_pad[:, :, :512, :512], mask_pad[:, :, :512, :512])

# 實測全頁推論
torch.cuda.synchronize()
t0 = time.time()
try:
    with torch.no_grad():
        out = model(img_pad, mask_pad)
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    vram_used = torch.cuda.max_memory_allocated() / (1024 * 1024)
    print(f"★ 全頁 2481x1754 LaMa 整頁直出成功！耗時: {elapsed:.3f} 秒, 顯存峰值: {vram_used:.1f} MB")
except RuntimeError as e:
    print(f"全頁直出顯存受限: {e}")
