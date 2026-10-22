#!/usr/bin/env python3
"""
DocLayout-YOLO (純 GPU 原生) 考卷版面結構與題號檢測實測
"""
import time
import cv2
import numpy as np
import fitz
import torch
from doclayout_yolo import YOLOv10

print("=== 啟動 DocLayout-YOLO 純 GPU 實測 ===")
weights_path = "weights/doclayout_yolo_docstructbench_imgsz1024.pt"

# 1. 載入模型至 CUDA GPU
t0 = time.time()
model = YOLOv10(weights_path)
print(f"★ DocLayout-YOLO 載入成功 (耗時: {time.time() - t0:.2f}s)")

# 2. 準備橫式考卷 Page 1
doc = fitz.open("20251020202938305.pdf")
pix = doc[0].get_pixmap(dpi=150)
img_bgr = cv2.imdecode(np.frombuffer(pix.tobytes("png"), np.uint8), cv2.IMREAD_COLOR)
doc.close()

# 3. GPU 原生前向推論
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

t_infer0 = time.time()
det_res = model.predict(
    img_bgr,
    imgsz=1024,
    conf=0.25,
    device="cuda:0"
)
t_infer = time.time() - t_infer0

vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
print(f"★ 純 GPU 版面檢測完成！耗時: {t_infer*1000:.1f} 毫秒 ({t_infer:.3f} 秒)，GPU 顯存峰值: {vram_mb:.1f} MB")

# 4. 解析檢測到的類別與邊界框
boxes = det_res[0].boxes
print(f"★ 共偵測到 {len(boxes)} 個版面物件元素！")

names = det_res[0].names
print("類別標籤表:", names)

vis = img_bgr.copy()
for box in boxes:
    cls_id = int(box.cls[0].item())
    cls_name = names.get(cls_id, str(cls_id))
    conf = float(box.conf[0].item())
    xyxy = box.xyxy[0].cpu().numpy().astype(int)
    x1, y1, x2, y2 = xyxy

    print(f"  [{cls_name}] Conf: {conf:.2f} -> BBox: [{x1}, {y1}, {x2}, {y2}]")
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(vis, f"{cls_name} {conf:.2f}", (x1, max(18, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

out_vis = "output/qa/doclayout_yolo_vis_p01.png"
cv2.imwrite(out_vis, vis)
print(f"★ 視覺化成果已儲存至: {out_vis}")
