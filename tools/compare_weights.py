import cv2
import numpy as np
import fitz
from tools.dual_scale_remover import DualScaleRemover

remover_old = DualScaleRemover("weights/best_exam_remover.pth")
remover_hybrid = DualScaleRemover("weights/best_exam_remover_hybrid.pth")

doc = fitz.open("20251020202938305.pdf")
pix = doc[0].get_pixmap(dpi=150)
bgr = cv2.imdecode(np.frombuffer(pix.tobytes("png"), np.uint8), cv2.IMREAD_COLOR)
doc.close()

crop_orig = bgr[360:660, 20:300]

res_old = remover_old.predict_dual_scale(bgr)
res_hybrid = remover_hybrid.predict_dual_scale(bgr)

p_fused_old = (res_old["p_fused"][360:660, 20:300] * 255).astype(np.uint8)
p_fused_hybrid = (res_hybrid["p_fused"][360:660, 20:300] * 255).astype(np.uint8)

clean_old = res_old["cleaned_bgr"][360:660, 20:300]
clean_hybrid = res_hybrid["cleaned_bgr"][360:660, 20:300]

comp = np.hstack([
    crop_orig,
    cv2.cvtColor(p_fused_old, cv2.COLOR_GRAY2BGR),
    clean_old,
    cv2.cvtColor(p_fused_hybrid, cv2.COLOR_GRAY2BGR),
    clean_hybrid
])

cv2.imwrite("output/qa/compare_old_vs_hybrid.png", comp)
print("Saved compare_old_vs_hybrid.png")
