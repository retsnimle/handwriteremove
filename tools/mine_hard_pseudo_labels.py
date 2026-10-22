import os
import glob
import cv2
import numpy as np
from pathlib import Path
import fitz

from dual_scale_remover import DualScaleRemover

def mine_hard_examples_from_scans(
    raw_scans_dir: str,
    output_dataset_dir: str,
    weights_path: str = "weights/best_exam_remover_hybrid.pth",
    patch_size: int = 768,
    min_discrepancy_area: int = 250,
):
    """
    掃描真實考卷，挖掘 (P_fused - P_1.0) 差異顯著的大尺度手寫難例，生成偽標籤訓練集
    """
    out_path = Path(output_dataset_dir)
    for sub in ["images", "clean_gt", "hw_masks", "text_masks"]:
        (out_path / sub).mkdir(parents=True, exist_ok=True)

    remover = DualScaleRemover(weights_path=weights_path, device="cuda")
    
    # 支援圖片與 PDF 考卷
    img_list = []
    
    # 1. 查找圖片
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        img_list.extend(glob.glob(os.path.join(raw_scans_dir, ext)))
        
    # 2. 查找 PDF 並提取高解析度圖像
    pdf_list = glob.glob(os.path.join(raw_scans_dir, "*.pdf"))
    temp_dir = Path("output/temp_scans")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    for pdf_path in pdf_list:
        p_name = Path(pdf_path).stem
        doc = fitz.open(pdf_path)
        for page_idx in range(min(5, doc.page_count)):  # 採樣前幾頁
            pix = doc[page_idx].get_pixmap(dpi=150)
            page_img_path = str(temp_dir / f"{p_name}_p{page_idx:02d}.png")
            pix.save(page_img_path)
            img_list.append(page_img_path)
        doc.close()

    print(f"[Flywheel] 待挖掘考卷頁數: {len(img_list)}")
    saved_count = 0
    
    for img_path in img_list:
        stem = Path(img_path).stem
        bgr = cv2.imread(img_path)
        if bgr is None:
            continue
        H, W, _ = bgr.shape
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # 執行升級版雙尺度推論
        res = remover.predict_dual_scale(bgr)
        cleaned_bgr = res["cleaned_bgr"]
        p_10 = res["p_10"]
        p_fused = res["p_fused"]

        # 1. 計算雙尺度喚醒差異圖 (找出 1.0x 徹底失效但 0.5x 成功喚醒的大跨度筆劃)
        diff_map = (p_fused - p_10) > 0.60
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(diff_map.astype(np.uint8), connectivity=8)

        # 2. 重建高精細手寫 GT 遮罩與純印刷保護遮罩
        printed_skeleton = ((gray < 95) & (p_10 < 0.06) & (p_fused < 0.22)).astype(np.uint8) * 255
        erased_diff = np.max(np.abs(cleaned_bgr.astype(np.int16) - bgr.astype(np.int16)), axis=-1)
        hw_mask_gt = ((erased_diff > 18) | (p_fused > 0.35)).astype(np.uint8) * 255
        hw_mask_gt[printed_skeleton > 0] = 0

        # 3. 針對每個挖掘出的難例連通域，以質心為中心裁切 768x768 訓練樣本
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            w_cc, h_cc = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]

            if area < min_discrepancy_area:
                continue
            if max(w_cc, h_cc) / (min(w_cc, h_cc) + 1e-5) > 18.0:
                continue

            cx, cy = int(centroids[i][0]), int(centroids[i][1])
            jx = np.random.randint(-64, 65)
            jy = np.random.randint(-64, 65)
            x1 = np.clip(cx + jx - patch_size // 2, 0, max(0, W - patch_size))
            y1 = np.clip(cy + jy - patch_size // 2, 0, max(0, H - patch_size))
            x2, y2 = x1 + patch_size, y1 + patch_size

            patch_name = f"{stem}_hard_{i:03d}.png"
            cv2.imwrite(str(out_path / "images" / patch_name), bgr[y1:y2, x1:x2])
            cv2.imwrite(str(out_path / "clean_gt" / patch_name), cleaned_bgr[y1:y2, x1:x2])
            cv2.imwrite(str(out_path / "hw_masks" / patch_name), hw_mask_gt[y1:y2, x1:x2])
            cv2.imwrite(str(out_path / "text_masks" / patch_name), printed_skeleton[y1:y2, x1:x2])
            saved_count += 1

    print(f"[Flywheel] ★ 飛輪難例挖掘大功告成！共產出 {saved_count} 組真實大字/包覆難例四元組！")

if __name__ == "__main__":
    # 對現有的考卷樣本作挖掘
    target_scan_dir = "handwritten_samples"
    if not os.path.exists(target_scan_dir) or len(os.listdir(target_scan_dir)) == 0:
        target_scan_dir = "."
    mine_hard_examples_from_scans(
        raw_scans_dir=target_scan_dir,
        output_dataset_dir="output/pseudo_hard_dataset",
        weights_path="weights/best_exam_remover_hybrid.pth"
    )
