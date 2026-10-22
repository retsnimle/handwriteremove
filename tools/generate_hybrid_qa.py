#!/usr/bin/env python3
import os
import fitz
import cv2
import numpy as np

def make_qa(orig_pdf, clean_pdf, prefix):
    if not os.path.exists(orig_pdf) or not os.path.exists(clean_pdf):
        print(f"File not found: {orig_pdf} or {clean_pdf}")
        return
    d_orig = fitz.open(orig_pdf)
    d_clean = fitz.open(clean_pdf)
    out_dir = "output/qa"
    os.makedirs(out_dir, exist_ok=True)
    
    for i in range(len(d_orig)):
        pix1 = d_orig[i].get_pixmap(dpi=100)
        pix2 = d_clean[i].get_pixmap(dpi=100)
        im1 = cv2.imdecode(np.frombuffer(pix1.tobytes("png"), np.uint8), cv2.IMREAD_COLOR)
        im2 = cv2.imdecode(np.frombuffer(pix2.tobytes("png"), np.uint8), cv2.IMREAD_COLOR)
        
        h = min(im1.shape[0], im2.shape[0])
        im1 = cv2.resize(im1, (int(im1.shape[1] * h / im1.shape[0]), h))
        im2 = cv2.resize(im2, (int(im2.shape[1] * h / im2.shape[0]), h))
        
        cv2.putText(im1, f"Original Page {i+1}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.putText(im2, f"Hybrid Cleaned Page {i+1}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 180, 0), 3)
        
        combo = np.hstack([im1, im2])
        out_name = os.path.join(out_dir, f"{prefix}_p{i+1:02d}.png")
        cv2.imwrite(out_name, combo)
        print("Generated QA:", out_name)
        
    d_orig.close()
    d_clean.close()

if __name__ == "__main__":
    make_qa("20251020202938305.pdf", "output_cleaned/20251020202938305_cleaned.pdf", "hybrid_20251020202938305")
    make_qa("20251025101835564.pdf", "output_cleaned/20251025101835564_cleaned.pdf", "hybrid_20251025101835564")
