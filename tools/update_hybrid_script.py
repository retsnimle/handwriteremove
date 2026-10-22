import os

file_path = "G:/CODE/handwriteremove/tools/dataset_hybrid.py"
with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

text = text.replace("\r\n", "\n")

# 1. 自適應 anchors_path
old_anchors = """        # 載入題號與作答框錨點
        if os.path.exists(anchors_path):"""

new_anchors = """        # 載入題號與作答框錨點
        if not os.path.exists(anchors_path):
            fallback_anchors = os.path.join(assets_root, "exam_anchors_catalog.json")
            if os.path.exists(fallback_anchors):
                anchors_path = fallback_anchors

        if os.path.exists(anchors_path):"""

text = text.replace(old_anchors, new_anchors)

# 2. 升級蓋印邏輯
old_stamp_start = "                # --- 蓋印作答括號 ---"
old_stamp_end = "                    self.overlay_stamp(synth_img, synth_mask, stamp, rx, ry,\n                                       scale=scale, angle=random.uniform(-12, 12))"

new_stamp = """                # --- 蓋印作答括號 ---
                for anc in sampled_brackets:
                    # 90% 概率蓋上真實手寫 A B C D E 字母
                    use_letter = (len(self.letter_marks) > 0) and (random.random() < 0.90)
                    if use_letter:
                        m = random.choice(self.letter_marks)
                    else:
                        m = random.choice(self.symbol_marks) if self.symbol_marks else random.choice(self.all_marks)

                    stamp = cv2.imread(os.path.join(self.marks_dir, m["file"]), cv2.IMREAD_UNCHANGED)
                    if stamp is None: continue
                    sh, sw = stamp.shape[:2]

                    # 括號蓋印：70% 正常尺度，30% 豪邁大字 (scale 1.20 ~ 1.70) 自然溢出覆蓋括號
                    if random.random() < 0.30:
                        scale = random.uniform(1.20, 1.70)
                    else:
                        scale = random.uniform(0.85, 1.15)
                    nw = int(sw * scale)
                    nh = int(sh * scale)

                    # 以括號中心為基準落筆
                    ix1, iy1, ix2, iy2 = anc["inner_bbox"]
                    cx = (ix1 + ix2) // 2
                    cy = (iy1 + iy2) // 2
                    rx = cx - nw // 2 + random.randint(-6, 6)
                    ry = cy - nh // 2 + random.randint(-6, 6)

                    self.overlay_stamp(synth_img, synth_mask, stamp, rx, ry,
                                       scale=scale, angle=random.uniform(-10, 10))

                # --- 蓋印題號 ---
                for anc in sampled_qnums:
                    # 題號落筆：若是字母答案，提供多種真實學生行為分佈
                    use_letter = (len(self.letter_marks) > 0) and (random.random() < (0.85 if not brackets else 0.50))
                    if use_letter:
                        m = random.choice(self.letter_marks)
                    else:
                        m = random.choice(self.symbol_marks) if self.symbol_marks else random.choice(self.all_marks)

                    stamp = cv2.imread(os.path.join(self.marks_dir, m["file"]), cv2.IMREAD_UNCHANGED)
                    if stamp is None: continue
                    sh, sw = stamp.shape[:2]

                    qx1, qy1, qx2, qy2 = anc["bbox"]
                    qw, qh = max(10, qx2 - qx1), max(10, qy2 - qy1)

                    if use_letter:
                        mode = random.random()
                        if mode < 0.40:
                            # 模式 A: 答案寫在題號左側空白 (正常尺寸)
                            scale = random.uniform(0.85, 1.15)
                            nw, nh = int(sw * scale), int(sh * scale)
                            rx = qx1 - nw - random.randint(4, 15)
                            ry = qy1 + (qh - nh) // 2 + random.randint(-4, 4)
                        elif mode < 0.80:
                            # 模式 B: 豪邁大字直接跨越覆蓋在題號與作答區之上 (scale 1.25 ~ 1.85)
                            scale = random.uniform(1.25, 1.85)
                            nw, nh = int(sw * scale), int(sh * scale)
                            rx = qx1 + (qw - nw) // 2 + random.randint(-10, 20)
                            ry = qy1 + (qh - nh) // 2 + random.randint(-15, 15)
                        else:
                            # 模式 C: 正常尺寸手寫字母直接覆蓋在題號上
                            scale = random.uniform(0.90, 1.20)
                            nw, nh = int(sw * scale), int(sh * scale)
                            rx = qx1 + (qw - nw) // 2 + random.randint(-4, 4)
                            ry = qy1 + (qh - nh) // 2 + random.randint(-4, 4)
                    else:
                        # 覆蓋/圈記在題號上 (可稍微偏大)
                        scale = random.uniform(0.90, 1.35)
                        nw, nh = int(sw * scale), int(sh * scale)
                        rx = qx1 + (qw - nw) // 2 + random.randint(-4, 4)
                        ry = qy1 + (qh - nh) // 2 + random.randint(-4, 4)

                    self.overlay_stamp(synth_img, synth_mask, stamp, rx, ry,
                                       scale=scale, angle=random.uniform(-12, 12))"""

idx1 = text.find(old_stamp_start)
idx2 = text.find(old_stamp_end)
if idx1 != -1 and idx2 != -1:
    target_len = idx2 + len(old_stamp_end) - idx1
    text = text[:idx1] + new_stamp + text[idx1 + target_len:]
    print("dataset_hybrid.py stamp section replaced!")
else:
    print(f"Indices for hybrid stamps: {idx1}, {idx2}")

old_mask = "pure_text_mask = (text_core_mask > 0) & (synth_mask == 0)"
new_mask = """# 稍微膨脹手寫 mask (kernel 3x3)，確保手寫筆劃邊緣的過渡像素不會被算作純印刷文字遭受 3.0x 誤殺罰分！
        synth_mask_dilated = cv2.dilate(synth_mask, np.ones((3, 3), np.uint8), iterations=1)
        pure_text_mask = (text_core_mask > 0) & (synth_mask_dilated == 0)"""

text = text.replace(old_mask, new_mask)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(text)

print("dataset_hybrid.py updated successfully!")
