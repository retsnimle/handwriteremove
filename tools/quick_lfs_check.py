import os, glob

sources = glob.glob("/content/ExamInk-Seg/data/train/source/*.jpg")
masks = glob.glob("/content/ExamInk-Seg/data/train/mask/*.png")
print(f"Total sources: {len(sources)}, masks: {len(masks)}")

if sources:
    s0 = sources[0]
    m0 = masks[0]
    print(f"File {s0}: {os.path.getsize(s0)} bytes")
    print(f"File {m0}: {os.path.getsize(m0)} bytes")
