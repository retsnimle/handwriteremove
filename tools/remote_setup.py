import os
import subprocess
import glob

print("Listing /content...")
print(os.listdir("/content"))

# 合併 chunk_* -> exam_hybrid_bundle.zip
chunks = sorted(glob.glob("/content/chunk_*"))
if chunks:
    print(f"Found {len(chunks)} chunks, merging...")
    with open("/content/exam_hybrid_bundle.zip", "wb") as out_f:
        for c in chunks:
            with open(c, "rb") as in_f:
                out_f.write(in_f.read())
            os.remove(c)
    print("Merged into /content/exam_hybrid_bundle.zip")

    # 解壓
    import zipfile
    print("Unzipping exam_hybrid_bundle.zip...")
    with zipfile.ZipFile("/content/exam_hybrid_bundle.zip", "r") as z:
        z.extractall("/content")
    print("Unzipped successfully!")
    print("New /content:", os.listdir("/content"))
