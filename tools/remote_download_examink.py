import os
import glob
from huggingface_hub import snapshot_download

local_dir = "/content/ExamInk-Seg"
if not os.path.exists(local_dir):
    print("[Colab] 正在從 Hugging Face 下載 ynyg/ExamInk-Seg 數據集...")
    snapshot_download(
        repo_id="ynyg/ExamInk-Seg",
        repo_type="dataset",
        local_dir=local_dir,
        max_workers=8
    )
    print("[Colab] ★ 下載完成！")
else:
    print("[Colab] ExamInk-Seg 已存在，跳過下載。")

train_source = glob.glob(f"{local_dir}/data/train/source/*.jpg") + glob.glob(f"{local_dir}/data/train/source/*.png")
train_mask = glob.glob(f"{local_dir}/data/train/mask/*.jpg") + glob.glob(f"{local_dir}/data/train/mask/*.png")
print(f"ExamInk samples verified: train_source={len(train_source)}, train_mask={len(train_mask)}")
