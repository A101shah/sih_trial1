import os
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image

from datasets.CD_dataset import ImageDataset
from models.basic_model import CDEvaluator


# ============================================================
# SETTINGS
# ============================================================

DEVICE_GPU = "0"

PROJECT_NAME = (
    "CD_ChangeFormerV6_LEVIR_b16_lr0.0001_adamw_train_test_200_"
    "linear_ce_multi_train_True_multi_infer_False_shuffle_AB_False_"
    "embed_dim_256"
)

CHECKPOINT_DIR = os.path.join(
    "checkpoints",
    PROJECT_NAME
)

OUTPUT_DIR = "custom_output"

# Your custom images
DATA_ROOT = "custom_input"

# ChangeFormer uses 256x256 inputs
IMG_SIZE = 256


# ============================================================
# CREATE ARGUMENTS FOR CHANGEFORMER
# ============================================================

args = SimpleNamespace(
    n_class=2,
    embed_dim=256,
    net_G="ChangeFormerV6",
    gpu_ids=[0],
    checkpoint_dir=CHECKPOINT_DIR,
    output_folder=OUTPUT_DIR
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

print("======================================")
print("ChangeFormer Custom Image Test")
print("======================================")
print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# LOAD CUSTOM DATASET
# ============================================================

dataset = ImageDataset(
    root_dir=DATA_ROOT,
    split="demo",
    img_size=IMG_SIZE,
    is_train=False,
    to_tensor=True
)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)

print("Images found:", len(dataset))


# ============================================================
# LOAD CHANGEFORMER
# ============================================================

print("Loading ChangeFormerV6...")

model = CDEvaluator(args)

model.load_checkpoint("best_ckpt.pt")
model.eval()

print("ChangeFormer loaded successfully.")


# ============================================================
# RUN INFERENCE
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

for batch in loader:

    name = batch["name"][0]

    print()
    print("Processing:", name)

    # Move images to GPU
    img1 = batch["A"].to(device)
    img2 = batch["B"].to(device)

    # Run ChangeFormer
    with torch.no_grad():
        prediction = model.net_G(img1, img2)[-1]

    # Convert logits to class prediction
    prediction = torch.argmax(
        prediction,
        dim=1
    )

    prediction = prediction.squeeze().cpu().numpy()

    # 0 = unchanged
    # 1 = changed

    changed_pixels = np.sum(prediction == 1)
    total_pixels = prediction.size

    change_percentage = (
        changed_pixels / total_pixels
    ) * 100

    print(
        f"Detected change: {change_percentage:.2f}%"
    )

    # Convert to black/white change map
    change_map = (
        prediction * 255
    ).astype(np.uint8)

    output_path = os.path.join(
        OUTPUT_DIR,
        "change_map.png"
    )

    Image.fromarray(
        change_map
    ).save(output_path)

    print(
        "Saved change map:",
        output_path
    )


print()
print("======================================")
print("DONE")
print("======================================")
