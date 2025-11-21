import os
import json
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

# -----------------------------
# Config
# -----------------------------
img_dir = "/scratch/ssubbar8/Segmentation/test_data/test"
json_path = "./test/ChestX_Det_test.json"
mask_dir = "/scratch/ssubbar8/Segmentation/test_data/mask"
# os.makedirs(mask_out_dir, exist_ok=True)

with open(json_path, "r") as f:
    data_list = json.load(f)

# Class names (consistent order)
CLASSES = [
    "Atelectasis", "Calcification", "Cardiomegaly", "Consolidation", "Diffuse Nodule",
    "Effusion", "Emphysema", "Fibrosis", "Fracture", "Mass",
    "Nodule", "Pleural Thickening", "Pneumothorax"
]

class_to_idx = {cls: i for i, cls in enumerate(CLASSES)}

# --- Convert ---
for item in tqdm(data_list, desc="Converting masks"):
    file_name = item["file_name"]
    polygons = item["polygons"]
    syms = item["syms"]

    # Load image to get size (important for correct mask dimensions)
    img_path = os.path.join(img_dir, file_name)
    with Image.open(img_path) as img:
        w, h = img.size

    mask = np.zeros((len(CLASSES), h, w), dtype=np.uint8)

    for poly, sym in zip(polygons, syms):
        if sym not in class_to_idx:
            continue
        class_idx = class_to_idx[sym]

        # Convert polygon list to tuple pairs
        if len(poly) > 0 and isinstance(poly[0], (list, tuple)):
            poly_points = [tuple(p) for p in poly]
            img_mask = Image.new("L", (w, h), 0)
            ImageDraw.Draw(img_mask).polygon(poly_points, outline=1, fill=1)
            mask[class_idx] = np.maximum(mask[class_idx], np.array(img_mask))

    np.save(os.path.join(mask_dir, file_name.replace(".png", ".npy")), mask)