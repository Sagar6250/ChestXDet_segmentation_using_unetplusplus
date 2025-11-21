import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import ChestXDetDataset
import segmentation_models_pytorch as smp
import numpy as np
import csv, os
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import colormaps as cm
from dataset import ChestXDetDataset
import segmentation_models_pytorch as smp
import argparse

import warnings
from timm.scheduler import CosineLRScheduler

num_classes = 13
epochs = 100
batch_size = 8
lr = 1e-4
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_unetplusplus(checkpoint_path, num_classes, in_channels=1, encoder_name="resnet50", device='cuda'):
    """
    Loads a Unet++ model from segmentation_models_pytorch with given encoder and weights.
    """
    model = smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights=None,  # grayscale input
        in_channels=in_channels,
        classes=num_classes,
        activation=None,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def dice_score(pred, target, eps=1e-7, threshold=0.5):
    pred = torch.sigmoid(pred) > threshold
    inter = (pred * target).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))
    print(inter.shape)
    print(union.shape)
    dice = (2 * inter + eps) / (union + eps)
    return dice.mean(dim=0)

if __name__ == "__main__":
    checkpoint_path = "/scratch/ssubbar8/Segmentation/models/0.8_0.2_1e-4_resnet50_from70/best_model.pth"
    
    test_img_dir = "/scratch/ssubbar8/Segmentation/test_data"
    test_dataset = ChestXDetDataset(images_path=test_img_dir,  image_size=(512,512), split="test", normalization='imagenet')
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=8)
    
    disease_names = [
        "Atelectasis", "Calcification", "Cardiomegaly", "Consolidation",
        "Diffuse Nodule", "Effusion", "Emphysema", "Fibrosis",
        "Fracture", "Mass", "Nodule", "Pleural Thickening", "Pneumothorax"
    ]

    csv_file = f"./CSVs/threshold_metrics.csv"
    if os.path.exists(csv_file):
        os.remove(csv_file)
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["threshold", "mean_dice"] + disease_names)

    model = load_unetplusplus(
        checkpoint_path=checkpoint_path,
        num_classes=13,
        in_channels=3,          # grayscale input
        encoder_name="resnet50",
        device=device
    )
    for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.98]:
        model.eval()
        test_loss = 0
        dices = torch.zeros(num_classes).to(device)
        pbar_test = tqdm(test_loader, desc=f"Threshold {thresh} Testing", leave=False)
        with torch.no_grad():
            for imgs, masks in pbar_test:
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                # loss = 0.8* dice_loss(outputs, masks) + 0.2 * focal_loss(outputs, masks)
                # test_loss += loss.item()
                dices += dice_score(outputs, masks, 1e-7, thresh)
                # pbar_test.set_postfix(loss=f"{loss.item():.4f}")

            dices /= len(test_loader)
            mean_dice = dices.mean().item()

            print(f"\nThreshold {thresh}")
            print(f"Mean Dice: {mean_dice:.4f}")

            with open(csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([thresh, mean_dice] + [d.item() for d in dices])