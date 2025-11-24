import torch
import csv, os
from tqdm import tqdm
import os
import torch
import numpy as np
from matplotlib import colormaps as cm
from settings import DEVICE

num_classes = 13

def dice_score(pred, target, eps=1e-7, threshold=0.5):
    pred = torch.sigmoid(pred) > threshold
    inter = (pred * target).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))
    print(inter.shape)
    print(union.shape)
    dice = (2 * inter + eps) / (union + eps)
    return dice.mean(dim=0)

def evaluate(model, test_loader):
    
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

    for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.98]:
        model.eval()
        test_loss = 0
        dices = torch.zeros(num_classes).to(DEVICE)
        pbar_test = tqdm(test_loader, desc=f"Threshold {thresh} Testing", leave=False)
        with torch.no_grad():
            for imgs, masks in pbar_test:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
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