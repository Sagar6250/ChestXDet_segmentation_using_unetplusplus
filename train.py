import os
import csv
import torch
from torch.utils.data import DataLoader
from dataset import ChestXDetDataset
import segmentation_models_pytorch as smp
from tqdm import tqdm
from dataset import ChestXDetDataset
from settings import DEVICE

from timm.scheduler import CosineLRScheduler

def dice_score(pred, target, eps=1e-7, threshold=0.5):
    # print(pred.shape)
    # print(target.shape)
    pred = torch.sigmoid(pred) > threshold
    inter = (pred * target).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))
    dice = (2 * inter + eps) / (union + eps)
    return dice.mean(dim=0)

def train_model(args, model, optimizer, scheduler, train_loader, test_loader, dice_loss, focal_loss, num_classes, start_epoch, epochs):
    
    save_dir = f"/scratch/ssubbar8/Segmentation/models/{args.experiment}"
    os.makedirs(save_dir, exist_ok=True)

    disease_names = [
    "Atelectasis", "Calcification", "Cardiomegaly", "Consolidation",
    "Diffuse Nodule", "Effusion", "Emphysema", "Fibrosis",
    "Fracture", "Mass", "Nodule", "Pleural Thickening", "Pneumothorax"
    ]

    csv_file = f"./CSVs/{args.experiment}_metrics.csv"
    if os.path.exists(csv_file):
        os.remove(csv_file)
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "test_loss", "mean_dice"] + disease_names)

    best_dice = 0.0

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}] Training", leave=False)
        for imgs, masks in pbar:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = 0.8 * dice_loss(outputs, masks) + 0.2 * focal_loss(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        model.eval()
        test_loss = 0
        dices = torch.zeros(num_classes).to(DEVICE)
        pbar_test = tqdm(test_loader, desc=f"Epoch [{epoch+1}/{epochs}] Testing", leave=False)
        with torch.no_grad():
            for imgs, masks in pbar_test:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                outputs = model(imgs)
                loss = 0.8* dice_loss(outputs, masks) + 0.2 * focal_loss(outputs, masks)
                test_loss += loss.item()
                dices += dice_score(outputs, masks)
                pbar_test.set_postfix(loss=f"{loss.item():.4f}")

        dices /= len(test_loader)
        mean_dice = dices.mean().item()

        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss/len(train_loader):.4f} | "
            f"Test Loss: {test_loss/len(test_loader):.4f} | "
            f"Mean Dice: {mean_dice:.4f}")

        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, train_loss/len(train_loader),
                            test_loss/len(test_loader),
                            mean_dice] + [d.item() for d in dices])

        if mean_dice > best_dice:
            best_dice = mean_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_dice': best_dice
            }, os.path.join(save_dir, "best_model.pth"))
            print(f"Saved new best model (Dice={best_dice:.4f})")

        if (epoch+1) % 10 == 0:
            ckpt_path = os.path.join(save_dir, f"epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_dice': best_dice
            }, ckpt_path)
            # run_inference(model, test_dataset, epoch+1)

        scheduler.step()
