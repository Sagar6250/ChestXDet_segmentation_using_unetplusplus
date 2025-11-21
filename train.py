import os
import csv
import torch
from torch.utils.data import DataLoader
from dataset import ChestXDetDataset
import segmentation_models_pytorch as smp
from tqdm import tqdm
from dataset import ChestXDetDataset
import argparse

import warnings
from timm.scheduler import CosineLRScheduler

warnings.filterwarnings("ignore", message=".*iCCP:.*")
# -----------------------------
# Config
# -----------------------------
parser = argparse.ArgumentParser(description="Train UNet++ on ChestXDet dataset")
parser.add_argument(
    "--experiment",
    type=str,
    required=True,
    help="Name of the experiment (used for saving logs, checkpoints, etc.)"
)
parser.add_argument(
    "--resume",
    type=str,
    default=None,
    help="Path to checkpoint to resume training from"
)
args = parser.parse_args()


num_classes = 13
epochs = 100
batch_size = 8
lr = 1e-4
device = "cuda" if torch.cuda.is_available() else "cpu"

train_img_dir = "/scratch/ssubbar8/Segmentation/train_data"
train_mask_dir = "/scratch/ssubbar8/Segmentation/train_data/mask"
test_img_dir = "/scratch/ssubbar8/Segmentation/test_data"
test_mask_dir = "/scratch/ssubbar8/Segmentation/test_data/mask"

save_dir = f"/scratch/ssubbar8/Segmentation/models/{args.experiment}"
os.makedirs(save_dir, exist_ok=True)

disease_names = [
    "Atelectasis", "Calcification", "Cardiomegaly", "Consolidation",
    "Diffuse Nodule", "Effusion", "Emphysema", "Fibrosis",
    "Fracture", "Mass", "Nodule", "Pleural Thickening", "Pneumothorax"
]

# -----------------------------
# Dataset & Dataloader
# -----------------------------
train_dataset = ChestXDetDataset(images_path=train_img_dir, image_size=(512,512), split="train", normalization='imagenet')
test_dataset = ChestXDetDataset(images_path=test_img_dir,  image_size=(512,512), split="test", normalization='imagenet')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

# -----------------------------
# Model
# -----------------------------
model = smp.UnetPlusPlus(
    encoder_name="resnet50",
    encoder_weights="imagenet",
    in_channels=3,
    classes=num_classes,
).to(device)

dice_loss = smp.losses.DiceLoss(mode='multilabel') 
focal_loss = smp.losses.FocalLoss(mode='multilabel')
# optimizer = optim.AdamW(model.parameters(), lr=lr)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
# scheduler = CosineLRScheduler(
#     optimizer,
#     t_initial=epochs,         # cosine period (100 epochs)
#     lr_min=1e-6,                  # minimum LR at the end of decay
#     warmup_t=5,       # number of warmup epochs
#     warmup_lr_init=1e-7,          # LR at start of warmup
#     t_in_epochs=True,             # step once per epoch
# )

# -----------------------------
# CSV Logger
# -----------------------------
csv_file = f"./CSVs/{args.experiment}_metrics.csv"
if os.path.exists(csv_file):
    os.remove(csv_file)
with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "test_loss", "mean_dice"] + disease_names)

# -----------------------------
# Dice Metric C,B,H,W
# -----------------------------
def dice_score(pred, target, eps=1e-7, threshold=0.5):
    # print(pred.shape)
    # print(target.shape)
    pred = torch.sigmoid(pred) > threshold
    inter = (pred * target).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))
    dice = (2 * inter + eps) / (union + eps)
    return dice.mean(dim=0)

# -----------------------------
# Training Loop
# -----------------------------
start_epoch = 0
best_dice = 0.0

if args.resume is not None and os.path.isfile(args.resume):
    print(f"Resuming from checkpoint: {args.resume}")
    checkpoint = torch.load(args.resume, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = checkpoint["epoch"] + 1
    best_dice = checkpoint.get("best_dice", 0.0)

    print(f"→ Resumed at epoch {start_epoch} (best_dice={best_dice:.4f})")
else:
    print("Starting fresh training.")

for epoch in range(start_epoch, epochs):
    model.train()
    train_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}] Training", leave=False)
    for imgs, masks in pbar:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = 0.8 * dice_loss(outputs, masks) + 0.2 * focal_loss(outputs, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    model.eval()
    test_loss = 0
    dices = torch.zeros(num_classes).to(device)
    pbar_test = tqdm(test_loader, desc=f"Epoch [{epoch+1}/{epochs}] Testing", leave=False)
    with torch.no_grad():
        for imgs, masks in pbar_test:
            imgs, masks = imgs.to(device), masks.to(device)
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
