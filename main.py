import os
from datetime import datetime
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
from test import *
from settings import *

import segmentation_models_pytorch as smp
from tqdm import tqdm
from dataset import ChestXDetDataset
from train import train_model

def main(args):
    num_classes = 13
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    wd = args.weight_decay

    print("Running on: ", DEVICE)
    transform = transforms.ToTensor()

    train_img_dir = "/scratch/ssubbar8/Segmentation/train_data"
    # train_mask_dir = "/scratch/ssubbar8/Segmentation/train_data/mask"
    test_img_dir = "/scratch/ssubbar8/Segmentation/test_data"
    # test_mask_dir = "/scratch/ssubbar8/Segmentation/test_data/mask"

    train_dataset = ChestXDetDataset(images_path=train_img_dir, image_size=(512,512), split="train", normalization='imagenet')
    test_dataset = ChestXDetDataset(images_path=test_img_dir,  image_size=(512,512), split="test", normalization='imagenet')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)



    model = smp.UnetPlusPlus(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
    ).to(DEVICE)

    dice_loss = smp.losses.DiceLoss(mode='multilabel')
    focal_loss = smp.losses.FocalLoss(mode='multilabel')
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    # loss_functions = [smp.losses.DiceLoss(mode='multilabel'), smp.losses.FocalLoss(mode='multilabel')]
    
    start_epoch = 0

    if(args.eval):
        print(args.eval)
        checkpoint = torch.load(args.eval, map_location=DEVICE)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict, strict=False)
        model.to(DEVICE)
        model.eval()
        evaluate(model, test_loader)
        return

    if args.resume is not None and os.path.isfile(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=DEVICE)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        start_epoch = checkpoint["epoch"] + 1
        best_dice = checkpoint.get("best_dice", 0.0)

        print(f"→ Resumed at epoch {start_epoch} (best_dice={best_dice:.4f})")
             
    train_model(args, model, optimizer, scheduler, train_loader, test_loader, dice_loss, focal_loss, num_classes, epochs)

if __name__ == '__main__':
    main(args)