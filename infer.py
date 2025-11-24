import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np
import segmentation_models_pytorch as smp
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dataset import ChestXDetDataset  # your dataset class


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


def run_inference(model, dataset, batch_size=2, num_workers=2, device='cuda'):
    """
    Run inference on the first 10 samples of the dataset.
    Returns predicted masks and ground-truth masks as numpy arrays.
    """
    # subset = Subset(dataset, range(10))  # first 10 samples only
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    pred_masks = []
    true_masks = []

    with torch.no_grad():
        for imgs, masks in tqdm(dataloader, desc="Running inference (first 10 samples)"):
            imgs = imgs.to(device)
            outputs = model(imgs)

            preds = torch.sigmoid(outputs)
            preds = (preds > 0.98).float()

            pred_masks.append(preds.cpu().numpy())
            true_masks.append(masks.numpy())

    pred_masks = np.concatenate(pred_masks, axis=0)
    true_masks = np.concatenate(true_masks, axis=0)
    return pred_masks, true_masks


def visualize_and_save(dataset, pred_masks, true_masks, save_dir, show=False):
    """
    Overlay predictions and ground truths on the original image,
    color each class differently, and save side-by-side visualization.
    """
    os.makedirs(save_dir, exist_ok=True)
    COLORS = plt.cm.tab20(np.linspace(0, 1, len(dataset.disease_labels)))[:, :3]  # RGB colors
    class_names = dataset.disease_labels

    for i in range(len(pred_masks)):
        # Load the original image (grayscale)
        img_path = dataset.img_list[i]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, dataset.image_size)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) / 255.0

        overlay_gt = image_rgb.copy()
        overlay_pred = image_rgb.copy()

        for c, cname in enumerate(class_names):
            color = COLORS[c]
            gt_mask = true_masks[i, c].astype(bool)
            pred_mask = pred_masks[i, c].astype(bool)

            overlay_gt[gt_mask] = 0.5 * overlay_gt[gt_mask] + 0.5 * color
            overlay_pred[pred_mask] = 0.5 * overlay_pred[pred_mask] + 0.5 * color

        combined = np.hstack((overlay_gt, overlay_pred))

        # Create legend
        patches = [mpatches.Patch(color=COLORS[c], label=class_names[c]) for c in range(len(class_names))]

        plt.figure(figsize=(12, 6))
        plt.imshow(combined)
        plt.axis("off")
        plt.title(f"Sample {i+1} — Left: Ground Truth | Right: Prediction")
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        plt.tight_layout()

        save_path = os.path.join(save_dir, f"overlay_{i}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        if show:
            plt.show()
        plt.close()

    print(f"✅ Saved {len(pred_masks)} overlay visualizations to {save_dir}/")


if __name__ == "__main__":
    # ==== CONFIG ====
    images_path = "/scratch/ssubbar8/Segmentation/test_data"
    split = "test"
    checkpoint_path = "/scratch/ssubbar8/Segmentation/models/0.8_0.2_1e-4_resnet50_from70/best_model.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = "/scratch/ssubbar8/Segmentation/Inference/0.8_0.2_1e-4_resnet50_from70_0.98_alltest"

    # ==== LOAD DATASET ====
    dataset = ChestXDetDataset(
        images_path=images_path,
        split=split,
        image_size=(512, 512),
        anno_percent=100,
        normalization="imagenet"
    )

    num_classes = len(dataset.disease_labels)

    # ==== LOAD MODEL ====
    model = load_unetplusplus(
        checkpoint_path=checkpoint_path,
        num_classes=num_classes,
        in_channels=3,          # grayscale input
        encoder_name="resnet50",
        device=device
    )

    # ==== RUN INFERENCE ====
    pred_masks, true_masks = run_inference(model, dataset, batch_size=2, num_workers=2, device=device)

    print("✅ Predicted mask shape:", pred_masks.shape)
    print("✅ Ground truth mask shape:", true_masks.shape)

    # ==== VISUALIZE & SAVE ====
    visualize_and_save(dataset, pred_masks, true_masks, save_dir, show=False)
