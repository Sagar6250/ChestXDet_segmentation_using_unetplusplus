import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import json

# class ChestXDetDataset(Dataset):
#     def __init__(self, img_dir, mask_dir, transform=None):
#         """
#         Args:
#             img_dir: directory containing images (.png)
#             mask_dir: directory containing npy masks (same filename prefix)
#             transform: optional torchvision transforms for data augmentation
#         """
#         self.img_dir = img_dir
#         self.mask_dir = mask_dir
#         self.transform = transform

#         # all image files (e.g. 36204.png)
#         self.img_files = sorted([
#             f for f in os.listdir(img_dir)
#             if f.endswith(".png") or f.endswith(".jpg")
#         ])

#     def __len__(self):
#         return len(self.img_files)

#     def __getitem__(self, idx):
#         img_name = self.img_files[idx]
#         img_path = os.path.join(self.img_dir, img_name)
#         mask_path = os.path.join(self.mask_dir, img_name.replace(".png", ".npy"))

#         # Load image
#         image = Image.open(img_path).convert("RGB")
#         image = np.array(image, dtype=np.float32) / 255.0  # normalize to [0,1]
#         image = torch.from_numpy(image).permute(2, 0, 1)   # HWC → CHW

#         # Load mask
#         mask = np.load(mask_path)  # shape: (13, H, W)
#         mask = torch.from_numpy(mask.astype(np.float32))

#         # Apply transforms if provided
#         if self.transform:
#             augmented = self.transform(image=image, mask=mask)
#             image = augmented["image"]
#             mask = augmented["mask"]

#         return image, mask


import os
import cv2
import json
import numpy as np
from torch.utils.data import Dataset

class ChestXDetDataset(Dataset):  # From DongAo
    def __init__(self, images_path, split, image_size=(224,224), anno_percent=100, normalization=None):
        # self.augmentation = augment
        self.img_list = []
        self.img_label = []
        self.image_size = image_size
        self.normalization = normalization
        self.disease_labels = [
            "Atelectasis", "Calcification", "Cardiomegaly", "Consolidation", "Diffuse Nodule", "Effusion", 
            "Emphysema", "Fibrosis", "Fracture", "Mass", "Nodule", "Pleural Thickening", "Pneumothorax"
        ]
        
        gt_file = os.path.join(images_path, f"ChestX_Det_{split}.json")
        images_path = os.path.join(images_path, split)
        with open(gt_file, 'r') as json_file:
            gt_data = json.load(json_file)
            for d in gt_data:
                fname = d['file_name']
                self.img_list.append(os.path.join(images_path, fname))
                self.img_label.append(d)
        
        if anno_percent < 100:
            raise NotImplementedError
    

    def __getitem__(self, index):
        imagePath = self.img_list[index]
        imageData = cv2.imread(imagePath, cv2.IMREAD_COLOR)
        imageData = cv2.cvtColor(imageData, cv2.COLOR_BGR2RGB)
        labelAll = [np.zeros((imageData.shape[:2]), dtype=np.uint8) for _ in range(len(self.disease_labels))]

        imageLabel = self.img_label[index]
        polygons = imageLabel['polygons']
        for i, sym in enumerate(imageLabel['syms']):
            pts = np.array(polygons[i], np.int32)
            label = labelAll[self.disease_labels.index(sym)]
            labelAll[self.disease_labels.index(sym)] = cv2.fillPoly(label, [pts], 1)
        labelAll = np.stack([cv2.resize(label, self.image_size, interpolation=cv2.INTER_AREA) for label in labelAll])
        mask = labelAll.transpose((1, 2, 0))

        image = cv2.resize(imageData, self.image_size, interpolation=cv2.INTER_AREA)
        
        # if self.augmentation:
        #     augmented = self.augmentation(image=image, mask=mask)
        #     image = augmented['image']
        #     mask = augmented['mask']
        #     image = np.array(image) / 255.
        # else:
        image = np.array(image) / 255.
        if self.normalization == "imagenet":
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            image = (image - mean) / std

        image = image.transpose((2, 0, 1)).astype('float32')
        mask = mask.transpose((2, 0, 1))

        return image, mask

    def __len__(self):
        return len(self.img_list)
