from PIL import Image
from pathlib import Path
from skimage.transform import resize
from torch.utils.data import DataLoader
import csv
import numpy as np
import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

N_CLASSES = 2
IMG_HEIGHT = 426
IMG_WIDTH = 640


def get_coco_sky_weak_bb(data_root=None, batch_size=None):
    if not data_root:
        data_root = Path(
            "D:/code/data/filtered_datasets/coco_stuff_sky_weak_bb"
        ) if os.name == "nt" else Path(
            "/mnt/hdd-4tb/abhay/datasets/coco_stuff_sky_weak_bb")

    train_data_path = Path(data_root, "train")
    val_data_path = Path(data_root, "val")

    if not batch_size:
        batch_size = 2 if os.name == "nt" else 7

    train_dataset = Coco(
        root=train_data_path,
        ann_file_path=os.path.join(train_data_path, "annotations.csv"))
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = Coco(
        root=val_data_path,
        ann_file_path=os.path.join(val_data_path, "annotations.csv"))
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, N_CLASSES


class Coco(data.Dataset):
    def __init__(self, root, ann_file_path):
        self.root = root
        self.img_names = []
        self.bbox_indexes = []
        with open(ann_file_path, newline='') as ann_file:
            reader = csv.reader(ann_file, delimiter=',')
            for row in reader:
                self.img_names.append(row[0])
                self.bbox_indexes.append(row[1])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is a list of captions for the
                   image.
        """
        img_name = self.img_names[index]
        img_path = Path(self.root, "images", img_name)
        img = Image.open(img_path).convert('RGB')
        img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.ANTIALIAS)
        img = transforms.ToTensor()(img)

        seg_name = img_name.replace(".jpg", ".png")
        seg_path = Path(self.root, "annotations", seg_name)
        seg = Image.open(seg_path)
        S = np.array(seg)
        S = resize(
            S, (IMG_HEIGHT, IMG_WIDTH), anti_aliasing=False, mode='constant')
        S = np.where(S > 0, 1, 0)
        seg = torch.from_numpy(S)

        bbox_name = seg_name.replace(
            ".png", "-" + str(self.bbox_indexes[index]) + ".png")
        bbox_path = Path(self.root, "bbox", bbox_name)
        bbox = Image.open(bbox_path)
        B = np.array(bbox)
        B = resize(
            B, (IMG_HEIGHT, IMG_WIDTH), anti_aliasing=False, mode='constant')
        B = np.where(B > 0, 1, 0)
        bbox = torch.from_numpy(B)

        return img, (seg, bbox)

    def __len__(self):
        return len(self.img_names)