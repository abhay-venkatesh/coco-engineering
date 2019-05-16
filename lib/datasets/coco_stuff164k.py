from PIL import Image
from pathlib import Path
import numpy as np
import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms


class COCOStuff164k(data.Dataset):
    def __init__(self, root, split="train"):
        self.root = root
        self.image_folder = Path(self.root, "images", split + "2017")
        self.ann_folder = Path(self.root, "annotations", split + "2017")
        self.img_names = [
            f for f in os.listdir(self.image_folder)
            if os.path.isfile(Path(self.image_folder, f))
        ]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is a list of captions for the
                   image.
        """
        img_name = self.img_names[index]
        img_path = Path(self.image_folder, img_name)
        img = Image.open(img_path).convert('RGB')
        img = transforms.ToTensor()(img)

        seg_name = img_name.replace(".jpg", ".png")
        seg_path = Path(self.ann_folder, seg_name)
        seg = Image.open(seg_path)
        seg_array = np.array(seg)
        seg = torch.from_numpy(seg_array)

        return img, seg

    def __len__(self):
        return len(self.img_names)