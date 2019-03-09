from PIL import Image
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy
import csv
import numpy as np
import os
import sys
import torch
import torchvision.datasets.folder as datasets
import torchvision.transforms as transforms
sys.path.append("lib/datasets/torchsample/")
sys.path.append("lib/datasets/torchsample/torchsample/")
sys.path.append("lib/datasets/torchsample/torchsample/transforms/")
from affine_transforms import RandomRotate, RandomChoiceShear  # noqa E402
from affine_transforms import RandomChoiceZoom  # noqa E402


class RandomTranslateWithReflect:
    """Translate image randomly
    Translate vertically and horizontally by n pixels where
    n is integer drawn uniformly independently for each axis
    from [-max_translation, max_translation].
    Fill the uncovered blank area with reflect padding.
    """

    def __init__(self, max_translation):
        self.max_translation = max_translation

    def __call__(self, old_image):
        xtranslation, ytranslation = np.random.randint(
            -self.max_translation, self.max_translation + 1, size=2)
        xpad, ypad = abs(xtranslation), abs(ytranslation)
        xsize, ysize = old_image.size

        flipped_lr = old_image.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_tb = old_image.transpose(Image.FLIP_TOP_BOTTOM)
        flipped_both = old_image.transpose(Image.ROTATE_180)

        new_image = Image.new("RGB", (xsize + 2 * xpad, ysize + 2 * ypad))

        new_image.paste(old_image,
                        (xpad, ypad))  # 2-tuple giving the upper left corner

        new_image.paste(flipped_lr, (xpad + xsize - 1, ypad))
        new_image.paste(flipped_lr, (xpad - xsize + 1, ypad))

        new_image.paste(flipped_tb, (xpad, ypad + ysize - 1))
        new_image.paste(flipped_tb, (xpad, ypad - ysize + 1))

        new_image.paste(flipped_both, (xpad - xsize + 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad - xsize + 1, ypad + ysize - 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad + ysize - 1))

        new_image = new_image.crop(
            (xpad - xtranslation, ypad - ytranslation,
             xpad + xsize - xtranslation, ypad + ysize - ytranslation))
        return new_image


class ImageNetLoader(datasets.ImageFolder):
    # TODO: Add data downloading code
    def __init__(self, root, transform=None, fraction=1):
        datasets.ImageFolder.__init__(self, root, transform)
        self.data, self.labels = [], []
        self.subset_length = round(len(self.imgs) * fraction)
        self.imgs = self.imgs[:self.subset_length]
        print("Loading images into memory ... ")
        class_id = None
        for img in tqdm(self.imgs):
            class_id_ = Path(img[0]).stem.split("_")[0]
            if not class_id or class_id != class_id_:
                class_id = class_id_
                boxes_file_path = Path(root, class_id, class_id + "_boxes.txt")
                with open(boxes_file_path) as csvfile:
                    reader = csv.reader(csvfile, delimiter='\t')
                    boxes = {
                        row[0]:
                        [int(row[1]),
                         int(row[2]),
                         int(row[3]),
                         int(row[4])]
                        for row in reader
                    }

            img_ = Image.open(Path(img[0])).convert('RGB')
            self.data.append(np.asarray(img_))
            label = img[1]
            box = boxes[Path(img[0]).stem + ".JPEG"]
            self.labels.append(np.asarray([label] + box))

    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]

        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        return img, label

    def __len__(self):
        return len(self.data)


class ImageNetValLoader(ImageNetLoader):
    def __init__(self, root, transform=None, fraction=1):
        datasets.ImageFolder.__init__(self, root, transform)
        self.data, self.labels, self.boxes = [], [], []
        self.subset_length = round(len(self.imgs) * fraction)
        self.imgs = self.imgs[:self.subset_length]
        print("Loading images into memory ... ")
        annotations_file = Path(root, "val_annotations.txt")
        with open(annotations_file) as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            boxes = {
                row[0]: [int(row[2]),
                         int(row[3]),
                         int(row[4]),
                         int(row[5])]
                for row in reader
            }
        for img in tqdm(self.imgs):
            img_ = Image.open(Path(img[0])).convert('RGB')
            self.data.append(np.asarray(img_))
            label = img[1]
            box = boxes[Path(img[0]).stem + ".JPEG"]
            self.labels.append(np.asarray([label] + box))


INPUT_SIZE = 32


def all_transforms(x):
    tens = transforms.ToTensor()
    norm = transforms.Normalize(
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    obj0 = RandomChoiceShear([-10, -5, 0, 5, 10])
    obj1 = RandomChoiceZoom([0.8, 0.9, 1.1, 1.2])
    obj2 = RandomRotate(10)
    obj3 = transforms.Lambda(augmented_crop_imagenet)
    obj4 = transforms.RandomHorizontalFlip()
    obj5 = transforms.ColorJitter(
        brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)

    x = obj5(obj4(obj3(x)))
    case = np.random.randint(3, size=1)[0]
    if case == 0:
        return obj0(norm(tens(x)))
    elif case == 1:
        return obj1(norm(tens(x)))
    else:
        return obj2(norm(tens(x)))


def augment_affine_imagenet():
    transform_train = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.Lambda(all_transforms),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return transform_train, transform_test


def augmented_crop_imagenet(x):
    case = np.random.randint(2, size=1)[0]
    if case == 0:
        obj = RandomTranslateWithReflect(4)
    else:
        obj = transforms.RandomCrop(
            INPUT_SIZE, padding=np.random.randint(5, size=1)[0])
    return obj(x)


def augment_mean_imagenet():
    transform_train = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.Lambda(augmented_crop_imagenet),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return transform_train, transform_test


def noaug_imagenet():
    transform_train = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return transform_train, transform_test


def get_loaders(dataset,
                nb_labelled,
                batch_size,
                unlab_rat,
                lab_inds=[],
                augment_type="affine"):

    if augment_type == "affine":
        transform_train, transform_test = augment_affine_imagenet()
    elif augment_type == "mean":
        transform_train, transform_test = augment_mean_imagenet()
    elif augment_type == "no":
        transform_train, transform_test = noaug_imagenet()

    trainset_l = ImageNetLoader(root=train_root, transform=transform_train)
    test_set = ImageNetValLoader(root=val_root, transform=transform_test)
    return train_loader, val_loader, Coco.N_CLASSES
