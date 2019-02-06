from PIL import Image, ImageDraw
from pathlib import Path
from pycocotools.coco import COCO
import csv
import numpy as np
import os
import random
import shutil
from tqdm import tqdm

if os.name == "nt":
    TRAIN_ANN_FILE = Path(
        "D:/code/data/cocostuff/dataset/annotations/stuff_train2017.json")
    TRAIN_DATA_ROOT = Path("D:/code/data/cocostuff/dataset/images/train2017")

    VAL_ANN_FILE = Path(
        "D:/code/data/cocostuff/dataset/annotations/stuff_val2017.json")
    VAL_DATA_ROOT = Path("D:/code/data/cocostuff/dataset/images/val2017")

    FILTERED_DATA_ROOT = Path(
        "D:/code/data/filtered_datasets/coco_stuff_sky_weak_bb")
    if not os.path.exists(FILTERED_DATA_ROOT):
        os.makedirs(FILTERED_DATA_ROOT)

else:
    TRAIN_ANN_FILE = Path("")
    TRAIN_DATA_ROOT = Path("")

    VAL_ANN_FILE = Path("")
    VAL_DATA_ROOT = Path("")

    FILTERED_DATA_ROOT = Path("")
    if not os.path.exists(FILTERED_DATA_ROOT):
        os.makedirs(FILTERED_DATA_ROOT)


def _random_bbox(bbox):
    x, y, w, h = bbox
    x_, y_ = random.randint(x, x + w), random.randint(y, y + h)
    w_, h_ = random.randint(x_, x + w), random.randint(y_, y + h)
    return x_, y_, w_, h_


def _draw_bbox_mask(coco, img_id, bbox):
    """ We want to get 10 bounding boxes per image. In order to do that,
        we need to:
        1. Convert (x, y, width, height) into 4 coordinates
        2. Generate random 4 coordinates bounded by those 4 coordinates """
    img_height = coco.loadImgs(img_id)[0]['height']
    img_width = coco.loadImgs(img_id)[0]['width']
    seg = Image.fromarray(np.zeros((img_height, img_width)))
    draw = ImageDraw.Draw(seg)
    x, y, mask_width, mask_height = _random_bbox(bbox)
    rect = _get_rect(x, y, mask_width, mask_height, 0)
    draw.polygon([tuple(p) for p in rect], fill=1)
    np_seg = np.asarray(seg)
    return np_seg


def _get_rect(x, y, width, height, angle):
    """ Get a rectangle from (x, y) and (width, height) """
    rect = np.array([(0, 0), (width, 0), (width, height), (0, height), (0, 0)])
    theta = (np.pi / 180.0) * angle
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    offset = np.array([x, y])
    transformed_rect = np.dot(rect, R) + offset
    return transformed_rect


def _preview_image(coco, img_id, data_root):
    img_name = coco.loadImgs(img_id)[0]['file_name']
    img_path = Path(data_root, img_name)
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    img = Image.open(img_path)
    cat_ids = [ann["category_id"] for ann in anns]
    print(coco.loadCats(ids=cat_ids))
    img.show()
    input("Press Enter to continue...")


def _filter_dataset(ann_file_path, data_root, target_class,
                    filtered_data_location):
    """ Filters out image/segmentation pairs that contain the target class. """
    coco = COCO(ann_file_path)
    img_ids = coco.getImgIds()

    # Filter image/annotation pairs
    for img_id in tqdm(img_ids):
        _preview_image(coco, img_id, data_root)
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            if ann["category_id"] == target_class:
                img_name = coco.loadImgs(img_id)[0]['file_name']
                img_path = Path(data_root, img_name)
                img_path_ = Path(filtered_data_location, "images", img_name)
                if not os.path.exists(Path(filtered_data_location, "images")):
                    os.makedirs(Path(filtered_data_location, "images"))
                shutil.copyfile(img_path, img_path_)

                mask = coco.annToMask(ann)
                seg = Image.fromarray(mask)
                seg_name = coco.loadImgs(img_id)[0]['file_name'].replace(
                    ".jpg", ".png")
                seg_path = Path(filtered_data_location, "annotations",
                                seg_name)
                if not os.path.exists(
                        Path(filtered_data_location, "annotations")):
                    os.makedirs(Path(filtered_data_location, "annotations/"))
                seg.save(seg_path)

    # Filter bounding boxes
    for img_id in range(len(img_ids)):
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            if ann["category_id"] == target_class:
                for i in range(10):
                    mask = _draw_bbox_mask(coco, img_id, ann["bbox"])
                    seg = Image.fromarray(mask).convert("L")
                    seg_name = coco.loadImgs(img_id)[0]['file_name'].replace(
                        ".jpg", "-" + str(i) + ".png")
                    seg_path = Path(filtered_data_location, "bbox", seg_name)
                    if not os.path.exists(
                            Path(filtered_data_location, "bbox")):
                        os.makedirs(Path(filtered_data_location, "bbox"))
                    seg.save(seg_path)

    # Filter the annotations.csv file
    filtered_ann_path = Path(filtered_data_location, "annotations.csv")
    with open(filtered_ann_path, mode='w', newline='') as filtered_ann_file:
        writer = csv.writer(
            filtered_ann_file,
            delimiter=',',
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL)
        for img_id in range(len(img_ids)):
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
            for ann in anns:
                if ann["category_id"] == target_class:
                    for i in range(10):
                        img_name = coco.loadImgs(img_id)[0]['file_name']
                        writer.writerow([img_name, i])


def build_coco_stuff_weak_bb(target_class=95, should_download=False):
    """ target_class_to_name = {
            95
            97 
            106
            111
            135
            142
            157
            169
            183
        } """
    if should_download:
        raise NotImplementedError("Download functionality not implemented. ")

    for split in zip(["train", "val"], [TRAIN_ANN_FILE, VAL_ANN_FILE],
                     [TRAIN_DATA_ROOT, VAL_DATA_ROOT]):
        FILTERED_SPLIT_FOLDER = Path(FILTERED_DATA_ROOT, "train")
        if not os.path.exists(FILTERED_SPLIT_FOLDER):
            os.mkdir(FILTERED_SPLIT_FOLDER)

        _filter_dataset(split[1], split[2], target_class,
                        FILTERED_SPLIT_FOLDER)
