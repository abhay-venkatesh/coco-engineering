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
    TRAIN_ANN_FILE = Path(
        "/mnt/hdd-4tb/abhay/cocostuff/dataset/annotations/stuff_train2017.json"
    )
    TRAIN_DATA_ROOT = Path(
        "/mnt/hdd-4tb/abhay/cocostuff/dataset/images/train2017")

    VAL_ANN_FILE = Path(
        "/mnt/hdd-4tb/abhay/cocostuff/dataset/annotations/stuff_val2017.json")
    VAL_DATA_ROOT = Path("/mnt/hdd-4tb/abhay/cocostuff/dataset/images/val2017")

    FILTERED_DATA_ROOT = Path(
        "/mnt/hdd-4tb/abhay/datasets/coco_stuff_sky_weak_bb")
    if not os.path.exists(FILTERED_DATA_ROOT):
        os.makedirs(FILTERED_DATA_ROOT)


def _random_bbox(bbox):
    x, y, w, h = bbox
    x_, y_ = random.randint(x, x + w), random.randint(y, y + h)
    w_, h_ = abs(x_ - random.randint(x_, x + w)), abs(y_ - random.randint(y_, y + h))
    return x_, y_, w_, h_


def _draw_random_bbox_mask(coco, img_id, bbox):
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


def _get_seg_boundary(seg):
    rows, cols = np.nonzero(seg)
    x = min(cols)
    w = max(cols)
    y = min(rows)
    h = max(rows)
    return x, y, w, h


def _draw_random_bbox_from_seg(coco, img_id, seg_array):
    """ We want to get 10 bounding boxes per image. In order to do that,
        we need to:
        1. Convert (x, y, width, height) into 4 coordinates
        2. Generate random 4 coordinates bounded by those 4 coordinates """
    img_height = coco.loadImgs(img_id)[0]['height']
    img_width = coco.loadImgs(img_id)[0]['width']
    seg = Image.fromarray(np.zeros((img_height, img_width)))
    draw = ImageDraw.Draw(seg)
    seg_boundary = _get_seg_boundary(seg_array)
    x, y, mask_width, mask_height = _random_bbox(seg_boundary)
    # x, y, mask_width, mask_height = seg_boundary
    rect = _get_rect(x, y, mask_width, mask_height, 0)
    draw.polygon([tuple(p) for p in rect], fill=1)
    np_seg = np.asarray(seg)
    return np_seg


def _draw_bbox_mask(coco, img_id, bbox):
    """ We want to get 10 bounding boxes per image. In order to do that,
        we need to:
        1. Convert (x, y, width, height) into 4 coordinates
        2. Generate random 4 coordinates bounded by those 4 coordinates """
    img_height = coco.loadImgs(img_id)[0]['height']
    img_width = coco.loadImgs(img_id)[0]['width']
    seg = Image.fromarray(np.zeros((img_height, img_width)))
    draw = ImageDraw.Draw(seg)
    x, y, mask_width, mask_height = bbox
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
    img = Image.open(img_path)
    img.show()
    input("Press Enter to continue...")


def _preview_mask(seg):
    seg_arr = np.asarray(seg)
    seg_arr = np.multiply(seg_arr, 100)
    mask = Image.fromarray(seg_arr)
    mask.show()
    input("Press Enter to continue...")


def _filter_dataset(ann_file_path, data_root, target_supercategories,
                    filtered_data_location):
    """ Filters out image/segmentation pairs that contain the target class. """
    coco = COCO(ann_file_path)
    img_ids = coco.getImgIds()
    target_cat_ids = coco.getCatIds(supNms=target_supercategories)

    for img_id in tqdm(img_ids):
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            if ann["category_id"] in target_cat_ids:
                # Filter image/annotation pairs
                img_name = coco.loadImgs(img_id)[0]['file_name']
                img_path = Path(data_root, img_name)
                img_path_ = Path(filtered_data_location, "images", img_name)
                if not os.path.exists(Path(filtered_data_location, "images")):
                    os.makedirs(Path(filtered_data_location, "images"))
                shutil.copyfile(img_path, img_path_)

                seg_array = coco.annToMask(ann)
                seg = Image.fromarray(seg_array)
                seg_name = coco.loadImgs(img_id)[0]['file_name'].replace(
                    ".jpg", ".png")
                seg_path = Path(filtered_data_location, "annotations",
                                seg_name)
                if not os.path.exists(
                        Path(filtered_data_location, "annotations")):
                    os.makedirs(Path(filtered_data_location, "annotations/"))
                seg.save(seg_path)
                _preview_image(coco, img_id, data_root)

                # Filter bounding boxes
                for i in range(10):
                    bbox = _draw_random_bbox_from_seg(coco, img_id, seg_array)
                    bbox = Image.fromarray(bbox).convert("L")
                    bbox_name = coco.loadImgs(img_id)[0]['file_name'].replace(
                        ".jpg", "-" + str(i) + ".png")
                    bbox_path = Path(filtered_data_location, "bbox", bbox_name)
                    if not os.path.exists(
                            Path(filtered_data_location, "bbox")):
                        os.makedirs(Path(filtered_data_location, "bbox"))
                    bbox.save(bbox_path)
                    _preview_mask(bbox)

    # Filter the annotations.csv file
    filtered_ann_path = Path(filtered_data_location, "annotations.csv")
    with open(filtered_ann_path, mode='w', newline='') as filtered_ann_file:
        writer = csv.writer(
            filtered_ann_file,
            delimiter=',',
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL)
        for img_id in img_ids:
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
            for ann in anns:
                if ann["category_id"] in target_cat_ids:
                    for i in range(10):
                        img_name = coco.loadImgs(img_id)[0]['file_name']
                        writer.writerow([img_name, i])


def build_coco_stuff_weak_bb(target_supercategories=["sky"],
                             should_download=False):
    if should_download:
        raise NotImplementedError("Download functionality not implemented. ")

    for split in zip(["train", "val"], [TRAIN_ANN_FILE, VAL_ANN_FILE],
                     [TRAIN_DATA_ROOT, VAL_DATA_ROOT]):
        FILTERED_SPLIT_FOLDER = Path(FILTERED_DATA_ROOT, "train")
        if not os.path.exists(FILTERED_SPLIT_FOLDER):
            os.mkdir(FILTERED_SPLIT_FOLDER)

        _filter_dataset(split[1], split[2], target_supercategories,
                        FILTERED_SPLIT_FOLDER)
