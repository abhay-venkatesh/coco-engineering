""" Coco Stuff with PU-Labeled Bounding Boxes """

from PIL import Image, ImageDraw
from lib.coco import get_coco_sky_weak_bb
from pathlib import Path
from pycocotools.coco import COCO
from tqdm import tqdm
import csv
import numpy as np
import os
import random
import shutil
import torchvision.transforms as transforms


def _random_bbox(seg_array, smoothing=2):
    """ For example, let
    seg_array = [
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]
    xb, yb, wb, hb = 0, 0, 5, 2 """
    _, _, wb, hb = _get_seg_boundary(seg_array)
    rows, cols = np.nonzero(seg_array)
    ri = random.randint(0, len(rows) - 1)
    """ Then,
    y in [0, 1]
    x in [0, 1, 2, 3, 4] """
    y, x = rows[ri], cols[ri]
    """ We want
    h in [0, hb - y]
    w in [0, wb - x] """
    h, w = random.randint(0, (hb - y) // smoothing), random.randint(
        0, (wb - x) // smoothing)
    return x, y, w, h


def _get_seg_boundary(seg_array):
    rows, cols = np.nonzero(seg_array)
    x = min(cols)
    w = max(cols)
    y = min(rows)
    h = max(rows)
    return x, y, w, h


def _draw_random_bbox_from_seg(coco, img_id, seg_array):
    img_height = coco.loadImgs(img_id)[0]['height']
    img_width = coco.loadImgs(img_id)[0]['width']
    seg = Image.fromarray(np.zeros((img_height, img_width)))
    draw = ImageDraw.Draw(seg)
    x, y, w, h = _random_bbox(seg_array)
    """
    img = [
        [1, 1, 1],
        [0, 0, 0],
        [1, 1, 1],
    ]

    then (x, y) indexes of each element are given by [
        [(0, 0), (0, 1), (0, 2)],
        [(1, 0), (1, 1), (1, 2)],
        [(2, 0), (2, 1), (2, 2)],
    ]
    """
    rect = _get_rect(x, y, w, h, 0)
    draw.polygon([tuple(p) for p in rect], fill=1)
    np_seg = np.asarray(seg, dtype=int)
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


def _filter_bounding_boxes(coco,
                           img_id,
                           seg_array,
                           filtered_data_location,
                           n_samples=10):
    assert n_samples >= 1, (
        "Need to have at least one sample for creating bounding boxes. ")

    if not os.path.exists(Path(filtered_data_location, "bbox")):
        os.makedirs(Path(filtered_data_location, "bbox"))

    for i in range(n_samples):
        bbox = _draw_random_bbox_from_seg(coco, img_id, seg_array)
        bbox = Image.fromarray(bbox).convert("L")
        bbox_name = coco.loadImgs(img_id)[0]['file_name'].replace(
            ".jpg", "-" + str(i) + ".png")
        bbox_path = Path(filtered_data_location, "bbox", bbox_name)
        bbox.save(bbox_path)


def _filter_agg_bounding_boxes(coco,
                               img_id,
                               seg_array,
                               filtered_data_location,
                               n_boxes=10):
    # Filters, and aggregates bounding boxes into a single mask
    assert n_boxes >= 1, ("Need to have at least one bounding box. ")

    if not os.path.exists(Path(filtered_data_location, "bbox")):
        os.makedirs(Path(filtered_data_location, "bbox"))

    bbox = _draw_random_bbox_from_seg(coco, img_id, seg_array)
    for i in range(1, n_boxes):
        bbox_ = bbox | _draw_random_bbox_from_seg(coco, img_id, seg_array)
        bbox = bbox_

    bbox = np.array(bbox, dtype=np.uint8)
    bbox = Image.fromarray(bbox).convert("L")
    bbox_name = coco.loadImgs(img_id)[0]['file_name'].replace(".jpg", "-0.png")
    bbox_path = Path(filtered_data_location, "bbox", bbox_name)
    # _preview_mask(bbox)
    bbox.save(bbox_path)


def _filter_full_bounding_boxes(coco, img_id, ann, filtered_data_location):
    if not os.path.exists(Path(filtered_data_location, "bbox")):
        os.makedirs(Path(filtered_data_location, "bbox"))

    img_height = coco.loadImgs(img_id)[0]['height']
    img_width = coco.loadImgs(img_id)[0]['width']
    seg = Image.fromarray(np.zeros((img_height, img_width)))
    draw = ImageDraw.Draw(seg)
    x, y, w, h = ann["bbox"]
    rect = _get_rect(x, y, w, h, 0)
    draw.polygon([tuple(p) for p in rect], fill=1)
    np_seg = np.asarray(seg, dtype=int)
    _preview_mask(np_seg)
    return np_seg


def _filter_annotations_file(coco, img_ids, target_cat_ids,
                             filtered_data_location, n_boxes, bbox_type,
                             fraction):
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
                    if bbox_type == "aggregated":
                        img_name = coco.loadImgs(img_id)[0]['file_name']
                        writer.writerow([img_name, 0])
                    else:
                        for i in range(n_boxes):
                            img_name = coco.loadImgs(img_id)[0]['file_name']
                            writer.writerow([img_name, 0])


def _filter_dataset(ann_file_path,
                    data_root,
                    target_supercategories,
                    filtered_data_location,
                    bbox_type,
                    n_boxes=10,
                    fraction=1.0):
    coco = COCO(ann_file_path)
    img_ids = coco.getImgIds()
    img_ids = img_ids[:int(fraction * len(img_ids))]
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
                # _preview_image(coco, img_id, data_root)

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
                # _preview_mask(seg)

                if bbox_type == "aggregated":
                    _filter_agg_bounding_boxes(coco, img_id, seg_array,
                                               filtered_data_location, n_boxes)
                elif bbox_type == "full":
                    _filter_full_bounding_boxes(coco, img_id, ann,
                                                filtered_data_location)
                else:
                    _filter_bounding_boxes(coco, img_id, seg_array,
                                           filtered_data_location, n_boxes)

    _filter_annotations_file(coco, img_ids, target_cat_ids,
                             filtered_data_location, n_boxes, bbox_type,
                             fraction)


def build_coco_stuff_bb(paths, config):
    if not os.path.exists(paths["filtered_data_root"]):
        os.makedirs(paths["filtered_data_root"])

    dataset_root = Path(paths["filtered_data_root"], config["name"])
    if not os.path.exists(dataset_root):
        os.makedirs(dataset_root)

    for split in config["splits"]:
        filtered_split_folder = Path(dataset_root, split["name"])
        if not os.path.exists(filtered_split_folder):
            os.mkdir(filtered_split_folder)

        ann_file_name = split["name"] + "_ann_file"
        root_name = split["name"] + "_root"
        _filter_dataset(paths[ann_file_name], paths[root_name],
                        config["target supercategories"],
                        filtered_split_folder, config["bbox type"],
                        config["number of boxes"], split["fraction"])


def verify_coco_stuff_weak_bb(config):
    train_loader, val_loader, _ = get_coco_sky_weak_bb(
        data_root=config["filtered_data_root"], batch_size=1)

    for image, labels in train_loader:
        img_tensor = image.squeeze(0)
        img = transforms.ToPILImage()(img_tensor)
        img.show()
        input("Press Enter to continue...")

        mask_array = labels[0].squeeze(0).numpy()
        mask_array *= 100
        mask = Image.fromarray(mask_array)
        mask.show()
        input("Press Enter to continue...")

        bbox_array = labels[1].squeeze(0).numpy()
        bbox_array *= 100
        bbox = Image.fromarray(bbox_array)
        bbox.show()
        input("Press Enter to continue...")
        break

    ri = random.randint(0, len(val_loader))
    for i, (image, labels) in enumerate(val_loader):
        if i == ri:
            img_tensor = image.squeeze(0)
            img = transforms.ToPILImage()(img_tensor)
            img.show()
            input("Press Enter to continue...")

            mask_array = labels[0].squeeze(0).numpy()
            mask_array *= 100
            mask = Image.fromarray(mask_array)
            mask.show()
            input("Press Enter to continue...")

            bbox_array = labels[1].squeeze(0).numpy()
            bbox_array *= 100
            bbox = Image.fromarray(bbox_array)
            bbox.show()
            input("Press Enter to continue...")
            break
