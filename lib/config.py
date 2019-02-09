import os
from pathlib import Path


def get_windows_config():
    return {
        "filtered_data_root":
        Path("D:/code/data/filtered_datasets/coco_sky_test"),
        "target_supercategories": ["sky"],
        "bbox_type":
        "aggregated",
        "n_boxes":
        10,
        "splits": [
            {
                "name":
                "train",
                "ann_file":
                Path("D:/code/data/cocostuff/dataset/annotations/",
                     "stuff_train2017.json"),
                "data_root":
                Path("D:/code/data/cocostuff/dataset/images/train2017"),
                "fraction":
                0.1
            },
            {
                "name":
                "val",
                "ann_file":
                Path("D:/code/data/cocostuff/dataset/annotations/",
                     "stuff_val2017.json"),
                "data_root":
                Path("D:/code/data/cocostuff/dataset/images/val2017"),
                "fraction":
                0.01
            },
        ]
    }


def get_ubuntu_config():
    return {
        "filtered_data_root":
        Path("/mnt/hdd-4tb/testuser2/datasets/coco_sky_weak_bb_agg"),
        "target_supercategories": ["sky"],
        "bbox_type":
        "aggregated",
        "n_boxes":
        10,
        "splits": [
            {
                "name":
                "train",
                "ann_file":
                Path("/mnt/hdd-4tb/abhay/cocostuff/dataset/annotations/",
                     "stuff_train2017.json"),
                "data_root":
                Path("/mnt/hdd-4tb/abhay/cocostuff/dataset/images/train2017"),
                "fraction":
                0.1
            },
            {
                "name":
                "val",
                "ann_file":
                Path("/mnt/hdd-4tb/abhay/cocostuff/dataset/annotations/",
                     "stuff_val2017.json"),
                "data_root":
                Path("/mnt/hdd-4tb/abhay/cocostuff/dataset/images/val2017"),
                "fraction":
                0.01
            },
        ]
    }


def get_config():
    if os.name == "nt":
        return get_windows_config()
    else:
        return get_ubuntu_config()
