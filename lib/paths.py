from pathlib import Path
import os


def get_windows_paths():
    return {
        "filtered_data_root":
        Path("D:/code/data/filtered_datasets"),
        "train_ann_file":
        Path("D:/code/data/cocostuff/dataset/annotations/",
             "stuff_train2017.json"),
        "train_root":
        Path("D:/code/data/cocostuff/dataset/images/train2017"),
        "val_ann_file":
        Path("D:/code/data/cocostuff/dataset/annotations/",
             "stuff_val2017.json"),
        "val_root":
        Path("D:/code/data/cocostuff/dataset/images/val2017"),
    }


def get_ubuntu_paths():
    return {
        "filtered_data_root":
        Path("/mnt/hdd-4tb/testuser2/datasets"),
        "train_ann_file":
        Path("/mnt/hdd-4tb/abhay/cocostuff/dataset/annotations/",
             "stuff_train2017.json"),
        "train_root":
        Path("/mnt/hdd-4tb/abhay/cocostuff/dataset/images/train2017"),
        "val_ann_file":
        Path("/mnt/hdd-4tb/abhay/cocostuff/dataset/annotations/",
             "stuff_val2017.json"),
        "val_root":
        Path("/mnt/hdd-4tb/abhay/cocostuff/dataset/images/val2017"),
    }


def get_paths():
    return get_windows_paths() if os.name == "nt" else get_ubuntu_paths()