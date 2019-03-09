from pathlib import Path
import platform


def get_anton_paths():
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


def get_tesla_paths():
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


def get_leibniz_paths():
    return {
        "filtered_data_root":
        Path("/home/abhay/data/filtered_datasets"),
        "train_ann_file":
        Path("/home/abhay/data/cocostuff/dataset/annotations/",
             "stuff_train2017.json"),
        "train_root":
        Path("/home/abhay/data/cocostuff/dataset/images/train2017"),
        "val_ann_file":
        Path("/home/abhay/data/cocostuff/dataset/annotations/",
             "stuff_val2017.json"),
        "val_root":
        Path("/home/abhay/data/cocostuff/dataset/images/val2017"),
    }


def get_paths():
    if platform.node() == "Anton":
        return get_anton_paths()
    elif platform.node() == "leibniz":
        return get_leibniz_paths()
    elif platform.node() == "tesla":
        return get_tesla_paths()
