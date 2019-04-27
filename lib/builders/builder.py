from pathlib import Path
from pycocotools.coco import COCO
from tqdm import tqdm
import os


class Builder:
    def __init__(self, config):
        annotations_path = Path(config["source"], "annotations",
                                "stuff_" + self.SPLIT + "2017.json")
        self.coco = COCO(annotations_path)

    def build(self, config):
        raise NotImplementedError

        # Load image ids
        cat_ids = self.coco.getCatIds(supNms=config["supercategories"])
        img_ids = self.coco.getImgIds(catIds=cat_ids)
        img_ids = img_ids[:len(img_ids) * config["size fraction"]]

        # Setup paths
        split_path = Path(config["destination"], self.SPLIT)
        image_path = Path(split_path, "images")
        if not os.path.exists(image_path):
            os.mkdir(image_path)

        target_path = Path(split_path, "targets")
        if not os.path.exists(target_path):
            os.mkdir(image_path)

        # Build the dataset
        print("Building " + self.SPLIT + " split...")
        for img_id in tqdm(img_ids):
            pass

    def _build_image(self, img_id, image_path):
        raise NotImplementedError


class ValBuilder(Builder):
    SPLIT = "val"


class TrainBuilder(Builder):
    SPLIT = "train"