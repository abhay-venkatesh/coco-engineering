from pathlib import Path
from pycocotools.coco import COCO
from tqdm import tqdm
import os


class COCOStuffBuilder:
    def build(self, split, config):
        raise NotImplementedError

        if split not in ["train", "val"]:
            raise ValueError("Split " + split + " not supported.")

        # Load image ids
        annotations_path = Path(config["source"], "annotations",
                                "stuff_" + split + "2017.json")
        coco = COCO(annotations_path)
        cat_ids = coco.getCatIds(supNms=config["supercategories"])
        img_ids = coco.getImgIds(catIds=cat_ids)
        img_ids = img_ids[:len(img_ids) * config["size fraction"]]

        # Setup paths
        split_path = Path(config["destination"], split)
        image_path = Path(split_path, "images")
        if not os.path.exists(image_path):
            os.mkdir(image_path)

        target_path = Path(split_path, "targets")
        if not os.path.exists(target_path):
            os.mkdir(image_path)

        # Build the dataset
        print("Building " + split + " split...")
        for img_id in tqdm(img_ids):
            pass

    def _build_image(self, img_id, image_path):
        raise NotImplementedError