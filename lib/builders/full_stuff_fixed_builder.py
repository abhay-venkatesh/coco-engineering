from PIL import Image
from lib.builders.builder import Builder
from pathlib import Path
from pycocotools.coco import COCO
from tqdm import tqdm
import numpy as np
import os


class FullStuffFixedBuilder(Builder):
    def __init__(self, config):
        self.config = config
        annotations_path = Path(
            self.config["source"], "annotations", "stuff_" + self.SPLIT + "2017.json"
        )
        self.coco = COCO(annotations_path)

    def _build_target(self, h, w, cat_id_map, img_id, target_dest_path):
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        target_name = self.coco.loadImgs(img_id)[0]["file_name"].replace(".jpg", ".png")
        target_exists = False
        for ann in anns:
            if ann["category_id"] in cat_id_map.keys():
                mask = self.coco.annToMask(ann)

                if not target_exists:
                    target = np.zeros_like(mask)
                    target_exists = True

                target[mask == 1] = cat_id_map[ann["category_id"]]

        if not target_exists:
            target = np.zeros((h, w))

        target = Image.fromarray(target)
        target = target.convert("L")
        target.save(Path(target_dest_path, target_name))


class ValFullStuffFixedBuilder(FullStuffFixedBuilder):
    SPLIT = "val"


class TrainFullStuffFixedBuilder(FullStuffFixedBuilder):
    SPLIT = "train"


class COCOStuff2018Builder(FullStuffFixedBuilder):
    def build(self):
        # Load image ids
        cat_ids = self.coco.getCatIds(supNms=self.config["supercategories"])
        img_ids = self.coco.getImgIds(catIds=[])
        length = round(len(img_ids) * self.config["size fraction"])
        img_ids = img_ids[:length]

        # Get category tuples
        cat_ids = self.coco.getCatIds(supNms=[])
        cat_id_tuples = list(enumerate(cat_ids))

        # Build map for categories to make them usable in a neural network
        # Ref: http://cocodataset.org/#stuff-eval
        # map ids 92-182 to 1-91, and map 183 to 0
        cat_id_map = dict((y, x + 1) for x, y in cat_id_tuples)
        cat_id_map[183] = 0

        # Build paths
        img_src_path = Path(self.config["source"], "images", self.SPLIT + "2017")
        split_path = Path(self.config["destination"], self.SPLIT)
        image_dest_path = Path(split_path, "images")
        target_dest_path = Path(split_path, "targets")
        for path in [split_path, image_dest_path, target_dest_path]:
            if not os.path.exists(path):
                os.mkdir(path)

        # Build the dataset
        print("Building " + self.SPLIT + " split...")
        for img_id in tqdm(img_ids):
            # Save image
            img_name = self.coco.loadImgs(img_id)[0]["file_name"]
            img = Image.open(Path(img_src_path, img_name))
            h, w = np.array(img).shape[0], np.array(img).shape[1]
            img.save(Path(image_dest_path, img_name))

            # Save target
            self._build_target(h, w, cat_id_map, img_id, target_dest_path)

        return self._get_dataset()


class ValCOCOStuff2018Builder(COCOStuff2018Builder):
    SPLIT = "val"


class TrainCOCOStuff2018Builder(COCOStuff2018Builder):
    SPLIT = "train"
