from PIL import Image
from lib.builders.builder import Builder
from pathlib import Path
from tqdm import tqdm
import numpy as np
import os


class StuffBuilder(Builder):
    def build(self):
        # Load image ids
        cat_ids = self.coco.getCatIds(supNms=self.config["supercategories"])
        img_ids = self.coco.getImgIds(catIds=[])
        length = round(len(img_ids) * self.config["size fraction"])
        img_ids = img_ids[:length]

        # Build paths
        img_src_path = Path(self.config["source"], "images",
                            self.SPLIT + "2017")
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
            img_name = self.coco.loadImgs(img_id)[0]['file_name']
            img = Image.open(Path(img_src_path, img_name))
            img = img.resize((self.IMG_WIDTH, self.IMG_HEIGHT))
            img.save(Path(image_dest_path, img_name))

            # Save target
            self._build_target(cat_ids, img_id, target_dest_path)

        return self._get_dataset()

    def _build_target(self, cat_ids, img_id, target_dest_path):
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        target_name = self.coco.loadImgs(img_id)[0]['file_name'].replace(
            ".jpg", ".png")
        target_exists = False
        for ann in anns:
            if ann["category_id"] in cat_ids:
                category_id = ann["category_id"]
                if category_id == 183:
                    category_id = 0

                mask = self.coco.annToMask(ann)

                if not target_exists:
                    target = np.zeros_like(mask)
                    target_exists = True

                target[mask == 1] = category_id

        if not target_exists:
            target = np.zeros((self.IMG_WIDTH, self.IMG_HEIGHT))

        target = Image.fromarray(target)
        target = target.convert("L")
        target = target.resize((self.IMG_WIDTH, self.IMG_HEIGHT))
        target.save(Path(target_dest_path, target_name))


class ValStuffBuilder(StuffBuilder):
    SPLIT = "val"


class TrainStuffBuilder(StuffBuilder):
    SPLIT = "train"