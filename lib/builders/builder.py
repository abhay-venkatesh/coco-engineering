from PIL import Image
from lib.datasets.coco_stuff import COCOStuff
from pathlib import Path
from pycocotools.coco import COCO
from tqdm import tqdm
import os


class Builder:
    IMG_HEIGHT = 426
    IMG_WIDTH = 640

    def __init__(self, config):
        self.config = config
        annotations_path = Path(self.config["source"], "annotations",
                                "stuff_" + self.SPLIT + "2017.json")
        self.coco = COCO(annotations_path)

    def build(self):
        # Load image ids
        cat_ids = self.coco.getCatIds(supNms=self.config["supercategories"])
        img_ids = self.coco.getImgIds(catIds=cat_ids)
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

    def _get_dataset(self):
        return COCOStuff(Path(self.config["destination"], self.SPLIT))

    def _build_target(self, cat_ids, img_id, target_dest_path):
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        target_name = self.coco.loadImgs(img_id)[0]['file_name'].replace(
            ".jpg", ".png")
        target_exists = False
        for ann in anns:
            if ann["category_id"] in cat_ids:
                mask = self.coco.annToMask(ann)

                if not target_exists:
                    target = mask
                    target_exists = True
                else:
                    target += mask

        target = Image.fromarray(target)
        target = target.resize((self.IMG_WIDTH, self.IMG_HEIGHT))
        target.save(Path(target_dest_path, target_name))


class ValBuilder(Builder):
    SPLIT = "val"


class TrainBuilder(Builder):
    SPLIT = "train"