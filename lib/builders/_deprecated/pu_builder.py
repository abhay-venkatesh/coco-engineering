from PIL import Image
from lib.builders.builder import Builder
from pathlib import Path
import numpy as np
from lib.utils.box_builder import BoxBuilder


class PUBuilder(Builder):
    def _build_target(self, cat_ids, img_id, target_dest_path):
        box_builder = BoxBuilder()

        target_exists = False
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        for ann in anns:
            if ann["category_id"] in cat_ids:
                mask = self.coco.annToMask(ann)
                bbox = box_builder.draw_pu_box(mask)
                bbox = np.array(bbox, dtype=np.uint8)

                if not target_exists:
                    target = bbox
                    target_exists = True
                else:
                    target += bbox

        target_name = self.coco.loadImgs(img_id)[0]['file_name'].replace(
            ".jpg", ".png")
        target = Image.fromarray(target)
        target = target.resize((self.img_width, self.img_height))
        target.save(Path(target_dest_path, target_name))


class PUValBuilder(PUBuilder):
    SPLIT = "val"


class PUTrainBuilder(PUBuilder):
    SPLIT = "train"