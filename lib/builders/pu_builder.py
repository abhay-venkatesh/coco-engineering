from PIL import Image
from lib.builders.builder import Builder
from pathlib import Path
import numpy as np


class PUBuilder(Builder):
    def _build_target(self, cat_ids, img_id, target_dest_path):
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        target_name = self.coco.loadImgs(img_id)[0]['file_name'].replace(
            ".jpg", ".png")
        target_exists = False
        for ann in anns:
            if ann["category_id"] in cat_ids:
                mask = self.coco.annToMask(ann)

                raise NotImplementedError
                bbox = self._draw_random_bbox_from_seg(img_id, mask)
                for i in range(1, self.box_type["num miniboxes"]):
                    bbox_ = bbox | self._draw_random_bbox_from_seg(
                        img_id, mask)
                    bbox = bbox_

                bbox = np.array(bbox, dtype=np.uint8)

                if not target_exists:
                    target = bbox
                    target_exists = True
                else:
                    target += bbox

        Image.fromarray(target).save(Path(target_dest_path, target_name))