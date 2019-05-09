from pycocotools.coco import COCO
from pathlib import Path
from PIL import Image
import numpy as np


class StuffAgent:
    def run(self, config):
        """
        train_ann_file = Path(config["source"], "annotations",
                              "stuff_train2017.json")
        coco = COCO(train_ann_file)
        print(coco.cats)
        """
        train_ann_folder = Path(config["source"], "annotations", "train2017")
        img = Image.open(Path(train_ann_folder, "000000000036.png"))
        img_array = np.array(img)
        print(np.unique(img_array))
        
