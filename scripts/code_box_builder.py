from lib.utils.box_builder import BoxBuilder
from pathlib import Path
from PIL import Image
import numpy as np
from lib.datasets.coco_stuff import COCOStuff

if __name__ == "__main__":
    dataset = COCOStuff(Path("D:/code/data/filtered_datasets/coco_sky/train"))
    for img, target in dataset:
        bbox = BoxBuilder.draw_pu_box(target)
