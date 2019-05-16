from lib.datasets.coco_stuff import COCOStuff
from pathlib import Path


class Builder:
    def _get_dataset(self):
        return COCOStuff(Path(self.config["destination"], self.SPLIT))


class ValBuilder(Builder):
    SPLIT = "val"


class TrainBuilder(Builder):
    SPLIT = "train"