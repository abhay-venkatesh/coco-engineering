from lib.builders.builder import Builder


class Stuff164KBuilder(Builder):
    def __init__(self, config):
        self.config = config

    def build(self):
        raise NotImplementedError

    def _build_target(self, cat_id_map, img_id, target_dest_path):
        raise NotImplementedError


class ValStuff164KBuilder(Stuff164KBuilder):
    SPLIT = "val"


class TrainStuff164KBuilder(Stuff164KBuilder):
    SPLIT = "train"