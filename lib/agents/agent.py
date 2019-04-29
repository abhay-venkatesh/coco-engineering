from lib.analyzers.analyzer import Analyzer
from lib.datasets.coco_stuff import COCOStuff
from pathlib import Path


class Agent:
    def run(self, config):
        dataset = COCOStuff(Path(config["destination"], "train"))
        Analyzer(config).compute_label_fraction_histogram(dataset)
