from lib.analyzers.analyzer import Analyzer
from lib.datasets.coco_stuff import COCOStuff
from pathlib import Path


class StuffAnalysisAgent:
    def run(self, config):
        dataset = COCOStuff(Path(config["destination root"], "stuff", "train"))
        analyzer = Analyzer(config)
        analyzer.compute_label_fraction_histograms(dataset)