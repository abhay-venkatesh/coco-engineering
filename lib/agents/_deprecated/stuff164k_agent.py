from lib.analyzers.analyzer import Analyzer
from lib.datasets.coco_stuff164k import COCOStuff164k


class Stuff164kAgent:
    def run(self, config):
        dataset = COCOStuff164k(config["source"])
        analyzer = Analyzer(config)
        analyzer.verify(dataset)
        analyzer.compute_label_fraction_histogram(dataset)

        dataset = COCOStuff164k(config["source"], "val")
        analyzer = Analyzer(config)
        analyzer.verify(dataset)
