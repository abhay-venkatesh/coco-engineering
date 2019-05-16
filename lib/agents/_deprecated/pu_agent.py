from lib.builders.pu_builder import PUTrainBuilder, PUValBuilder
from lib.analyzers.analyzer import Analyzer


class PUAgent:
    def run(self, config):
        dataset = PUTrainBuilder(config).build()
        analyzer = Analyzer(config)
        analyzer.verify(dataset)
        analyzer.compute_label_fraction_histogram(dataset)

        dataset = PUValBuilder(config).build()
        analyzer = Analyzer(config)
        analyzer.verify(dataset)
