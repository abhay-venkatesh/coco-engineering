from lib.analyzers.analyzer import Analyzer
from lib.builders.builder import TrainBuilder, ValBuilder


class Agent:
    def run(self, config):
        dataset = TrainBuilder(config).build()
        analyzer = Analyzer(config)
        analyzer.verify(dataset)
        analyzer.compute_label_fraction_histogram(dataset)

        dataset = ValBuilder(config).build()
        analyzer = Analyzer(config)
        analyzer.verify(dataset)
