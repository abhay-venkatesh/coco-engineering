from lib.builders.builder import TrainBuilder, ValBuilder
from lib.analyzers.analyzer import Analyzer


class Agent:
    def run(self, config):
        dataset = TrainBuilder(config).build()
        Analyzer.compute_label_fraction_histogram(config, dataset)

        # dataset = ValBuilder(config).build()
        # Analyzer.verify(config, dataset)
