from lib.analyzers.analyzer import Analyzer
from lib.builders.full_stuff_builder import TrainFullStuffBuilder, \
    ValFullStuffBuilder


class FullStuffAgent:
    def run(self, config):
        dataset = TrainFullStuffBuilder(config).build()
        analyzer = Analyzer(config)
        analyzer.verify(dataset)
        analyzer.compute_label_fraction_histogram(dataset)

        dataset = ValFullStuffBuilder(config).build()
        analyzer = Analyzer(config)
        analyzer.verify(dataset)