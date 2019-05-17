from lib.analyzers.analyzer import Analyzer
from lib.builders.single_stuff_builder import TrainSingleStuffBuilder, \
    ValSingleStuffBuilder


class SingleStuffAgent:
    def run(self, config):
        dataset = TrainSingleStuffBuilder(config).build()
        analyzer = Analyzer(config)
        analyzer.verify(dataset)
        analyzer.compute_label_fraction_histogram(dataset)

        dataset = ValSingleStuffBuilder(config).build()
        analyzer = Analyzer(config)
        analyzer.verify(dataset)
