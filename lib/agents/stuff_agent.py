from lib.analyzers.analyzer import Analyzer
from lib.builders.stuff_builder import TrainStuffBuilder, ValStuffBuilder


class StuffAgent:
    def run(self, config):
        dataset = TrainStuffBuilder(config).build()
        analyzer = Analyzer(config)
        analyzer.verify(dataset)

        dataset = ValStuffBuilder(config).build()
        analyzer = Analyzer(config)
        analyzer.verify(dataset)
