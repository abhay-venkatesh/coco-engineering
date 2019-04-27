from lib.builders.pu_builder import PUTrainBuilder
from lib.analyzers.analyzer import Analyzer


class PUAgent:
    def run(self, config):
        builder = PUTrainBuilder(config)
        dataset = builder.build()
        Analyzer.verify(config, dataset)
