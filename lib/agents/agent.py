from lib.builders.builder import TrainBuilder
from lib.analyzers.analyzer import Analyzer


class Agent:
    def run(self, config):
        builder = TrainBuilder(config)
        dataset = builder.build()
        Analyzer.verify(config, dataset)
