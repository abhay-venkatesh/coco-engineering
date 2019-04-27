from lib.builders.builder import TrainBuilder


class Agent:
    def run(self, config):
        builder = TrainBuilder(config)
        builder.build()
