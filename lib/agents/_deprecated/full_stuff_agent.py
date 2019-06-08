from lib.builders.full_stuff_builder import TrainFullStuffBuilder, \
    ValFullStuffBuilder


class FullStuffAgent:
    def run(self, config):
        TrainFullStuffBuilder(config).build()
        ValFullStuffBuilder(config).build()
