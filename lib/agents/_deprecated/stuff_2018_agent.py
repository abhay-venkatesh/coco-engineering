from lib.builders.full_stuff_builder import TrainCOCOStuff2018Builder, \
    ValCOCOStuff2018Builder


class FullStuffAgent:
    def run(self, config):
        TrainCOCOStuff2018Builder(config).build()
        ValCOCOStuff2018Builder(config).build()
