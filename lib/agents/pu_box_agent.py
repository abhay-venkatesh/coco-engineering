from lib.builders.pu_box_builder import PUBoxBuilder


class PUBoxAgent:
    def run(self, config):
        builder = PUBoxBuilder()
        for split in ["train", "val"]:
            builder.build(split, config)

        raise NotImplementedError
