from lib.builder_base import BuilderBase


class Builder(BuilderBase):
    def __init__(self, config):
        self.config = config

    def build(self):
        raise NotImplementedError