import yaml


class Configurator:
    def configure(self, config_file):
        with open(config_file, 'r') as stream:
            try:
                return self._set_defaults(yaml.load(stream))
            except yaml.YAMLError as exc:
                print(exc)

    def _set_defaults(self, config):
        if "dataset" not in config.keys():
            config["dataset"] = "coco"
        return config