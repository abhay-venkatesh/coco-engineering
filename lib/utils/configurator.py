from pathlib import Path
import os
import yaml


class Configurator:
    def configure(self, config_file):
        config = self._load(config_file)
        config = self._set_defaults(config)
        config = self._build_paths(config)
        return config

    def _set_defaults(self, config):
        if "size" not in config.keys():
            config["size"] = 640
        return config

    def _load(self, config_file):
        with open(config_file, 'r') as stream:
            try:
                config = yaml.load(stream)
                config["name"] = Path(config_file).stem
                return config
            except yaml.YAMLError as exc:
                print(exc)

    def _build_paths(self, config):
        config["dataset path"] = Path(config["destination"], config["name"])
        if os.path.exists(config["dataset path"]):
            raise RuntimeError("Dataset already exists. ")
        else:
            os.mkdir(config["dataset path"])
        return config
