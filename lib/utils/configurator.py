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
        if "size fraction" not in config.keys():
            config["size fraction"] = 0.1
        return config

    def _load(self, config_file):
        with open(config_file, 'r') as stream:
            try:
                config = yaml.load(stream, Loader=yaml.SafeLoader)
                config["name"] = Path(config_file).stem
                return config
            except yaml.YAMLError as exc:
                print(exc)

    def _build_paths(self, config):
        config["destination"] = Path(config["destination root"],
                                     config["name"])
        if not os.path.exists(config["destination"]):
            os.mkdir(config["destination"])

        config["examples folder"] = Path("examples", config["name"])
        if not os.path.exists(config["examples folder"]):
            os.mkdir(config["examples folder"])
        return config
