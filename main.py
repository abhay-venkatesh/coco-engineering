from lib.builder import Builder
from lib.paths import get_paths
import yaml


def get_config(config_file_path):
    with open(config_file_path, 'r') as stream:
        try:
            return yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)


if __name__ == "__main__":
    paths = get_paths()
    config = get_config("./configs/full_bb.yml")
    Builder(paths, config).build()
