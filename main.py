from lib.paths import get_paths
import argparse
import yaml
import importlib


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="path to config file")
    parser.add_argument("mode", help="Options: [build verify]")
    args = parser.parse_args()
    return args


def get_config(config_file_path):
    with open(config_file_path, 'r') as stream:
        try:
            return yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)


if __name__ == "__main__":
    paths = get_paths()
    args = get_args()
    config = get_config(args.config_file)
    if args.mode == "build":
        importlib.import_module("lib.builder").Builder(paths, config).build()
    elif args.mode == "verify":
        importlib.import_module("lib.analytics").verify_coco_stuff_weak_bb(
            paths, config)
