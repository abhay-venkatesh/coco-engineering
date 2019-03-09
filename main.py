from lib.configure import get_config
import argparse
import importlib


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="path to config file")
    parser.add_argument("mode", help="Options: [build verify]")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    config = get_config(args.config_file)
    if args.mode == "build":
        importlib.import_module("lib." + config["dataset"] +
                                ".builder").Builder(config).build()
    elif args.mode == "verify":
        raise NotImplementedError