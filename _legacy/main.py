from lib.configure import configure
import argparse
import importlib  # noqa F401


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="path to config file")
    parser.add_argument("mode", help="Options: [build verify]")
    args = parser.parse_args()
    return args


def execute(mode):
    if mode in ["build", "verify"]:
        exec("importlib.import_module(\"lib.{mode}\").{mode}(config)".format(
            mode=mode))
    elif mode in ["histogram"]:
        importlib.import_module("lib.analyze").analyze(config, mode)
    else:
        raise NotImplementedError("Mode: " + mode + " not supported.")


if __name__ == "__main__":
    args = get_args()
    config = configure(args.config_file)
    execute(args.mode)
