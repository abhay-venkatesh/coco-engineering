from lib.configure import get_config
import argparse
import importlib  # noqa F401


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="path to config file")
    parser.add_argument("mode", help="Options: [build verify]")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    config = get_config(args.config_file)
    if args.mode in ["build", "verify", "histogram"]:
        exec("importlib.import_module(\"lib.{mode}\").{mode}(config)".format(
            mode=args.mode))
    else:
        raise NotImplementedError("Mode: " + args.mode + " not supported.")
