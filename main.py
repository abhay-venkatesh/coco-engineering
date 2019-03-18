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
        if config["dataset"] != "coco":
            raise NotImplementedError

        analytics = importlib.import_module("lib." + config["dataset"] +
                                            ".analytics")
        analytics.verify_images(config)
    elif args.mode == "histogram":
        if config["dataset"] != "coco":
            raise NotImplementedError

        importlib.import_module("lib." + config["dataset"] +
                                ".analytics").Analyzer(
                                    config).compute_label_fraction_histogram()

    else:
        raise NotImplementedError("Mode: " + args.mode + " not supported.")
