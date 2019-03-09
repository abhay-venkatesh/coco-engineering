import yaml


def get_config(config_file_path):
    with open(config_file_path, 'r') as stream:
        try:
            return set_defaults(yaml.load(stream))
        except yaml.YAMLError as exc:
            print(exc)


def set_defaults(config):
    if "dataset" not in config.keys():
        config["dataset"] = "coco"
    return config
