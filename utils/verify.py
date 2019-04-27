import importlib


def verify(config):
    if config["dataset"] in ["coco"]:
        module_name = "datasets." + config["dataset"] + ".lib.analytics"
        importlib.import_module(module_name).verify_images(config)
    else:
        raise NotImplementedError