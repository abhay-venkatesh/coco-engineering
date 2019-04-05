import importlib


def build(config):
    if config["dataset"] in ["coco", "mnist"]:
        module_name = "datasets." + config["dataset"] + ".lib.builder"
        importlib.import_module(module_name).Builder(config).build()
    else:
        raise NotImplementedError