import argparse
from lib.utils.configurator import Configurator
import importlib
import inflection

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="path to config file")
    args = parser.parse_args()

    config = Configurator().configure(args.config_file)

    agent_module = importlib.import_module(("lib.agents.{}").format(
        inflection.underscore(config["agent"])))
    Agent = getattr(agent_module, config["agent"])
    Agent(config).run()
