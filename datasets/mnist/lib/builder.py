from pathlib import Path
from torchvision.datasets import MNIST
import numpy as np
import platform


def get_path_based_on_machine():
    if platform.node() == "Anton":
        return Path("D:/code/data/filtered_datasets")
    else:
        raise ValueError("Platform not supported")


class Builder:
    def __init__(self, config):
        self.root = get_path_based_on_machine()

        self.trainset = MNIST(self.root, train=True, download=True)
        self.testset = MNIST(self.root, train=False)

    def build(self):
        for [img, target] in self.trainset:
            target = target.numpy()  # Convert from PyTorch Tensor
            img_ = np.zeros(256, 256)
            
            raise RuntimeError
