from PIL import Image
from pathlib import Path
import torchvision.transforms as transforms
import numpy as np


class Analyzer:
    @classmethod
    def verify(self, config, dataset):
        for i, (img, target) in enumerate(dataset[:10]):
            img = transforms.ToPILImage()(img)
            img.save(
                Path(config["examples folder"], "img-" + str(i) + ".jpg"))

            target = target.numpy()
            target = Image.fromarray(np.uint8(target * 100))
            target.save(
                Path(config["examples folder"], "target-" + str(i) + ".jpg"))
