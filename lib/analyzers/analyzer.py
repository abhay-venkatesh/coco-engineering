from PIL import Image
from pathlib import Path
from scipy.stats import norm
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import torchvision.transforms as transforms


class Analyzer:
    @classmethod
    def verify(self, config, dataset, max_examples=4):
        # Setup examples folder for the split
        split = Path(dataset.root).stem
        examples_path = Path(config["examples folder"], split)
        if not os.path.exists(examples_path):
            os.mkdir(examples_path)

        # Get img, target pairs
        for i, (img, target) in enumerate(dataset):
            img = transforms.ToPILImage()(img)
            img.save(Path(examples_path, "img-" + str(i) + ".jpg"))

            target = target.numpy()
            target = Image.fromarray(np.uint8(target * 100))
            target.save(Path(examples_path, "target-" + str(i) + ".jpg"))

            if i + 1 == max_examples:
                break

    @classmethod
    def compute_label_fraction_histogram(self, config, dataset):
        raise NotImplementedError
        
        xs = []
        cache_file = Path(config["examples folder"],
                          "label_frac_histogram.cache")
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fp:
                xs = pickle.load(fp)
        else:

            for _, target in tqdm(dataset):
                xs.append(
                    (target == 1).sum() / (target.shape[0] * target.shape[1]))

            with open(cache_file, 'wb') as fp:
                pickle.dump(xs, fp)

        # best fit of data
        (mu, sigma) = norm.fit(xs)

        # the histogram of the data
        n, bins, patches = plt.hist(
            xs, 50, density=1, facecolor='orange', alpha=0.75)

        # add a 'best fit' line
        y = norm.pdf(bins, mu, sigma)
        plt.plot(bins, y, 'r--', linewidth=2, color="firebrick")

        # plot
        plt.xlabel(r'fraction of pixels == 1')
        plt.ylabel(r'frequency')
        plt.title((r'$\mathrm{Fraction\ of\ Positive\ Labels\$ ' +
                   r'$Histogram:}\ \mu=%.3f,\ \sigma=%.3f$') % (mu, sigma))

        histogram_image_file = Path(config["examples folder"],
                                    "label_frac_histogram.png")
        plt.savefig(histogram_image_file)