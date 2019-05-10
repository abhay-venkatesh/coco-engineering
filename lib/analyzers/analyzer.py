from PIL import Image
from collections import defaultdict
from pathlib import Path
from scipy.stats import norm
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import torchvision.transforms as transforms


class Analyzer:
    def __init__(self, config):
        self.config = config

    def verify(self, dataset, max_examples=4):
        split = Path(dataset.root).stem
        examples_path = Path(self.config["stats folder"], split, "examples")
        if not os.path.exists(examples_path):
            os.makedirs(examples_path)

        # Get img, target pairs
        for i, (img, target) in enumerate(dataset):
            img = transforms.ToPILImage()(img)
            img.save(Path(examples_path, "img-" + str(i) + ".jpg"))

            target = target.numpy()
            target = Image.fromarray(np.uint8(target * 100))
            target.save(Path(examples_path, "target-" + str(i) + ".jpg"))

            if i + 1 == max_examples:
                break

    def compute_label_fraction_histogram(self, dataset):
        split = Path(dataset.root).stem
        histogram_folder = Path(self.config["stats folder"], split,
                                "histogram")
        if not os.path.exists(histogram_folder):
            os.makedirs(histogram_folder)

        xs = []
        cache_file = Path(histogram_folder, "label_frac_histogram.cache")
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fp:
                xs = pickle.load(fp)
        else:

            for _, target in tqdm(dataset):
                target = target.numpy()
                xs.append(
                    (target == 1).sum() / (target.shape[0] * target.shape[1]))

            with open(cache_file, 'wb') as fp:
                pickle.dump(xs, fp)

        self._write_histogram(xs, histogram_folder, 0)

    def _write_histogram(self, xs, histogram_folder, i):
        # best fit of data
        (mu, sigma) = norm.fit(xs)

        # the histogram of the data
        n, bins, patches = plt.hist(
            xs, 50, density=1, facecolor='black', alpha=0.75)

        # add a 'best fit' line
        y = norm.pdf(bins, mu, sigma)
        plt.plot(bins, y, 'r--', linewidth=2, color="coral")

        # plot
        plt.xlabel(r'fraction of pixels == 1')
        plt.ylabel(r'frequency')
        plt.title((
            r'$\mathrm{Fraction\ of\ Positive\ Labels\ Histogram:}\ \mu=%.3f,\ \sigma=%.3f$'
        ) % (mu, sigma))

        histogram_image_file = Path(histogram_folder,
                                    "label_frac_histogram_" + str(i) + ".png")
        plt.savefig(histogram_image_file)
        plt.close()

    def compute_label_fraction_histograms(self, dataset):
        if len(dataset) == 0:
            raise ValueError("Dataset is empty.")

        split = Path(dataset.root).stem
        histogram_folder = Path(self.config["stats folder"], split,
                                "histogram")
        if not os.path.exists(histogram_folder):
            os.makedirs(histogram_folder)

        Xs = defaultdict(list)
        cache_file = Path(histogram_folder, "label_frac_histogram.cache")
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fp:
                Xs = pickle.load(fp)
        else:

            for _, target in tqdm(dataset):
                target = target.numpy()
                for i in np.unique(target):
                    ratio = ((target == i).sum() /
                             (target.shape[0] * target.shape[1]))

                    # Only add to histogram if the target class is present
                    if ratio > 0:
                        Xs[i].append(ratio)

            with open(cache_file, 'wb') as fp:
                pickle.dump(Xs, fp)

        for i in Xs.keys():
            self._write_histogram(Xs[i], histogram_folder, i)
