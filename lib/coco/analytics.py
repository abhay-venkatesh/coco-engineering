from PIL import Image
from lib.coco.coco import get_coco_stuff_loaders
from lib.coco.paths import get_paths
from pathlib import Path
from scipy.stats import norm
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import random
import shutil
import torchvision.transforms as transforms


def verify_images(config):
    paths = get_paths()
    data_root = Path(paths["filtered_data_root"], config["name"])

    if not os.path.exists(Path("examples")):
        os.mkdir(Path("examples"))

    example_path = Path("examples", config["name"])
    if os.path.exists(example_path):
        shutil.rmtree(example_path)
    os.makedirs(example_path)

    train_loader, val_loader, _ = get_coco_stuff_loaders(
        data_root=data_root, batch_size=1)

    ri = random.randint(0, len(train_loader))
    for i, (image, labels) in enumerate(train_loader):
        if i == ri:
            img_tensor = image.squeeze(0)
            img = transforms.ToPILImage()(img_tensor)
            img.save(Path(example_path, "img-" + str(i) + ".png"))

            mask_array = labels[0].squeeze(0).numpy()
            mask = Image.fromarray(np.uint8(mask_array * 100), 'L')
            mask.save(Path(example_path, "mask-" + str(i) + ".png"))

            bbox_array = labels[1].squeeze(0).numpy()
            bbox = Image.fromarray(np.uint8(bbox_array * 100), 'L')
            bbox.save(Path(example_path, "bbox-" + str(i) + ".png"))
            break

    ri = random.randint(0, len(val_loader))
    for i, (image, labels) in enumerate(val_loader):
        if i == ri:
            img_tensor = image.squeeze(0)
            img = transforms.ToPILImage()(img_tensor)
            img.save(Path(example_path, "img-" + str(i) + ".png"))

            mask_array = labels[0].squeeze(0).numpy()
            mask = Image.fromarray(np.uint8(mask_array * 100), 'L')
            mask.save(Path(example_path, "mask-" + str(i) + ".png"))

            bbox_array = labels[1].squeeze(0).numpy()
            bbox = Image.fromarray(np.uint8(bbox_array * 100), 'L')
            bbox.save(Path(example_path, "bbox-" + str(i) + ".png"))
            break


def compute_noise_histogram(paths, config):
    """ Want to compute:
        P(B=0|Y=1) = (B[Y == 1] == 0).sum() / (Y == 1).sum()

    """
    if not os.path.exists(Path("cache")):
        os.mkdir(Path("cache"))

    xs = []
    cache_file = Path("cache", config["name"] + "_noise_histogram")
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as fp:
            xs = pickle.load(fp)
    else:
        data_root = Path(paths["filtered_data_root"], config["name"])
        train_loader, _, _ = get_coco_stuff_loaders(
            data_root=data_root, batch_size=1)

        for _, labels in tqdm(train_loader):
            Y = labels[0].squeeze(0).numpy()
            B = labels[1].squeeze(0).numpy()
            xs.append((B[Y == 1] == 0).sum() / (Y == 1).sum())

        with open(cache_file, 'wb') as fp:
            pickle.dump(xs, fp)

    # best fit of data
    (mu, sigma) = norm.fit(xs)

    # the histogram of the data
    n, bins, patches = plt.hist(
        xs, 60, density=1, facecolor='orange', alpha=0.75)

    # add a 'best fit' line
    y = norm.pdf(bins, mu, sigma)
    plt.plot(bins, y, 'r--', linewidth=2, color="firebrick")

    # plot
    plt.xlabel(r'$\rho$')
    plt.ylabel(r'$\rho$ frequency')
    plt.title(r'$\mathrm{Histogram\ of\ \rho:}\ \mu=%.3f,\ \sigma=%.3f$' %
              (mu, sigma))

    if not os.path.exists(Path("stats")):
        os.mkdir(Path("stats"))

    histogram_image_file = Path("stats",
                                config["name"] + "_noise_histogram.png")
    plt.savefig(histogram_image_file)


class Analyzer:
    def __init__(self, config):
        self.paths = get_paths()
        self.config = config

    def compute_label_fraction_histogram(self):
        if not os.path.exists(Path("cache")):
            os.mkdir(Path("cache"))

        xs = []
        cache_file = Path("cache",
                          self.config["name"] + "_label_frac_histogram")
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fp:
                xs = pickle.load(fp)
        else:
            data_root = Path(self.paths["filtered_data_root"],
                             self.config["name"])
            train_loader, _, _ = get_coco_stuff_loaders(
                data_root=data_root, batch_size=1)

            for _, labels in tqdm(train_loader):
                B = labels[1].squeeze(0).numpy()
                xs.append((B == 1).sum() / (B.shape[0] * B.shape[1]))

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
        plt.title(
            r'$\mathrm{Fraction\ of\ Positive\ Labels\ Histogram:}\ \mu=%.3f,\ \sigma=%.3f$'
            % (mu, sigma))

        if not os.path.exists(Path("stats")):
            os.mkdir(Path("stats"))

        histogram_image_file = Path(
            "stats", self.config["name"] + "_label_frac_histogram.png")
        plt.savefig(histogram_image_file)


def compute_supervision_percentage(paths, config):
    """ Supervision Percentage = (B == 1).sum() / (Y == 1).sum()

    Compute in rolling fashion:

    while Y, B = next(train_loader):
        def rolling(Y, B, cum, Y_):
            cent = (B == 1).sum() / (Y == 1).sum()
            cum *= Y_
            cum += (B == 1).sum()  
            cum /= (Y == 1).sum()
    """
    if not os.path.exists(Path("cache")):
        os.mkdir(Path("cache"))

    supervision_percentage = 0.0
    cache_file = Path("cache", config["name"] + "_supervision_percentage")
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as fp:
            supervision_percentage = pickle.load(fp)
    else:
        data_root = Path(paths["filtered_data_root"], config["name"])
        train_loader, _, _ = get_coco_stuff_loaders(
            data_root=data_root, batch_size=1)

        ysum = 0
        bsum = 0
        for _, labels in tqdm(train_loader):
            ysum += (labels[0].squeeze(0).numpy() == 1).sum()
            bsum += (labels[1].squeeze(0).numpy() == 1).sum()

        supervision_percentage = bsum / ysum
        with open(cache_file, 'wb') as fp:
            pickle.dump(supervision_percentage, fp)

    print(supervision_percentage)
