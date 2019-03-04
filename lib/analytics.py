from PIL import Image
from datasets.coco_stuff.coco import get_coco_stuff_loaders
from pathlib import Path
from scipy.stats import norm
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import pickle
import random
import torchvision.transforms as transforms


def verify_coco_stuff_weak_bb(paths, config):
    data_root = Path(paths["filtered_data_root"], config["name"])
    train_loader, val_loader, _ = get_coco_stuff_loaders(
        data_root=data_root, batch_size=1)

    for image, labels in train_loader:
        img_tensor = image.squeeze(0)
        img = transforms.ToPILImage()(img_tensor)
        img.show()
        input("Press Enter to continue...")

        mask_array = labels[0].squeeze(0).numpy()
        mask_array *= 100
        mask = Image.fromarray(mask_array)
        mask.show()
        input("Press Enter to continue...")

        bbox_array = labels[1].squeeze(0).numpy()
        bbox_array *= 100
        bbox = Image.fromarray(bbox_array)
        bbox.show()
        input("Press Enter to continue...")
        break

    ri = random.randint(0, len(val_loader))
    for i, (image, labels) in enumerate(val_loader):
        if i == ri:
            img_tensor = image.squeeze(0)
            img = transforms.ToPILImage()(img_tensor)
            img.show()
            input("Press Enter to continue...")

            mask_array = labels[0].squeeze(0).numpy()
            mask_array *= 100
            mask = Image.fromarray(mask_array)
            mask.show()
            input("Press Enter to continue...")

            bbox_array = labels[1].squeeze(0).numpy()
            bbox_array *= 100
            bbox = Image.fromarray(bbox_array)
            bbox.show()
            input("Press Enter to continue...")
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
