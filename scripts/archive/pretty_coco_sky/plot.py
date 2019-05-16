from pathlib import Path
from scipy.stats import norm
import matplotlib.pyplot as plt
import os
import pickle


def write_histogram(xs):
    # best fit of data
    (mu, sigma) = norm.fit(xs)

    # the histogram of the data
    n, bins, patches = plt.hist(
        xs, 50, density=1, facecolor='black', alpha=0.75)

    # add a 'best fit' line
    y = norm.pdf(bins, mu, sigma)
    plt.plot(bins, y, 'r--', linewidth=2, color="#b2fef7")

    # Color
    for bin_, patch in zip(bins, patches):
        if bin_ >= 0.6:
            patch.set_facecolor("#4f9a94")
        else:
            patch.set_facecolor("black")

    histogram_image_file = Path("label_frac_histogram.png")
    plt.axis("off")
    plt.savefig(histogram_image_file)
    plt.close()


if __name__ == "__main__":
    xs = []
    cache_file = "label_frac_histogram.cache"
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as fp:
            xs = pickle.load(fp)

    write_histogram(xs)