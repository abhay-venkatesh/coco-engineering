import matplotlib.patches as mpatches
import os
import pickle
import seaborn as sns

PALETTE = sns.diverging_palette(220, 10, n=7)
FONT_SCALE = 2
STYLE = "ticks"
CONTEXT = "poster"


def set_styles():
    sns.set_context(CONTEXT)
    sns.set(style=STYLE, font_scale=FONT_SCALE)
    sns.set_palette(PALETTE)


def write_histogram(xs):
    plotobj = sns.kdeplot(xs, shade=True)
    plotobj.axis("off")
    # sns.despine()
    fig = plotobj.get_figure()
    fig.tight_layout()
    fig.savefig("hist.png")


def sky_histogram():
    xs = []
    cache_file = "sky_histogram.cache"
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as fp:
            xs = pickle.load(fp)

    write_histogram(xs)


def all_histogram():
    with open("histogram.cache", 'rb') as fp:
        Xs = pickle.load(fp)

    sns.set_context("poster")
    # Plot entire dataset
    xs = []
    for i in Xs.keys():
        xs.extend(Xs[i])
    plotobj = sns.kdeplot(xs)

    # Plot Sky
    sky = []
    for i in [14, 92]:
        sky.extend(Xs[i])
    plotobj = sns.kdeplot(sky, shade=True)

    # Legend
    label_patches = []
    label_patch = mpatches.Patch(color=PALETTE[0], label="Full Dataset")
    label_patches.append(label_patch)
    label_patch = mpatches.Patch(color=PALETTE[1], label='"Sky" Class')
    label_patches.append(label_patch)
    plotobj.legend(handles=label_patches)

    sns.despine()
    fig = plotobj.get_figure()
    fig.tight_layout()
    fig.savefig("hist.png")


if __name__ == "__main__":
    set_styles()
    all_histogram()