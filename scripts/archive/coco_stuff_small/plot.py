import matplotlib.patches as mpatches
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

PALETTE = sns.diverging_palette(220, 10, n=7)
FONT_SCALE = 1.2
STYLE = "ticks"
CONTEXT = "poster"


def set_styles():
    sns.set(
        style=STYLE,
        context=CONTEXT,
        font_scale=FONT_SCALE,
        rc={
            "lines.linewidth": 5,
            'xtick.major.width': 5,
            'ytick.major.width': 5,
            'axes.linewidth': 5,
            'axes.labelweight': 2,
        })
    sns.set_palette(PALETTE)


def all_histogram():
    set_styles()
    with open("histogram.cache", 'rb') as fp:
        Xs = pickle.load(fp)

    # Plot entire dataset
    xs = []
    for i in Xs.keys():
        xs.extend(Xs[i])
    plotobj = sns.kdeplot(xs)

    # Plot Sky
    sky = []
    for i in [14, 94]:
        sky.extend(Xs[i])
    plotobj = sns.kdeplot(sky, shade=True)

    # Legend
    label_patches = []
    label_patch = mpatches.Patch(color=PALETTE[0], label="Full Dataset")
    label_patches.append(label_patch)
    label_patch = mpatches.Patch(color=PALETTE[1], label='"Sky" Class')
    label_patches.append(label_patch)
    plotobj.legend(handles=label_patches)

    # Aesthetics
    sns.despine()
    fig = plotobj.get_figure()
    fig.tight_layout()

    fig.savefig("hist.png")


def plot_histogram(xs, i):
    plotobj = sns.kdeplot(xs)
    # Aesthetics
    sns.despine()
    fig = plotobj.get_figure()
    fig.tight_layout()

    fig.savefig("hist-" + str(i) + "-.png")
    plt.clf()


def plot_all_histograms():
    set_styles()
    with open("histogram.cache", 'rb') as fp:
        Xs = pickle.load(fp)

    for i in Xs.keys():
        plot_histogram(Xs[i], i)


if __name__ == "__main__":
    plot_all_histograms()