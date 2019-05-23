import matplotlib.patches as mpatches
import pickle
import seaborn as sns

PALETTE = sns.diverging_palette(10, 220, n=7)
FONT_SCALE = 2
STYLE = "ticks"
CONTEXT = "paper"


def set_styles():
    sns.set(
        style=STYLE,
        context=CONTEXT,
        font_scale=FONT_SCALE,
    )
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


def plot_sky():
    set_styles()
    with open("sky_histogram.cache", 'rb') as fp:
        xs = pickle.load(fp)

    plotobj = sns.kdeplot(xs, shade=True)
    plotobj.set_xlabel("Size of Sky")
    plotobj.set_ylabel("Frequency")

    # Aesthetics
    sns.despine()
    fig = plotobj.get_figure()
    fig.tight_layout()

    fig.savefig("hist_sky.png")


if __name__ == "__main__":
    plot_sky()