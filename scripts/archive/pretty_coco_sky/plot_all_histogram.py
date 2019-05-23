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
    plotobj = sns.kdeplot(xs, shade=True)
    plotobj.set_xlabel("Size of All Stuff classes")
    plotobj.set_ylabel("Frequency")

    # Aesthetics
    sns.despine()
    fig = plotobj.get_figure()
    fig.tight_layout()

    fig.savefig("hist_all.png")


if __name__ == "__main__":
    all_histogram()