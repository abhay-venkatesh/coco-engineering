import os
import pickle
import seaborn as sns

PALETTE = sns.diverging_palette(220, 10, n=7)
FONT_SCALE = 1.5
STYLE = "ticks"
CONTEXT = "paper"


def set_styles():
    sns.set_context(CONTEXT)
    sns.set(style=STYLE, font_scale=FONT_SCALE)
    sns.set_palette(PALETTE)


def write_histogram(xs):
    set_styles()
    plotobj = sns.kdeplot(xs, shade=True)
    plotobj.axis("off")
    # sns.despine()
    fig = plotobj.get_figure()
    fig.tight_layout()
    fig.savefig("hist.png")


if __name__ == "__main__":
    xs = []
    cache_file = "label_frac_histogram.cache"
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as fp:
            xs = pickle.load(fp)

    write_histogram(xs)