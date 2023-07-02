"""
Note:
  lifetime_l = numpy.sort(lifetime_l)
  print("len(lifetime_l)= {}".format(len(lifetime_l) ) )
  # plot.hist(lifetime_l, bins=100, histtype='step', normed=True, lw=2)
  x_l = lifetime_l[::-1]
  y_l = numpy.arange(lifetime_l.size)/lifetime_l.size
"""

import colorsys
import itertools
import matplotlib.colors as matplotlib_colors

# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42
# matplotlib.use('Agg')
import matplotlib.pyplot as plot

from src.utils.debug import *


plot.rcParams.update({"text.usetex": True})

NICE_BLUE = "#66b3ff"
NICE_RED = "#ff9999"
NICE_GREEN = "#99ff99"
NICE_ORANGE = "#ffcc99"

nice_color_cycle = itertools.cycle((NICE_BLUE, NICE_RED, NICE_ORANGE, NICE_GREEN))
dark_color_cycle = itertools.cycle(
    (
        "green",
        "purple",
        "blue",
        "magenta",
        "gray",
        "brown",
        "turquoise",
        "gold",
        "olive",
        "silver",
        "rosybrown",
        "plum",
        "goldenrod",
        "lightsteelblue",
        "lightpink",
        "orange",
        "darkgray",
        "orangered",
    )
)

# dark_color_cycle = itertools.cycle(('magenta', 'purple', 'gray', 'brown', 'turquoise', 'gold', 'olive', 'silver', 'rosybrown', 'plum', 'goldenrod', 'lightsteelblue', 'lightpink', 'orange', 'darkgray', 'orangered'))

light_color_cycle = itertools.cycle(
    (
        "silver",
        "rosybrown",
        "plum",
        "lightsteelblue",
        "lightpink",
        "orange",
        "turquoise",
    )
)

linestyle_cycle = itertools.cycle(("-", "--", ":", "-."))

marker_cycle = itertools.cycle(
    ("o", "v", "^", "p", "d", "<", ">", "h", "H", "*", "s", "1", "2", "3", "4")
)

skinny_marker_l = ["x", "+", "1", "2", "3", "4"]

mew, ms = 1, 2  # 3, 5


def prettify(ax):
    # plot.tick_params(top='off', right='off', which='both')
    plot.tick_params(top=False, right=False, which="both")
    ax.patch.set_alpha(0.2)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)


def plot_points(x_y_l, file_name):
    x_l, y_l = [], []
    for x_y in x_y_l:
        x_l.append(x_y[0])
        y_l.append(x_y[1])

    plot.plot(x_l, y_l, color=NICE_BLUE, marker="o", ls="None")
    plot.savefig("plots/{}.png".format(file_name), bbox_inches="tight")
    plot.gcf().clear()
    log(INFO, "done.")


def lighten_color(color, amount=0.5):
    """Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)

    Ref: https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib
    """

    try:
        c = matplotlib_colors.cnames[color]
    except:
        c = color

    c = colorsys.rgb_to_hls(*matplotlib_colors.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
