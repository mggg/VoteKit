from votekit.plots import multi_bar_plot
from matplotlib.axes import Axes


data = {
    "Profile 1": {"Chris": 5, "Peter": 6, "Moon": 7},
    "Profile 2": {"Chris": 4, "Peter": 3, "Moon": 2},
}


def test_barplot_with_defaults():
    ax = multi_bar_plot(data)

    assert isinstance(ax, Axes)
