from votekit.plots.bar_plot import _set_default_bar_plot_args
from votekit.utils import COLOR_LIST
from matplotlib.axes import Axes
import matplotlib.pyplot as plt

data = {
    "Profile 1": {"Chris": 5, "Peter": 6, "Moon": 7},
    "Profile 2": {"Chris": 4, "Peter": 3, "Moon": 2},
}


def test_set_defaults_none_provided():

    default_dict = _set_default_bar_plot_args(
        data=data,
        data_set_colors=None,
        bar_width=None,
        category_ordering=None,
        legend_font_size=None,
        threshold_values=None,
        threshold_kwds=None,
        ax=None,
    )

    assert default_dict["data_set_to_color"] == {
        "Profile 1": COLOR_LIST[0],
        "Profile 2": COLOR_LIST[1],
    }
    assert default_dict["bar_width"] == 0.7 / 2
    assert default_dict["category_ordering"] == ["Chris", "Peter", "Moon"]
    assert default_dict["font_size"] == 13
    assert default_dict["legend_font_size"] == 13
    assert not default_dict["threshold_values"]
    assert not default_dict["threshold_kwds"]
    assert isinstance(default_dict["ax"], Axes)


def test_set_defaults_values_provided():
    fig, ax = plt.subplots()
    default_dict = _set_default_bar_plot_args(
        data=data,
        data_set_colors={"Profile 2": "green"},
        bar_width=1 / 2,
        category_ordering=["Peter", "Moon", "Chris"],
        legend_font_size=20,
        threshold_values=[5, 4],
        threshold_kwds=[{"linestyle": "--"}, {"linewidth": 4}],
        ax=ax,
    )

    assert default_dict["data_set_to_color"] == {
        "Profile 1": COLOR_LIST[0],
        "Profile 2": "green",
    }
    assert default_dict["bar_width"] == 1 / 4
    assert default_dict["category_ordering"] == ["Peter", "Moon", "Chris"]
    assert default_dict["font_size"] == 13
    assert default_dict["legend_font_size"] == 20
    assert default_dict["threshold_values"] == [5, 4]
    assert default_dict["threshold_kwds"] == [
        {"linestyle": "--", "color": COLOR_LIST[-1], "label": "Line 1", "linewidth": 2},
        {"linewidth": 4, "linestyle": "-", "color": COLOR_LIST[-2], "label": "Line 2"},
    ]
    assert default_dict["ax"] == ax
