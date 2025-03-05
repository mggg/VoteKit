from votekit.plots import bar_plot
from matplotlib.axes import Axes
import matplotlib.pyplot as plt

data = {"Chris": 5, "Peter": 6, "Moon": 7}


def test_barplot_with_defaults():
    ax = bar_plot(data)

    assert isinstance(ax, Axes)
    plt.close()


def test_barplot_with_kwds():
    ax = bar_plot(
        data,
        data_set_label="Profile",
        normalize=True,
        data_set_color="red",
        bar_width=1,
        category_ordering=["Chris", "Moon", "Peter"],
        x_axis_name="candidates",
        y_axis_name="fpv",
        title="plot",
        show_data_set_legend=True,
        categories_legend={"Chris": 5, "Peter": 6, "Moon": 7},
        threshold_values=1,
        threshold_kwds={"label": "threshold"},
        legend_font_size=4,
    )

    assert isinstance(ax, Axes)
    plt.close()
