from votekit.plots import bar_plot
from votekit.plots.profile_plots import (
    add_null_keys,
    _normalize_data,
    _set_default_bar_plot_args,
    _prepare_data_bar_plot,
)
import pytest
from matplotlib.axes import Axes
from votekit.utils import COLOR_LIST
import matplotlib.pyplot as plt

test_data_1 = {"Moon": 4.0, "Peter": 0.0, "Chris": 2.0}
test_data_2 = {"Moon": 4.0, "Peter": 1.0, "Chris": 1.0}
test_data_3 = {"Mala": 4.0, "David": 1.0, "Jeanne": 1.0}


def test_add_null_keys():
    data = add_null_keys([test_data_1, {"Peter": 1.0, "Chris": 1.0, "Moon": 4.0}])
    assert data[0].keys() == data[1].keys()
    assert list(data[0].keys()) == ["Moon", "Peter", "Chris"]

    data = add_null_keys([test_data_1, test_data_3])
    assert data[0].keys() == data[1].keys()
    assert list(data[0].keys()) == ["Moon", "Peter", "Chris", "Mala", "David", "Jeanne"]
    assert data[0]["Jeanne"] == 0 and data[1]["Moon"] == 0


def test_bar_plot_simple():
    assert isinstance(bar_plot(test_data_1), Axes)


def test_bar_plot_multi_sets():
    assert isinstance(
        bar_plot([test_data_1, test_data_2], data_set_labels=["1", "2"]), Axes
    )

    data = add_null_keys([test_data_1, test_data_3])
    assert isinstance(bar_plot(data, data_set_labels=["1", "3"]), Axes)


def test_bar_plot_threshold_lines():
    assert isinstance(bar_plot(test_data_1, threshold_values=[2, 3]), Axes)


def test_bar_plot_validation_errors():
    with pytest.raises(
        ValueError, match="All data dictionaries must have the same keys."
    ):
        bar_plot([test_data_1, test_data_3])

    with pytest.raises(ValueError, match="Each data set must have a unique label."):
        bar_plot([test_data_1, test_data_2], data_set_labels=["1", "1"])

    with pytest.raises(
        ValueError, match="There must be one data set label for each data set."
    ):
        bar_plot([test_data_1, test_data_2], data_set_labels=["1"])

    with pytest.raises(
        ValueError, match="There must be one data set label for each data set."
    ):
        bar_plot(test_data_1, data_set_labels=["1", "2"])

    with pytest.raises(
        ValueError,
        match=(
            "x_label_ordering must match the keys of the data."
            " Here is the symmetric difference of the two sets:"
        ),
    ):
        bar_plot(test_data_1, x_label_ordering=["Chris"])

    with pytest.raises(
        ValueError,
        match=(
            "x_label_ordering must match the keys of the data."
            " Here is the symmetric difference of the two sets:"
        ),
    ):
        bar_plot(test_data_1, x_label_ordering=["Chris", "Moon", "Peter", "Mala"])

    with pytest.raises(
        ValueError,
        match=(
            "x_label_ordering must match the keys of the data."
            " Here is the symmetric difference of the two sets:"
        ),
    ):
        bar_plot(
            add_null_keys([test_data_1, test_data_3]),
            x_label_ordering=["Chris", "Moon", "Peter", "Mala"],
            data_set_labels=["1", "2"],
        )

    with pytest.raises(ValueError, match="Bar width must be positive."):
        bar_plot(test_data_1, bar_width=0)

    with pytest.raises(ValueError, match="Bar width must be positive."):
        bar_plot(test_data_1, bar_width=-1)

    with pytest.warns(
        UserWarning,
        match="Bar width must be less than 1. Bar width has now been set to 1.",
    ):
        bar_plot(test_data_1, bar_width=2)

    with pytest.raises(
        ValueError,
        match="threshold_values must have the same length as threshold_kwds.",
    ):
        bar_plot(test_data_1, threshold_values=[2, 3], threshold_kwds=[{"lw": 4}])


def test_normalization():
    assert _normalize_data(test_data_1) == {"Moon": 4 / 6, "Chris": 2 / 6, "Peter": 0}


def test_normalizaton_errors():
    with pytest.raises(
        ValueError, match="Total mass of observations must be non-zero."
    ):
        _normalize_data({"Chris": 0})


def test_default_bar_plot_args_with_defaults():

    (
        data,
        data_set_labels,
        data_set_to_color,
        bar_width,
        x_label_ordering,
        FONT_SIZE,
        legend_font_size,
        threshold_values,
        threshold_kwds,
        ax,
    ) = _set_default_bar_plot_args(
        test_data_1, None, None, None, None, None, None, None, None
    )

    assert isinstance(data, list)
    assert data[0] == test_data_1

    assert data_set_labels == ["Data Set 1"]
    assert data_set_to_color == {"Data Set 1": COLOR_LIST[0]}

    assert bar_width == 0.7

    assert isinstance(x_label_ordering, list)
    assert isinstance(x_label_ordering[0], str)

    assert FONT_SIZE == 13
    assert legend_font_size == FONT_SIZE

    assert isinstance(ax, Axes)

    assert ax.figure.get_size_inches()[0] == 3 * len(test_data_1)
    assert ax.figure.get_size_inches()[1] == 6

    assert threshold_values is None
    assert threshold_kwds is None


def test_default_bar_plot_args_without_defaults():
    fig, ax = plt.subplots()
    (
        data,
        data_set_labels,
        data_set_to_color,
        bar_width,
        x_label_ordering,
        FONT_SIZE,
        legend_font_size,
        threshold_values,
        threshold_kwds,
        ax_new,
    ) = _set_default_bar_plot_args(
        [test_data_1, test_data_2],
        ["data", "datum"],
        {"data": "red", "datum": "green"},
        bar_width=0.35,
        x_label_ordering=["Chris", "Moon", "Peter"],
        legend_font_size=15,
        threshold_values=5,
        threshold_kwds={"ls": "--"},
        ax=ax,
    )

    assert data == [test_data_1, test_data_2]

    assert data_set_labels == ["data", "datum"]

    assert data_set_to_color == {"data": "red", "datum": "green"}

    assert bar_width == 0.35 / len(data)

    assert x_label_ordering == ["Chris", "Moon", "Peter"]

    assert FONT_SIZE == 13
    assert legend_font_size == 15

    assert threshold_values == [5]
    assert threshold_kwds == [
        {"ls": "--", "linewidth": 2, "color": "#6200EA", "label": "Line 0"}
    ]

    assert ax_new == ax


def test_prepare_data_bar_plot():

    y_data_norm = _prepare_data_bar_plot(
        True, [test_data_1, test_data_2], ["Chris", "Moon", "Peter"]
    )
    y_data = _prepare_data_bar_plot(
        False, [test_data_1, test_data_2], ["Chris", "Moon", "Peter"]
    )
    # y_data_disjoint_labels = _prepare_data_bar_plot(
    #     False, [test_data_1, {"Mala": 4.0}], ["Chris", "Moon", "Peter", "Mala"]
    # )

    assert len(y_data_norm) == 2
    assert y_data_norm[0][0] == 2 / 6
    assert y_data[0][0] == 2
    # assert y_data_disjoint_labels[0][3] == 0 and y_data_disjoint_labels[1][0] == 0
