from votekit.plots import bar_plot
from votekit.plots.profile_plots import (
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


def test_bar_plot_simple():
    assert isinstance(bar_plot(test_data_1), Axes)


def test_bar_plot_multi_sets():
    assert isinstance(
        bar_plot([test_data_1, test_data_2], data_set_labels=["1", "2"]), Axes
    )
    assert isinstance(
        bar_plot([test_data_1, test_data_3], data_set_labels=["1", "3"]), Axes
    )


def test_bar_plot_validation_errors():
    with pytest.raises(ValueError, match="Data set labels must be unique."):
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
        ValueError, match="x_label_ordering must match the keys of the data."
    ):
        bar_plot(test_data_1, x_label_ordering=["Chris"])

    with pytest.raises(
        ValueError, match="x_label_ordering must match the keys of the data."
    ):
        bar_plot(test_data_1, x_label_ordering=["Chris", "Moon", "Peter", "Mala"])

    with pytest.raises(
        ValueError, match="x_label_ordering must match the keys of the data."
    ):
        bar_plot(
            [test_data_1, test_data_3],
            x_label_ordering=["Chris", "Moon", "Peter", "Mala"],
            data_set_labels=["1", "2"],
        )


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
        ax,
    ) = _set_default_bar_plot_args(test_data_1, None, None, None, None, None)

    assert isinstance(data, list)
    assert data[0] == test_data_1

    assert data_set_labels == ["Data Set 1"]
    assert data_set_to_color == {"Data Set 1": COLOR_LIST[0]}

    assert bar_width == 0.7

    assert isinstance(x_label_ordering, list)
    assert isinstance(x_label_ordering[0], str)

    assert isinstance(ax, Axes)


def test_default_bar_plot_args_without_defaults():
    fig, ax = plt.subplots()
    (
        data,
        data_set_labels,
        data_set_to_color,
        bar_width,
        x_label_ordering,
        ax_new,
    ) = _set_default_bar_plot_args(
        [test_data_1, test_data_2],
        ["data", "datum"],
        {"data": "red", "datum": "green"},
        bar_width=0.35,
        x_label_ordering=["Chris", "Moon", "Peter"],
        ax=ax,
    )

    assert data == [test_data_1, test_data_2]

    assert data_set_labels == ["data", "datum"]

    assert data_set_to_color == {"data": "red", "datum": "green"}

    assert bar_width == 0.35

    assert x_label_ordering == ["Chris", "Moon", "Peter"]

    assert ax_new == ax


def test_prepare_data_bar_plot():

    y_data_norm = _prepare_data_bar_plot(
        True, [test_data_1, test_data_2], ["Chris", "Moon", "Peter"]
    )
    y_data = _prepare_data_bar_plot(
        False, [test_data_1, test_data_2], ["Chris", "Moon", "Peter"]
    )
    y_data_disjoint_labels = _prepare_data_bar_plot(
        False, [test_data_1, {"Mala": 4.0}], ["Chris", "Moon", "Peter", "Mala"]
    )

    assert len(y_data_norm) == 2
    assert y_data_norm[0][0] == 2 / 6
    assert y_data[0][0] == 2
    assert y_data_disjoint_labels[0][3] == 0 and y_data_disjoint_labels[1][0] == 0
