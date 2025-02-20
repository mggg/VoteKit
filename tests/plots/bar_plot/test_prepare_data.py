from votekit.plots.bar_plot import _prepare_data_bar_plot

data = {
    "Profile 1": {"Chris": 5, "Peter": 6, "Moon": 7},
    "Profile 2": {"Chris": 4, "Peter": 3, "Moon": 2},
}


def test_prepare_data():
    y_data = _prepare_data_bar_plot(
        data=data, normalize=False, category_ordering=["Chris", "Moon", "Peter"]
    )

    assert y_data == [[5, 7, 6], [4, 2, 3]]


def test_normalize_parameter():
    y_data = _prepare_data_bar_plot(
        data=data, normalize=True, category_ordering=["Chris", "Moon", "Peter"]
    )

    assert y_data == [[5 / 18, 7 / 18, 6 / 18], [4 / 9, 2 / 9, 3 / 9]]
