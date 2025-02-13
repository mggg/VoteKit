import pytest

from votekit.plots.bar_plot import _validate_bar_plot_args


def test_validate_non_neg_bars():
    data = {
        "Profile 1": {"Peter": 6, "Chris": -7},
        "Profile 2": {
            "Chris": 4,
            "Peter": 3,
        },
    }

    with pytest.raises(ValueError, match="All bar heights must be non-negative."):
        _validate_bar_plot_args(
            data=data,
            category_ordering=["Peter", "Chris"],
            bar_width=1 / 4,
            threshold_values=None,
            threshold_kwds=None,
        )


def test_validate_sub_dictionary_keys():
    data = {
        "Profile 1": {"David": 5, "Peter": 6, "Moon": 7},
        "Profile 2": {
            "Chris": 4,
            "Peter": 3,
        },
    }

    with pytest.raises(
        ValueError, match="All data dictionaries must have the same categorical keys."
    ):
        _validate_bar_plot_args(
            data=data,
            category_ordering=["Peter"],
            bar_width=1 / 4,
            threshold_values=None,
            threshold_kwds=None,
        )


def test_validate_x_label_ordering():
    data = {
        "Profile 1": {
            "Chris": 5,
            "Peter": 6,
        },
        "Profile 2": {
            "Chris": 4,
            "Peter": 3,
        },
    }

    with pytest.raises(
        ValueError,
        match=(
            "category_ordering must match the keys of the data sub-dictionaries. Here is the"
            " symmetric difference of the two sets: (.*?)"
        ),
    ):
        _validate_bar_plot_args(
            data=data,
            category_ordering=["Peter", "Moon"],
            bar_width=1 / 4,
            threshold_values=None,
            threshold_kwds=None,
        )


def test_validate_bar_width():
    data = {
        "Profile 1": {
            "Chris": 5,
            "Peter": 6,
        },
        "Profile 2": {
            "Chris": 4,
            "Peter": 3,
        },
    }

    with pytest.raises(ValueError, match="Bar width must be positive."):
        _validate_bar_plot_args(
            data=data,
            category_ordering=["Peter", "Chris"],
            bar_width=-1 / 4,
            threshold_values=None,
            threshold_kwds=None,
        )

    with pytest.warns(
        UserWarning,
        match="Bar width must be less than 1. Bar width has now been set to 1.",
    ):
        _validate_bar_plot_args(
            data=data,
            category_ordering=["Peter", "Chris"],
            bar_width=2,
            threshold_values=None,
            threshold_kwds=None,
        )


def test_validate_thresholds():
    data = {
        "Profile 1": {
            "Chris": 5,
            "Peter": 6,
        },
        "Profile 2": {
            "Chris": 4,
            "Peter": 3,
        },
    }

    with pytest.raises(
        ValueError,
        match="threshold_values must have the same length as threshold_kwds.",
    ):
        _validate_bar_plot_args(
            data=data,
            category_ordering=["Peter", "Chris"],
            bar_width=1 / 4,
            threshold_values=[4, 5],
            threshold_kwds=[{"h": "4"}],
        )

    with pytest.raises(ValueError, match="Must use linestyle, not ls."):
        _validate_bar_plot_args(
            data=data,
            category_ordering=["Peter", "Chris"],
            bar_width=1 / 4,
            threshold_values=[4],
            threshold_kwds=[{"ls": "-"}],
        )

    with pytest.raises(ValueError, match="Must use linewidth, not lw."):
        _validate_bar_plot_args(
            data=data,
            category_ordering=["Peter", "Chris"],
            bar_width=1 / 4,
            threshold_values=[4],
            threshold_kwds=[{"lw": 4}],
        )
