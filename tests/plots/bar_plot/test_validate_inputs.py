import pytest

from votekit.plots.bar_plot import _validate_bar_plot_args
from votekit.utils import COLOR_LIST


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
            categories_legend=None,
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
            categories_legend=None,
            bar_width=1 / 4,
            threshold_values=None,
            threshold_kwds=None,
        )


def test_validate_number_of_data_sets():
    data = {f"Label_{i}": {"C": 4} for i in range(len(COLOR_LIST) + 1)}
    with pytest.raises(
        ValueError, match=f"Cannot plot more than {len(COLOR_LIST)} data sets."
    ):
        _validate_bar_plot_args(
            data=data,
            category_ordering=["C"],
            categories_legend=None,
            bar_width=1 / 4,
            threshold_values=None,
            threshold_kwds=None,
        )


def test_validate_category_ordering_length():
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
        match="category_ordering must be the same length as sub-dictionaries.",
    ):
        _validate_bar_plot_args(
            data=data,
            category_ordering=["Peter", "Moon", "Peter"],
            categories_legend=None,
            bar_width=1 / 4,
            threshold_values=None,
            threshold_kwds=None,
        )


def test_validate_category_ordering_extraneous():
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

    with pytest.raises(ValueError, match="category_ordering has extraneous labels: "):
        _validate_bar_plot_args(
            data=data,
            category_ordering=["Peter", "Moon"],
            categories_legend=None,
            bar_width=1 / 4,
            threshold_values=None,
            threshold_kwds=None,
        )


def test_validate_category_ordering_missing():
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

    with pytest.raises(ValueError, match="category_ordering has missing labels: "):
        _validate_bar_plot_args(
            data=data,
            category_ordering=["Peter", "Peter"],
            categories_legend=None,
            bar_width=1 / 4,
            threshold_values=None,
            threshold_kwds=None,
        )


def test_validate_categories_legend():
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
        ValueError, match="{'xyz'} in categories_legend must be a subset of"
    ):
        _validate_bar_plot_args(
            data=data,
            category_ordering=["Peter", "Chris"],
            categories_legend={"Peter": 4, "xyz": 5},
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
            categories_legend=None,
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
            categories_legend=None,
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
            categories_legend=None,
            bar_width=1 / 4,
            threshold_values=[4, 5],
            threshold_kwds=[{"h": "4"}],
        )

    with pytest.raises(ValueError, match="Must use linestyle, not ls."):
        _validate_bar_plot_args(
            data=data,
            category_ordering=["Peter", "Chris"],
            categories_legend=None,
            bar_width=1 / 4,
            threshold_values=[4],
            threshold_kwds=[{"ls": "-"}],
        )

    with pytest.raises(ValueError, match="Must use linewidth, not lw."):
        _validate_bar_plot_args(
            data=data,
            category_ordering=["Peter", "Chris"],
            categories_legend=None,
            bar_width=1 / 4,
            threshold_values=[4],
            threshold_kwds=[{"lw": 4}],
        )
