from votekit.plots.bar_plot import _normalize_data_dict
import pytest


data = {
    "Profile 1": {"Chris": 5, "Peter": 6, "Moon": 7},
    "Profile 2": {"Chris": 4, "Peter": 3, "Moon": 2},
}


def test_normalize_data():
    normalized_dict = _normalize_data_dict(data["Profile 1"])

    assert normalized_dict == {"Chris": 5 / 18, "Peter": 6 / 18, "Moon": 7 / 18}


def test_normalize_data_0():

    with pytest.raises(
        ValueError, match="Total mass of observations must be non-zero."
    ):
        _normalize_data_dict({"Chris": 0})
