from votekit.plots.profiles.profile_bar_plot import _validate_inputs
import pytest


def test_validate_inputs():
    with pytest.raises(
        ValueError,
        match="stat_function string foo not an available statistic."
        " Available stats include: ",
    ):
        _validate_inputs("foo")

    _validate_inputs("first place votes")
    _validate_inputs("mentions")
    _validate_inputs("borda")
    _validate_inputs("ballot lengths")
