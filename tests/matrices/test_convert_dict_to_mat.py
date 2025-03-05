from votekit.matrices._utils import _convert_dict_to_matrix
import numpy as np
import pytest


def test_convert():
    data = {
        "Chris": {"Chris": 2, "Peter": 3, "Moon": 4},
        "Peter": {"Chris": 1, "Peter": 5, "Moon": 2},
        "Moon": {"Chris": 0, "Peter": -1, "Moon": 0.25},
    }

    mat = _convert_dict_to_matrix(data)

    assert isinstance(mat, np.ndarray)
    assert mat[0][0] == 2
    assert mat[2][1] == -1


def test_convert_float_error():
    data = {
        "Chris": {"Chris": 2, "Peter": "hi", "Moon": 4},
        "Peter": {"Chris": 1, "Peter": 5, "Moon": 2},
        "Moon": {"Chris": 0, "Peter": -1, "Moon": 0.25},
    }

    with pytest.raises(ValueError, match="could not convert (.*?) to float"):
        _convert_dict_to_matrix(data)


def test_convert_size_error():
    data = {
        "Chris": {"Chris": 2, "Peter": 1, "Moon": 4},
        "Peter": {"Chris": 1, "Peter": 5, "Moon": 2, "David": 5},
        "Moon": {"Chris": 0, "Peter": -1, "Moon": 0.25},
    }

    with pytest.raises(AssertionError, match="Inner keys do not match across all rows"):
        _convert_dict_to_matrix(data)
