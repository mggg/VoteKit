import pytest
from votekit.matrices.heatmap import _validate_heatmap_inputs
import numpy as np


def test_2d_numpy():
    with pytest.raises(
        ValueError, match="Please provide a 2D matrix to plot. Found a (.*?)-D matrix."
    ):
        _validate_heatmap_inputs(matrix=np.array([1, 2, 3]))

    with pytest.raises(
        ValueError, match="Please provide a 2D matrix to plot. Found a (.*?)-D matrix."
    ):
        _validate_heatmap_inputs(matrix=np.array([[[1], [2]], [[1], [2]]]))


def test_row_labels():
    with pytest.raises(
        ValueError,
        match="Please provide (.*?) labels for the rows of the matrix. Found (.*?) labels.",
    ):
        _validate_heatmap_inputs(
            matrix=np.array([[1, 2, 3], [4, 5, 6]]), row_labels=["chris"]
        )


def test_col_labels():
    with pytest.raises(
        ValueError,
        match="Please provide (.*?) labels for the columns of the matrix. Found (.*?) labels.",
    ):
        _validate_heatmap_inputs(
            matrix=np.array([[1, 2, 3], [4, 5, 6]]), column_labels=["chris"]
        )


def test_row_legend():
    with pytest.raises(ValueError, match="Row labels do not match row legend keys."):
        _validate_heatmap_inputs(
            matrix=np.array([[1, 2, 3], [4, 5, 6]]),
            row_labels=["chris", "peter"],
            row_legend={"chris": "2", "Moon": "4"},
        )


def test_col_legend():
    with pytest.raises(
        ValueError, match="Column labels do not match column legend keys."
    ):
        _validate_heatmap_inputs(
            matrix=np.array([[1, 2, 3], [4, 5, 6]]),
            column_labels=["chris", "peter", "moon"],
            column_legend={"chris": "2", "Moon": "4"},
        )


def test_legend_conflict():
    with pytest.raises(
        ValueError,
        match="Conflicting legend descriptions for (.*?): got (.*?) and (.*?).",
    ):
        _validate_heatmap_inputs(
            matrix=np.array([[1, 2], [4, 5]]),
            column_labels=["chris", "peter"],
            column_legend={"chris": "2", "peter": "4"},
            row_labels=["chris", "peter"],
            row_legend={"chris": "2", "peter": "whoopps"},
        )
