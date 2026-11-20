"""
Helpers shared by the transform protocol, the transforms, and ``perturb``.

This module imports nothing from the rest of the package, so every other module can import it.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from votekit.pref_profile.profile_matrix import RankMatrix, ScoreMatrix, WeightKind
from votekit.pref_profile.profile_matrix.shared_data import ProfileVoterWeightData


def _expand_to_unit_weights(
    matrix: RankMatrix | ScoreMatrix,
) -> tuple[NDArray[np.intp], ProfileVoterWeightData]:
    """
    Splits each row of the matrix into one row per unit of weight.

    If a row has weight greater than 1 and a non-empty voter set, the set must be split. Each unit
    is assigned one voter from its row's voter set, in sorted order. Voters were already sampled
    without replacement when the profile was masked. Otherwise, the voter set is left untouched.

    A non-empty voter set must have one voter per unit of weight. ``perturb`` checks this for
    every row when masking, before any rows are selected, so it is not checked again here.

    Args:
        matrix (RankMatrix | ScoreMatrix): Matrix with integral weights.

    Returns:
        tuple[NDArray[np.intp], ProfileVoterWeightData]: Row index in ``matrix`` that each unit
            comes from, and voter weight data with weight 1 and at most one voter per unit.

    Raises:
        ValueError: The matrix has non-integral weights. The row cannot be split per unit of weight.
    """
    matrix_data = matrix.voter_weight_data

    if np.all(matrix_data.weights == 1):
        return np.arange(len(matrix)), matrix_data

    int_weights = _integral_weights(matrix_data)
    source_row_idxs = np.repeat(np.arange(len(matrix)), int_weights)

    unit_voter_sets = None
    if matrix_data.voter_sets is not None:
        unit_voter_sets = np.empty(len(source_row_idxs), dtype=object)
        unit_idx = 0
        for voter_set, weight in zip(matrix_data.voter_sets, int_weights):
            end_unit = unit_idx + weight
            if voter_set == frozenset():
                unit_voter_sets[unit_idx:end_unit] = voter_set
            else:
                unit_voter_sets[unit_idx:end_unit] = [
                    frozenset({voter}) for voter in sorted(voter_set, key=str)
                ]
            unit_idx = end_unit

    unit_matrix_data = ProfileVoterWeightData(
        weights=np.ones(len(source_row_idxs), dtype=np.float64),
        voter_sets=unit_voter_sets,
        weight_kind=WeightKind.FLOAT,
    )
    return source_row_idxs, unit_matrix_data


def _decode_rows(matrix: RankMatrix | ScoreMatrix, row_idxs: NDArray[np.intp]) -> NDArray:
    """
    Writeable copies of the rows at the given row indices.

    Rows hold candidate sets for rankings and floats for scores.

    Args:
        matrix (RankMatrix | ScoreMatrix): Source matrix.
        row_idxs (NDArray[np.intp]): Row indices in ``matrix`` to decode, repeats allowed.

    Returns:
        NDArray: One decoded row per row index.
    """
    if isinstance(matrix, RankMatrix):
        return matrix.get_ranking_array(row_idxs)
    return matrix.scores[row_idxs]


def _read_only_rows(matrix: RankMatrix | ScoreMatrix) -> NDArray:
    """
    Read-only rows of the matrix, in the same form a per-voter transform function receives.

    Args:
        matrix (RankMatrix | ScoreMatrix): Source matrix.

    Returns:
        NDArray: Candidate sets for rankings, or the matrix's own read-only scores.
    """
    if isinstance(matrix, RankMatrix):
        rows = matrix.get_ranking_array()
        rows.setflags(write=False)
        return rows
    return matrix.scores


def _ballot_label(matrix_data: ProfileVoterWeightData, row_idx: int) -> Any:
    """
    The Ballot Index label of the row at a row index, as shown by ``profile.to_df()``.

    Args:
        matrix_data (ProfileVoterWeightData): Shared ballot data of the matrix holding the row.
        row_idx (int): Position of the row in the matrix's arrays.

    Returns:
        Any: The row's Ballot Index label, which is the row index itself when the profile uses the
            default RangeIndex.
    """
    ballot_index = matrix_data.ballot_index
    return row_idx if ballot_index is None else ballot_index[row_idx]


def _integral_weights(data: ProfileVoterWeightData) -> NDArray[np.int64]:
    """
    Converts the profile's weights to integers, one per voter.

    Args:
        data (ProfileVoterWeightData): Shared ballot data of the profile's matrix.

    Returns:
        NDArray[np.int64]: Integer weight of each row.

    Raises:
        ValueError: The profile has non-integral weights.
    """
    if data.has_fraction_weights:
        integral = all(weight.denominator == 1 for weight in data.weights)
    else:
        integral = bool(np.all(data.weights % 1 == 0))
    if not integral:
        raise ValueError(
            "perturb treats each unit of weight as one voter and does not support non-integral"
            " weights. Convert a copy of the profile to integer weights."
        )
    if data.has_fraction_weights:
        return np.array([int(weight) for weight in data.weights], dtype=np.int64)

    return data.weights.astype(np.int64)
