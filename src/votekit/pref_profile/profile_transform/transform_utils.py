"""
Helpers shared by the transform protocol, the transforms, ``map``, and ``transform``.

This module imports only ``protocol`` from the rest of the package, so every other module can
import it.
"""

from collections.abc import Iterable
from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from votekit.exceptions import ProfileError
from votekit.pref_profile.profile_matrix import RankMatrix, ScoreMatrix, WeightKind
from votekit.pref_profile.profile_matrix.shared_data import ProfileVoterWeightData
from votekit.pref_profile.profile_transform.protocol import MatrixT
from votekit.types import Candidate


def _split_per_voter(matrix: MatrixT) -> MatrixT:
    """
    Splits each row of the matrix into one row per voter, that is, per unit of weight.

    Each voter's row repeats its source row's ballot and has weight 1. A row with a non-empty voter
    set gives each of its rows one voter, in sorted order; a row without voters gives its rows an
    empty voter set. A row of weight 0 has no voters, so it has no rows in the result.

    The matrix must already satisfy ``_validate_one_voter_per_unit_weight``.

    Args:
        matrix (RankMatrix | ScoreMatrix): Matrix to split.

    Returns:
        RankMatrix | ScoreMatrix: Matrix of the same kind with one row of weight 1 per voter.
    """
    matrix_data = matrix.voter_weight_data
    int_weights = _int_weights(matrix_data)
    if np.all(int_weights == 1):
        return matrix

    source_row_idxs = np.repeat(np.arange(len(matrix)), int_weights)

    voter_sets = None
    if matrix_data.voter_sets is not None:
        voter_sets = np.empty(len(source_row_idxs), dtype=object)
        unit_idx = 0
        for voter_set, weight in zip(matrix_data.voter_sets, int_weights):
            end_unit = unit_idx + weight
            if voter_set == frozenset():
                voter_sets[unit_idx:end_unit] = voter_set
            else:
                voter_sets[unit_idx:end_unit] = [
                    frozenset({voter}) for voter in sorted(voter_set, key=str)
                ]
            unit_idx = end_unit

    voter_weight_data = ProfileVoterWeightData(
        weights=np.ones(len(source_row_idxs), dtype=np.float64),
        voter_sets=voter_sets,
        weight_kind=WeightKind.FLOAT,
    )
    if isinstance(matrix, RankMatrix):
        return replace(
            matrix, voter_weight_data=voter_weight_data, rankings=matrix.rankings[source_row_idxs]
        )
    return replace(
        matrix, voter_weight_data=voter_weight_data, scores=matrix.scores[source_row_idxs]
    )


def _decode_rows(matrix: RankMatrix | ScoreMatrix) -> NDArray:
    """
    Writeable copies of every row of the matrix, decoded in one step.

    Rows hold candidate sets for rankings and floats for scores.

    Args:
        matrix (RankMatrix | ScoreMatrix): Source matrix.

    Returns:
        NDArray: One decoded row per row of ``matrix``.
    """
    if isinstance(matrix, RankMatrix):
        return matrix.get_ranking_array()
    return matrix.scores.copy()


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


def _validate_one_voter_per_unit_weight(matrix: RankMatrix | ScoreMatrix) -> None:
    """
    Checks that each unit of weight can be treated as one voter.

    Every weight must be integral, and every non-empty voter set must have one voter per unit of
    weight. Needed before voters are sampled from a row's weight or a row is split per voter.

    Args:
        matrix (RankMatrix | ScoreMatrix): Matrix to check.

    Raises:
        ValueError: The matrix has non-integral weights.
        ProfileError: A row's non-empty voter set does not match its weight.
    """
    matrix_data = matrix.voter_weight_data
    if matrix_data.has_fraction_weights:
        integral = all(weight.denominator == 1 for weight in matrix_data.weights)
    else:
        integral = bool(np.all(matrix_data.weights % 1 == 0))

    if not integral:
        raise ValueError(
            "Each unit of weight is treated as one voter, so non-integral weights are not"
            " supported when mapping per voter or sampling voters with a per_voter_probability"
            " other than 0 or 1. Convert the profile to integer weights."
        )

    if matrix_data.voter_sets is None:
        return

    weights = _int_weights(matrix_data)
    voter_set_sizes = np.fromiter(
        map(len, matrix_data.voter_sets), dtype=np.int64, count=len(matrix_data.voter_sets)
    )
    mismatched = np.flatnonzero((voter_set_sizes > 0) & (voter_set_sizes != weights))
    if len(mismatched):
        row_idx = int(mismatched[0])
        if isinstance(matrix, RankMatrix):
            ballot = matrix.get_ranking(row_idx)
        else:
            ballot = matrix.get_scores(row_idx)
        raise ProfileError(
            "Cannot split the voters of a ballot whose voter set does not match its weight. The"
            f" ballot {ballot} has a voter set of size {voter_set_sizes[row_idx]} but weight"
            f" {weights[row_idx]}."
        )


def _int_weights(data: ProfileVoterWeightData) -> NDArray[np.int64]:
    """
    The weights as integers, one per voter.

    The weights are assumed to be integral, and can be converted to integers without error.

    Args:
        data (ProfileVoterWeightData): Voter Set and Weight data of a matrix.

    Returns:
        NDArray[np.int64]: Integer weight of each row.
    """
    if data.has_fraction_weights:
        return np.array([int(weight) for weight in data.weights], dtype=np.int64)
    return data.weights.astype(np.int64)


def _with_added_candidates(
    candidates: tuple[Candidate, ...], new_candidates: Iterable[Candidate]
) -> tuple[Candidate, ...]:
    """
    The candidates followed by any of ``new_candidates`` they lack, in the given order.

    Candidates are only ever added, never removed.

    Args:
        candidates (tuple[Candidate, ...]): Existing candidates.
        new_candidates (Iterable[Candidate]): Candidates to add if missing.

    Returns:
        tuple[Candidate, ...]: The candidates with missing ones appended.
    """
    existing = set(candidates)
    return candidates + tuple(cand for cand in new_candidates if cand not in existing)
