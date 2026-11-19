from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import (
    Any,
    Protocol,
    TypeVar,
    runtime_checkable,
)

import numpy as np
from numpy.typing import NDArray

from votekit.constants import END_OF_RANKING_SET
from votekit.exceptions import TransformContractError
from votekit.pref_profile.profile_matrix import RankMatrix, ScoreMatrix
from votekit.pref_profile.profile_transform.transform_utils import (
    _ballot_label,
    _decode_rows,
    _expand_to_unit_weights,
)

RankingRow = NDArray[np.object_]

ScoresRow = NDArray[np.float64]

MatrixT = TypeVar("MatrixT", RankMatrix, ScoreMatrix)


@runtime_checkable
class ProfileTransform(Protocol[MatrixT]):
    """
    Contract for profile transformations that act on all selected voters at once.

    Any object with a ``transform_matrix`` method can be passed to ``perturb()`` and is assignable
    to this protocol. ``transform_matrix`` consumes a matrix of the selected rows, of the same kind
    as the profile's matrix. Each row's weight is its number of selected voters, stored as
    float64 whatever the profile's weight kind. It returns a matrix of the same kind, must
    conserve total weight, and must keep the same ranking or score columns, so rankings keep the
    profile's ``max_ranking_length`` positions. A transform may be deterministic or random; a random
    transform owns its randomness, e.g. by holding an ``np.random.Generator``,
    and should treat each unit of weight as an independent voter, e.g. by splitting each row into
    one row per unit of weight.
    """

    def transform_matrix(self, matrix: MatrixT) -> MatrixT: ...


@dataclass(frozen=True)
class _PerVoterTransformFn:
    """
    A per-voter transform function lifted to the ProfileTransform protocol.

    ``transform_matrix`` splits each row of the selected voters' matrix into one row per unit of
    weight, calls the function once per unit on a writeable copy of that unit's ballot, checks
    each result, and constructs a matrix of the same kind from the results. The output has one row
    of weight 1 per unit of input weight, so total weight is conserved.

    The RankMatrix ballots are transformed from IDs to candidate sets for the transform_fn.
    """

    transform_vote_fn: Callable[[NDArray], Any]

    def transform_matrix(self, matrix: MatrixT) -> MatrixT:
        source_row_idxs, unit_matrix_data = _expand_to_unit_weights(matrix)
        unit_rows = _decode_rows(matrix, source_row_idxs)

        for unit_idx, source_row_idx in enumerate(source_row_idxs.tolist()):
            perturbed_row = self.transform_vote_fn(unit_rows[unit_idx])
            unit_rows[unit_idx] = _validate_row(perturbed_row, matrix, source_row_idx)

        if isinstance(matrix, RankMatrix):
            _validate_cast_candidates(unit_rows, source_row_idxs, matrix)
            return type(matrix).from_array(
                unit_rows,
                voter_weight_data=unit_matrix_data,
                ranking_columns=matrix.ranking_columns,
                candidates=matrix.candidates,
            )
        return replace(matrix, voter_weight_data=unit_matrix_data, scores=unit_rows)


def _validate_row(
    perturbed_row: Any, matrix: RankMatrix | ScoreMatrix, source_row_idx: int
) -> NDArray:
    """
    Converts one row returned by a per-voter transform function to a 1D array and checks it.

    A list or tuple is converted with ``np.asarray``; an array is used as is, without a copy.
    Ranking entries may be sets or frozensets, and sets are converted to frozensets as
    ``RankProfile`` does. Score entries must be real numbers.

    Args:
        perturbed_row (Any): Row returned by the function.
        matrix (RankMatrix | ScoreMatrix): Matrix the function's input was decoded from.
        source_row_idx (int): Row index in ``matrix`` of the input row. Errors report its Ballot
            Index.

    Returns:
        NDArray: The row as a 1D array.

    Raises:
        TransformContractError: The row is not a 1D array of the matrix's width, a ranking row
            holds entries other than sets or frozensets, or a score row holds non-numeric or
            negative scores.
    """
    try:
        row = np.asarray(perturbed_row)
    except ValueError as error:
        # e.g. a ragged list, which numpy cannot make into an array
        raise TransformContractError(
            "transform_fn must return a 1D array, list, or tuple. Returned"
            f" {perturbed_row} for Ballot Index"
            f" {_ballot_label(matrix.voter_weight_data, source_row_idx)}."
        ) from error

    width = (
        matrix.num_ranking_positions
        if isinstance(matrix, RankMatrix)
        else len(matrix.score_columns)
    )
    if row.shape != (width,):
        raise TransformContractError(
            f"transform_fn must return a 1D array of length {width}, one entry per"
            f" {'ranking position' if isinstance(matrix, RankMatrix) else 'score column'}."
            f" Returned {perturbed_row} for Ballot Index"
            f" {_ballot_label(matrix.voter_weight_data, source_row_idx)}."
        )

    if isinstance(matrix, RankMatrix):
        for position, cand_set in enumerate(row):
            if not isinstance(cand_set, (set, frozenset)):
                raise TransformContractError(
                    "transform_fn must return an array of sets or frozensets of candidates,"
                    " padding the end with frozenset({'~'}). Returned"
                    f" {perturbed_row} for Ballot Index"
                    f" {_ballot_label(matrix.voter_weight_data, source_row_idx)}."
                )
            if isinstance(cand_set, set):
                row[position] = frozenset(cand_set)
        return row

    # ScoreMatrix
    if row.dtype.kind not in "iuf":
        raise TransformContractError(
            f"transform_fn returned a non-numeric score. Returned {perturbed_row} for Ballot"
            f" Index {_ballot_label(matrix.voter_weight_data, source_row_idx)}."
        )
    if np.any(row < 0):
        raise TransformContractError(
            f"transform_fn returned a negative score. Returned {perturbed_row} for Ballot Index"
            f" {_ballot_label(matrix.voter_weight_data, source_row_idx)}."
        )
    return row


def _validate_cast_candidates(
    perturbed_rankings: NDArray[np.object_], source_row_idxs: NDArray[np.intp], matrix: RankMatrix
) -> None:
    """
    Validates only the profile's candidates are ranked in the perturbed rankings.

    Candidates cast are those ranked on positive-weight ballots. Every array row has weight 1, so
    every candidate in a perturbed ranking is cast and must be one of the profile's candidates.

    Args:
        unit_rows (NDArray[np.object_]): Perturbed rankings of candidate sets, one row per unit.
        source_row_idxs (NDArray[np.intp]): Row index in ``matrix`` of each unit's input row.
        matrix (RankMatrix): Matrix the units were decoded from.

    Raises:
        TransformContractError: A ranking ranks a candidate outside the profile's candidates.
    """
    distinct_cand_sets = set(perturbed_rankings.ravel())
    cands_cast = frozenset().union(*distinct_cand_sets) - END_OF_RANKING_SET
    undeclared_cands = cands_cast - frozenset(matrix.candidates)
    if not undeclared_cands:
        return

    invalid_ranking_idx = next(
        ranking_idx
        for ranking_idx, ranking in enumerate(perturbed_rankings)
        if any(cand_set & undeclared_cands for cand_set in ranking)
    )
    ballot_label = _ballot_label(matrix.voter_weight_data, source_row_idxs[invalid_ranking_idx])
    raise TransformContractError(
        f"transform_fn returned candidates {set(undeclared_cands)} not in the profile's"
        " candidates. Declare them in the profile's candidates before perturbing. Returned"
        f" {perturbed_rankings[invalid_ranking_idx]} for Ballot Index {ballot_label}."
    )


def _lift_to_profile_transform(transform_fn: Any) -> ProfileTransform[Any]:
    """
    Lifts the transform function to the ProfileTransform protocol.

    A ProfileTransform is returned unchanged. A callable is a per-voter transform function and is
    wrapped so that it is called once per unit of weight.

    Args:
        transform_fn (ProfileTransform | Callable[[NDArray], NDArray]): ProfileTransform instance
            or per-voter transform function.

    Returns:
        ProfileTransform: ``transform_fn`` itself, or the per-voter function lifted to a
            ProfileTransform.

    Raises:
        TransformContractError: ``transform_fn`` is a class, or neither callable nor a
            ProfileTransform.
    """

    if isinstance(transform_fn, type):
        raise TransformContractError(
            f"transform_fn was given the class {transform_fn.__name__}, not an instance."
            f" Did you mean to construct it like {transform_fn.__name__}(...)?"
        )

    if isinstance(transform_fn, ProfileTransform):
        return transform_fn
    if callable(transform_fn):
        return _PerVoterTransformFn(transform_fn)
    raise TransformContractError(
        "transform_fn must be a per-voter transform function or ProfileTransform instance,"
        f" received {transform_fn}."
    )
