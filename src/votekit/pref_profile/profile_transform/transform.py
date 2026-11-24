"""
How ``map`` and ``transform`` select voters, change their ballots, and build the result.
"""

from collections.abc import Callable, Sequence
from dataclasses import replace
from fractions import Fraction
from numbers import Real
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from votekit._df_schema import RANKING_COLUMN_PREFIX
from votekit.constants import END_OF_RANKING_SET
from votekit.exceptions import TransformContractError
from votekit.pref_profile.profile_matrix import RankMatrix, ScoreMatrix
from votekit.pref_profile.profile_matrix.rank_matrix import _concatenate_rank_matrices
from votekit.pref_profile.profile_matrix.shared_data import _concatenate_voter_weight_data
from votekit.pref_profile.profile_transform.protocol import (
    MatrixT,
    ProfileTransform,
    RowTransformFn,
)
from votekit.pref_profile.profile_transform.transform_utils import (
    _decode_rows,
    _int_weights,
    _read_only_rows,
    _with_added_candidates,
)
from votekit.types import Candidate
from votekit.utils import sort_candidates_pseudo_lexicographically

# Selecting the voters to change, each with a per-voter probability


def _validate_probability(probability: Any, source: str, row: NDArray | None = None) -> float:
    """
    Checks that a probability is a real number in [0, 1] and returns it as a float.

    Args:
        probability (Any): Probability to check.
        source (str): Where the probability came from, for error messages.
        row (NDArray, optional): Row the probability was computed from, shown in error messages.
            Defaults to None.

    Returns:
        float: The probability.

    Raises:
        TypeError: The probability is a boolean or not a real number.
        ValueError: The probability lies outside [0, 1].
    """
    if isinstance(probability, (bool, np.bool_)) or not isinstance(probability, Real):
        label = source if row is None else f"{source}({row})"
        raise TypeError(f"{label} must be a real number, received {type(probability)}.")

    value = float(probability)
    if not 0.0 <= value <= 1.0:
        label = source if row is None else f"{source}({row})"
        raise ValueError(f"{label} must lie in [0, 1], received {probability}.")
    return value


def _evaluate_row_probabilities(
    probability_fn: Callable[[NDArray], float], matrix: RankMatrix | ScoreMatrix
) -> NDArray[np.float64]:
    """
    Evaluates a probability function once per row, on a read-only array of the row's ballot.

    Args:
        probability_fn (Callable[[NDArray], float]): Probability that each voter of a row is
            selected, given the row's ballot.
        matrix (RankMatrix | ScoreMatrix): Matrix of the profile.

    Returns:
        NDArray[np.float64]: Probability of each row.

    Raises:
        TypeError: A probability is not a real number.
        ValueError: A probability lies outside [0, 1].
    """
    probabilities = []
    for row in _read_only_rows(matrix):
        row_probability = probability_fn(row)
        probabilities.append(_validate_probability(row_probability, "per_voter_probability", row))
    return np.asarray(probabilities, dtype=np.float64)


def _sample_selected_weights(
    matrix: RankMatrix | ScoreMatrix,
    per_voter_probability: float | Callable[[NDArray], float],
    rng: np.random.Generator,
) -> NDArray[np.intp]:
    """
    Draws how many voters of each row are selected, each independently with the row's probability.

    Each unit of weight is one voter, so a row of weight w has Binomial(w, p) voters selected. The
    result is the bin mask for ``matrix._apply_bin_mask``, which bins the selected weight out of
    each row.

    The matrix must already satisfy ``_validate_one_voter_per_unit_weight``, and a float
    ``per_voter_probability`` must already be validated.

    Args:
        matrix (RankMatrix | ScoreMatrix): Matrix of the profile.
        per_voter_probability (float | Callable[[NDArray], float]): Probability that each voter is
            selected, or a function evaluated once per row of the matrix.
        rng (np.random.Generator): Generator for sampling voters.

    Returns:
        NDArray[np.intp]: Number of selected voters in each row.

    Raises:
        TypeError: A probability function returns a value that is not a real number.
        ValueError: A probability function returns a value outside [0, 1].
    """
    if isinstance(per_voter_probability, (int, float)):
        probabilities = np.full(len(matrix), per_voter_probability)
    else:
        probabilities = _evaluate_row_probabilities(per_voter_probability, matrix)

    return rng.binomial(_int_weights(matrix.voter_weight_data), probabilities)


# map: applying a row transform function to each row


MapMode = Literal["row", "voter"]


def _validate_mode(mode: Any) -> None:
    """
    Validate the ``mode`` of ``map()``.

    Args:
        mode (Any): Whether to call ``row_transform_fn`` once per row or once per voter.

    Raises:
        ValueError: ``mode`` is not "row" or "voter".
    """
    if mode not in ("row", "voter"):
        raise ValueError(f'mode must be "row" or "voter", received {mode!r}.')


def _validate_row_transform_fn(row_transform_fn: Any) -> None:
    """
    Validate the ``row_transform_fn`` of ``map()``.

    Args:
        row_transform_fn (Any): Function to apply to one ballot at a time.

    Raises:
        TransformContractError: ``row_transform_fn`` is a class, a ProfileTransform, or not
            callable.
    """
    if isinstance(row_transform_fn, type):
        raise TransformContractError(
            f"row_transform_fn was given the class {row_transform_fn.__name__}, not a function."
        )
    if isinstance(row_transform_fn, ProfileTransform) and not callable(row_transform_fn):
        raise TransformContractError(
            "row_transform_fn was given a ProfileTransform. Pass it to transform instead of map."
        )
    if not callable(row_transform_fn):
        raise TransformContractError(
            "row_transform_fn must be a function that receives a ranking and a RNG object, received"
            f" {row_transform_fn}."
        )


def _map_matrix(
    row_transform_fn: RowTransformFn[Any, Any], matrix: MatrixT, rng: np.random.Generator
) -> MatrixT:
    """
    Applies a row function to each row of the matrix.

    Rows are decoded once for the whole matrix and the results are encoded once. Each row keeps its
    weight and voter set.

    A returned ranking may have any number of positions: the result is as wide as the longest
    returned ranking, and shorter rankings are padded with ``frozenset({"~"})``. A returned
    candidate that the matrix lacks is added to its candidates.

    Args:
        row_transform_fn (RowTransformFn): Function applied to one ballot at a time.
        matrix (RankMatrix | ScoreMatrix): Matrix of the selected rows.
        rng (np.random.Generator): Generator passed to every call of ``row_transform_fn``.

    Returns:
        RankMatrix | ScoreMatrix: Matrix of the returned ballots.

    Raises:
        TransformContractError: A returned ballot cannot be built into a matrix.
    """
    mapped_rows = [
        _validate_row(row_transform_fn(row, rng), matrix, row_idx)
        for row_idx, row in enumerate(_decode_rows(matrix))
    ]
    return _build_matrix(mapped_rows, matrix)


def _validate_row(
    mapped_row: Any, matrix: RankMatrix | ScoreMatrix, source_row_idx: int
) -> NDArray:
    """
    Converts one row returned by a row function to a 1D array and checks it can be stored.

    A sequence, such as a list or tuple, is converted with ``np.asarray``. Ranking entries may be
    sets or frozensets, and sets are converted to frozensets. Score entries must be real numbers,
    one per score column.

    Errors show the returned row and the original ballot.

    Args:
        mapped_row (Any): Row returned by the transform function.
        matrix (RankMatrix | ScoreMatrix): Matrix the function's input was decoded from.
        source_row_idx (int): Row index in ``matrix`` of the input row.

    Returns:
        NDArray: The row as a 1D array.

    Raises:
        TransformContractError: The row is not a 1D array or sequence, a ranking row holds
            entries other than sets or frozensets, uses "~" inside a candidate set, or pads
            before a ranked position, or a score row does not have one non-negative number per
            score column.
    """
    if not isinstance(mapped_row, (np.ndarray, Sequence)):
        if isinstance(matrix, RankMatrix):
            ballot = matrix.get_ranking(source_row_idx)
        else:
            ballot = matrix.get_scores(source_row_idx)
        raise TransformContractError(
            "row_transform_fn must return a 1D array or a sequence, such as a list or tuple."
            f" Returned {mapped_row} for the ballot {ballot}."
        )
    try:
        row = (
            np.asarray(mapped_row, dtype=object)
            if isinstance(matrix, RankMatrix)
            else np.asarray(mapped_row)
        )
    except ValueError as error:
        if isinstance(matrix, RankMatrix):
            ballot = matrix.get_ranking(source_row_idx)
        else:
            ballot = matrix.get_scores(source_row_idx)
        raise TransformContractError(
            "row_transform_fn must return a 1D array or a sequence, such as a list or tuple."
            f" Returned {mapped_row} for the ballot {ballot}."
        ) from error

    if isinstance(matrix, RankMatrix):
        if row.ndim != 1:
            raise TransformContractError(
                "row_transform_fn must return a 1D array, one entry per ranking position."
                f" Returned {mapped_row} for the ballot {matrix.get_ranking(source_row_idx)}."
            )
        seen_padding = False
        for position, cand_set in enumerate(row):
            if not isinstance(cand_set, (set, frozenset)):
                raise TransformContractError(
                    "row_transform_fn must return an array of sets or frozensets of candidates,"
                    " padding the end with frozenset({'~'})."
                    f" Returned {mapped_row} for the ballot {matrix.get_ranking(source_row_idx)}."
                )
            if isinstance(cand_set, set):
                row[position] = frozenset(cand_set)
            if "~" in row[position] and row[position] != END_OF_RANKING_SET:
                raise TransformContractError(
                    "row_transform_fn may only use '~' as the padding frozenset({'~'}), not inside"
                    " a candidate set."
                    f" Returned {mapped_row} for the ballot {matrix.get_ranking(source_row_idx)}."
                )

            if row[position] == END_OF_RANKING_SET:
                seen_padding = True
            elif seen_padding:
                raise TransformContractError(
                    "row_transform_fn must put frozenset({'~'}) only at the end of a ranking."
                    f" Returned {mapped_row} for the ballot {matrix.get_ranking(source_row_idx)}."
                )
        return row

    # ScoreMatrix-specific errors
    width = len(matrix.score_columns)
    if row.shape != (width,):
        raise TransformContractError(
            f"row_transform_fn must return a 1D array of length {width}, one entry per score"
            f" column. Returned {mapped_row} for the ballot {matrix.get_scores(source_row_idx)}."
        )
    if row.dtype.kind not in "iuf":
        raise TransformContractError(
            "row_transform_fn returned a non-numeric score."
            f" Returned {mapped_row} for the ballot {matrix.get_scores(source_row_idx)}."
        )
    if np.any(row < 0):
        raise TransformContractError(
            "row_transform_fn returned a negative score."
            f" Returned {mapped_row} for the ballot {matrix.get_scores(source_row_idx)}."
        )
    return row


def _build_matrix(rows: list[NDArray], matrix: MatrixT) -> MatrixT:
    """
    Encodes validated rows as a matrix of the same kind, adding any new candidates.

    The matrix constructor validates the result, which enforces that the returned ballots form a
    valid matrix.

    Args:
        rows (list[NDArray]): One validated row per row of ``matrix``.
        matrix (RankMatrix | ScoreMatrix): Matrix the rows were decoded from, whose weights and
            voter sets the rows keep.

    Returns:
        RankMatrix | ScoreMatrix: Matrix of the rows. Rankings are as wide as the longest
            returned row, with shorter rows padded. Candidates are ``matrix``'s, followed by new
            candidates in sorted order.
    """
    if isinstance(matrix, RankMatrix):
        width = max(map(len, rows), default=matrix.num_ranking_positions)
        cand_set_array = np.full((len(rows), width), END_OF_RANKING_SET, dtype=object)
        for row_idx, row in enumerate(rows):
            cand_set_array[row_idx, : len(row)] = row

        cands_ranked = frozenset().union(*set(cand_set_array.ravel())) - END_OF_RANKING_SET
        return type(matrix).from_array(
            cand_set_array,
            voter_weight_data=matrix.voter_weight_data,
            ranking_columns=[f"{RANKING_COLUMN_PREFIX}{i + 1}" for i in range(width)],
            candidates=_with_added_candidates(
                matrix.candidates, sort_candidates_pseudo_lexicographically(cands_ranked)
            ),
        )

    scores = np.array(rows, dtype=np.float64).reshape(len(rows), len(matrix.score_columns))
    scored_columns = np.flatnonzero(
        np.any(scores[matrix.voter_weight_data.weights > 0] > 0, axis=0)
    )
    cands_scored = {matrix.score_columns[column] for column in scored_columns}
    return type(matrix)(
        voter_weight_data=matrix.voter_weight_data,
        scores=scores,
        score_columns=matrix.score_columns,
        candidates=_with_added_candidates(
            matrix.candidates, sort_candidates_pseudo_lexicographically(cands_scored)
        ),
    )


# transform: applying a ProfileTransform to the selected rows


def _validate_profile_transform(profile_transform: Any) -> None:
    """
    Checks that ``profile_transform`` is a ProfileTransform instance.

    Args:
        profile_transform (Any): Object passed to ``transform()``.

    Raises:
        TransformContractError: ``profile_transform`` is a class, or not a ProfileTransform.
    """
    if isinstance(profile_transform, type):
        raise TransformContractError(
            f"profile_transform was given the class {profile_transform.__name__}, not an"
            f" instance. Did you mean to construct it like {profile_transform.__name__}(...)?"
        )
    if not isinstance(profile_transform, ProfileTransform):
        raise TransformContractError(
            "profile_transform must be a ProfileTransform instance, with a transform_matrix"
            f" method, received {profile_transform}. To apply a function to one ballot at a"
            " time, use map."
        )


def _apply_profile_transform(
    profile_transform: ProfileTransform[MatrixT],
    selected_matrix: MatrixT,
    rng: np.random.Generator,
) -> MatrixT:
    """
    Applies a ProfileTransform to the matrix of selected voters.

    Args:
        profile_transform (ProfileTransform): Transform to apply to matrix.
        selected_matrix (RankMatrix | ScoreMatrix): Matrix of the selected rows, each weighted by
            its number of selected voters.
        rng (np.random.Generator): Generator passed to the transform.

    Returns:
        RankMatrix | ScoreMatrix: Perturbed voters.

    Raises:
        TransformContractError: The transform does not return a matrix of the same type.
    """
    result = profile_transform.transform_matrix(selected_matrix, rng)
    if not isinstance(result, type(selected_matrix)):
        raise TransformContractError(
            f"profile_transform must return a {type(selected_matrix).__name__}, got {type(result)}."
        )
    return result


# Building the result from the transformed and unselected rows


def _as_fraction_weights(matrix: MatrixT) -> MatrixT:
    """
    Stores the matrix's weights as Fractions, converting each weight exactly.

    Args:
        matrix (RankMatrix | ScoreMatrix): Matrix with float64 or Fraction weights.

    Returns:
        RankMatrix | ScoreMatrix: The same rows with Fraction weights.
    """
    fractions = np.empty(len(matrix), dtype=object)
    fractions[:] = [Fraction(weight) for weight in matrix.voter_weight_data.weights]
    return replace(matrix, voter_weight_data=matrix.voter_weight_data.with_weights(fractions))


def _combine_rank_matrices(
    unselected_matrix: RankMatrix | None,
    transformed_matrix: RankMatrix,
    candidates: tuple[Candidate, ...],
) -> RankMatrix:
    """
    Fits the transformed rows to the profile, under the unselected rows if there are any.

    ``unselected_matrix`` is None when every voter was selected, s only the transformed rows are
    kept. The matrices are the truth for the ranking length. The max_ranking_length is the max
    number of columns amongst the matrices unless the max_candidates_ranked on any ballot is
    greater. The transformed matrix's new candidates are added to ``candidates``.

    Args:
        unselected_matrix (RankMatrix | None): Unselected voters, or None if every voter was
            selected.
        transformed_matrix (RankMatrix): Transformed voters.
        candidates (tuple[Candidate, ...]): Candidates of the profile.

    Returns:
        RankMatrix: Unselected rows, if any, followed by transformed rows, with reset labels.
    """
    if unselected_matrix is None:
        matrices = [transformed_matrix]
    else:
        matrices = [unselected_matrix, transformed_matrix]

    width = max(matrix.num_ranking_positions for matrix in matrices)
    if transformed_matrix.has_ties().any():
        width = max(width, transformed_matrix.max_candidates_ranked())

    return _concatenate_rank_matrices(
        matrices, _with_added_candidates(candidates, transformed_matrix.candidates), width
    )


def _combine_score_matrices(
    unselected_matrix: ScoreMatrix | None,
    transformed_matrix: ScoreMatrix,
    candidates: tuple[Candidate, ...],
) -> ScoreMatrix:
    """
    Fits the transformed rows to the profile, under the unselected rows if there are any.

    ``unselected_matrix`` is None when every voter was selected, so there is nothing to stack and
    only the transformed rows are kept. The matrices are the truth for the score columns: the
    result's columns are the union of the parts' columns, in order of first appearance, with NaN
    (unscored) where a part lacks a column. Candidates are the profile's metadata, so the
    transformed matrix's new candidates are added to ``candidates`` and none are removed.

    Args:
        unselected_matrix (ScoreMatrix | None): Unselected voters, or None if every voter was
            selected.
        transformed_matrix (ScoreMatrix): Transformed voters.
        candidates (tuple[Candidate, ...]): Candidates of the profile.

    Returns:
        ScoreMatrix: Unselected rows, if any, followed by transformed rows, with reset labels.
    """
    if unselected_matrix is None:
        matrices = [transformed_matrix]
    else:
        matrices = [unselected_matrix, transformed_matrix]

    score_columns = tuple(
        dict.fromkeys(cand for matrix in matrices for cand in matrix.score_columns)
    )
    column_positions = {cand: column for column, cand in enumerate(score_columns)}

    scores = np.full((sum(map(len, matrices)), len(score_columns)), np.nan)
    start = 0
    for matrix in matrices:
        end = start + len(matrix)
        positions = [column_positions[cand] for cand in matrix.score_columns]
        scores[start:end, positions] = matrix.scores
        start = end

    return ScoreMatrix(
        voter_weight_data=_concatenate_voter_weight_data(
            [matrix.voter_weight_data for matrix in matrices]
        ),
        scores=scores,
        score_columns=score_columns,
        candidates=_with_added_candidates(candidates, transformed_matrix.candidates),
    )
