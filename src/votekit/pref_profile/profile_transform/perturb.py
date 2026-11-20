from collections.abc import Callable
from dataclasses import replace
from fractions import Fraction
from numbers import Real
from typing import Any

import numpy as np
from numpy.typing import NDArray

from votekit.exceptions import ProfileError, TransformContractError
from votekit.pref_profile.profile_matrix import RankMatrix, ScoreMatrix, WeightKind
from votekit.pref_profile.profile_matrix.shared_data import (
    ProfileVoterWeightData,
    _concatenate_voter_weight_data,
)
from votekit.pref_profile.profile_transform.protocol import MatrixT, ProfileTransform
from votekit.pref_profile.profile_transform.transform_utils import (
    _ballot_label,
    _integral_weights,
    _read_only_rows,
)
from votekit.types import Candidate


def _perturb_matrix(
    matrix: MatrixT,
    transform: ProfileTransform[Any],
    per_voter_probability: float | Callable[[NDArray], float],
    rng_seed: int | None,
) -> tuple[MatrixT, MatrixT]:
    """
    Selects voters and transforms their ballots.

    Each unit of weight is selected with ``per_voter_probability``, each row's voter set is split
    between its selected and remaining voters, and the selected rows are transformed.

    Args:
        matrix (RankMatrix | ScoreMatrix): Nonempty matrix of the profile to perturb.
        transform (ProfileTransform): Transform lifted by ``_lift_to_profile_transform``.
        per_voter_probability (float | Callable[[NDArray], float]): Probability that each voter is
            perturbed, or a function evaluated once per row on a read-only array of its ballot.
        rng_seed (int, optional): Seed for voter selection and voter set sampling.

    Returns:
        tuple[MatrixT, MatrixT]: Unperturbed rows with their remaining weight, and the perturbed
            rows. Weights are float64; the profile method restores the profile's weight kind.

    Raises:
        TypeError: A probability is not a real number.
        ValueError: A probability lies outside [0, 1].
        ValueError: The profile has non-integral weights.
        ProfileError: A row's non-empty voter set does not match its weight.
        TransformContractError: The transform does not return a matrix of the same kind, does
            not conserve weight, or changes the ranking or score columns.
    """
    rng = np.random.default_rng(rng_seed)
    matrix_data = matrix.voter_weight_data
    total_weights = _integral_weights(matrix_data)
    _check_voter_sets_match_weights(matrix_data, total_weights)
    probabilities = _row_probabilities(per_voter_probability, matrix)
    weight_selected = rng.binomial(total_weights, probabilities)

    selected_voter_sets, remaining_voter_sets = _split_voter_sets(
        matrix_data.voter_sets, total_weights, weight_selected, rng
    )

    remaining_rows = (weight_selected < total_weights) | (total_weights == 0)
    selected_rows = weight_selected > 0
    unperturbed_matrix = _mask_and_reweight(
        matrix, remaining_rows, total_weights - weight_selected, remaining_voter_sets
    )
    selected_matrix = _mask_and_reweight(
        matrix, selected_rows, weight_selected, selected_voter_sets
    )

    return unperturbed_matrix, _apply_profile_transform(transform, selected_matrix)


def _as_fraction_weights(matrix: MatrixT) -> MatrixT:
    """
    Stores the matrix's weights as Fractions, converting each float64 weight exactly.

    Args:
        matrix (RankMatrix | ScoreMatrix): Matrix with float64 weights.

    Returns:
        RankMatrix | ScoreMatrix: The same rows with Fraction weights.
    """
    fractions = np.empty(len(matrix), dtype=object)
    fractions[:] = [Fraction(weight) for weight in matrix.voter_weight_data.weights]
    return replace(matrix, voter_weight_data=matrix.voter_weight_data.with_weights(fractions))


def _row_probabilities(
    per_voter_probability: float | Callable[[NDArray], float], matrix: RankMatrix | ScoreMatrix
) -> NDArray[np.float64]:
    """
    The probability that each voter of each row is perturbed.

    Args:
        per_voter_probability (float | Callable[[NDArray], float]): Probability for all voters,
            or a function evaluated once per row on a read-only array of its ballot.
        matrix (RankMatrix | ScoreMatrix): Matrix of the profile to perturb.

    Returns:
        NDArray[np.float64]: Probability of each row.

    Raises:
        TypeError: A probability is not a real number.
        ValueError: A probability lies outside [0, 1].
    """
    if callable(per_voter_probability):
        probabilities = []
        for row in _read_only_rows(matrix):
            row_probability = per_voter_probability(row)
            validated = _validate_probability(row_probability, "per_voter_probability", row)
            probabilities.append(validated)
        return np.asarray(probabilities, dtype=np.float64)

    probability = _validate_probability(per_voter_probability, "per_voter_probability")
    return np.full(len(matrix), probability)


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


def _check_voter_sets_match_weights(
    matrix_data: ProfileVoterWeightData, weights: NDArray[np.int64]
) -> None:
    """
    Checks that every non-empty voter set has one voter per unit of weight.

    Every row is checked before any voters are selected, so whether a profile can be perturbed
    does not depend on the probability or the seed.

    Args:
        matrix_data (ProfileVoterWeightData): Shared ballot data of the profile's matrix.
        weights (NDArray[np.int64]): Integer weight of each row.

    Raises:
        ProfileError: A row's non-empty voter set does not match its weight.
    """
    if matrix_data.voter_sets is None:
        return

    voter_set_sizes = np.fromiter(
        map(len, matrix_data.voter_sets), dtype=np.int64, count=len(matrix_data.voter_sets)
    )
    mismatched = np.flatnonzero((voter_set_sizes > 0) & (voter_set_sizes != weights))
    if len(mismatched):
        row_idx = mismatched[0]
        raise ProfileError(
            "Cannot perturb a profile whose voter sets do not match their weights. Ballot Index"
            f" {_ballot_label(matrix_data, row_idx)} has a voter set of size"
            f" {voter_set_sizes[row_idx]} but weight {weights[row_idx]}."
        )


def _split_voter_sets(
    voter_sets: NDArray[np.object_] | None,
    weights: NDArray[np.int64],
    weight_selected: NDArray[np.int64],
    rng: np.random.Generator,
) -> tuple[NDArray[np.object_] | None, NDArray[np.object_] | None]:
    """
    Splits each row's voter set between its selected and remaining voters.

    Only rows with a non-empty voter set are split; rows without voters keep their empty set on
    both sides. Voters are sampled without replacement only when part of a row is selected, and
    are sorted before sampling so a seed gives the same split across runs. Voter set sizes must
    already match their weights.

    Args:
        voter_sets (Optional[NDArray[np.object_]]): Voter set of each row, or None when the
            profile has no voter data.
        weights (NDArray[np.int64]): Integer weight of each row.
        weight_selected (NDArray[np.int64]): Weight selected from each row.
        rng (np.random.Generator): Generator for sampling voters.

    Returns:
        tuple[Optional[NDArray[np.object_]], Optional[NDArray[np.object_]]]: Selected voters and
            remaining voters of each row. Both are None when the profile has no voter data.
    """
    if voter_sets is None:
        return None, None

    selected_voter_set = voter_sets.copy()
    remaining_voter_set = voter_sets.copy()
    has_voters = np.fromiter(map(len, voter_sets), dtype=np.int64, count=len(voter_sets)) > 0
    for row_idx in np.flatnonzero(has_voters):
        voter_set, perturb_weight = voter_sets[row_idx], weight_selected[row_idx]
        if perturb_weight == 0:
            selected_voter_set[row_idx] = frozenset()
        elif perturb_weight == weights[row_idx]:
            remaining_voter_set[row_idx] = frozenset()
        else:
            voters = sorted(voter_set, key=str)
            selected_voters = set(
                rng.choice(len(voters), size=perturb_weight, replace=False).tolist()
            )
            selected_voter_set[row_idx] = frozenset(
                v for i, v in enumerate(voters) if i in selected_voters
            )
            remaining_voter_set[row_idx] = frozenset(
                v for i, v in enumerate(voters) if i not in selected_voters
            )

    return selected_voter_set, remaining_voter_set


def _mask_and_reweight(
    matrix: MatrixT,
    mask: NDArray[np.bool_],
    weights: NDArray[np.int64],
    voter_sets: NDArray[np.object_] | None,
) -> MatrixT:
    """
    Masks the matrix's rows and replaces their weights and voter sets.

    Rows keep the source's Ballot Index labels, so errors can name the profile's ballots. Weights
    are stored as float64; perturb restores the profile's weight kind at the end.

    Args:
        matrix (RankMatrix | ScoreMatrix): Source matrix.
        mask (NDArray[np.bool_]): One flag per row of ``matrix``; rows marked True are kept in
            order.
        weights (NDArray[np.int64]): New integer weight of each row of ``matrix``.
        voter_sets (Optional[NDArray[np.object_]]): New voter set of each row of ``matrix``, or
            None when the profile has no voter data.

    Returns:
        RankMatrix | ScoreMatrix: Matrix of the kept rows with their new weights and voter sets.
    """
    data = matrix.voter_weight_data
    ballot_labels = data.ballot_index if data.ballot_index is not None else np.arange(len(matrix))
    masked_data = ProfileVoterWeightData(
        weights=weights[mask].astype(np.float64),
        voter_sets=None if voter_sets is None else voter_sets[mask],
        weight_kind=WeightKind.FLOAT,
        ballot_index=ballot_labels[mask],
    )
    if isinstance(matrix, RankMatrix):
        return replace(matrix, voter_weight_data=masked_data, rankings=matrix.rankings[mask])
    return replace(matrix, voter_weight_data=masked_data, scores=matrix.scores[mask])


def _apply_profile_transform(
    transform: ProfileTransform[MatrixT], selected_matrix: MatrixT
) -> MatrixT:
    """
    Applies a ProfileTransform to the matrix of selected voters.

    Args:
        transform (ProfileTransform): Transform to apply.
        selected_matrix (RankMatrix | ScoreMatrix): Matrix of the selected rows, each weighted by
            its number of selected voters.

    Returns:
        RankMatrix | ScoreMatrix: Perturbed voters.

    Raises:
        TransformContractError: The transform does not return a matrix of the same kind, does
            not conserve weight, returns non-integral weights, or changes ranking or score columns.
    """
    result = transform.transform_matrix(selected_matrix)
    if not isinstance(result, type(selected_matrix)):
        raise TransformContractError(
            f"transform_fn must return a {type(selected_matrix).__name__}, got {type(result)}."
        )
    received = sum(selected_matrix.voter_weight_data.weights)
    returned = sum(result.voter_weight_data.weights)
    if returned != received:
        raise TransformContractError(
            "transform_fn must conserve weight. Received total weight"
            f" {received}, returned {returned}."
        )
    # each unit of weight is one voter, so the result can be perturbed again
    returned_data = result.voter_weight_data
    non_integral = np.flatnonzero(returned_data.weights % 1 != 0)
    if len(non_integral):
        row_idx = int(non_integral[0])
        where = (
            f"row {row_idx} of the returned matrix"
            if returned_data.ballot_index is None
            else f"Ballot Index {returned_data.ballot_index[row_idx]}"
        )
        raise TransformContractError(
            "transform_fn must return integral weights, one unit of weight per voter. Returned"
            f" weight {returned_data.weights[row_idx]} at {where}."
        )
    if isinstance(selected_matrix, RankMatrix) and isinstance(result, RankMatrix):
        if result.ranking_columns != selected_matrix.ranking_columns:
            raise TransformContractError(
                "transform_fn must keep the ranking columns"
                f" {selected_matrix.ranking_columns}, one per position up to the profile's"
                f" max_ranking_length, returned {result.ranking_columns}."
            )
    if isinstance(selected_matrix, ScoreMatrix) and isinstance(result, ScoreMatrix):
        if result.score_columns != selected_matrix.score_columns:
            raise TransformContractError(
                "transform_fn must keep the score columns"
                f" {selected_matrix.score_columns}, returned {result.score_columns}."
            )
    return result


def _validate_perturbed_rankings(
    perturbed_matrix: RankMatrix, max_ranking_length: int, candidates: tuple[Candidate, ...]
) -> None:
    """
    Checks that perturbed rankings fit the profile they are returned to.

    The ranking columns were already checked against the selected matrix's, so the rankings
    have the profile's max_ranking_length positions.

    Args:
        perturbed_matrix (RankMatrix): Perturbed voters.
        max_ranking_length (int): Maximum ranking length of the profile to perturb.
        candidates (tuple[Candidate, ...]): Candidates of the profile to perturb.

    Raises:
        TransformContractError: The rankings rank more candidates than the profile's
            max_ranking_length allows, or rank candidates outside the profile's candidates.
    """

    if perturbed_matrix.has_ties().any():
        max_ranked = perturbed_matrix.max_candidates_ranked()
        if max_ranked > max_ranking_length:
            raise TransformContractError(
                f"transform_fn returned a ranking of {max_ranked} candidates, more than the"
                f" profile's max_ranking_length {max_ranking_length}."
            )

    undeclared_cands = perturbed_matrix.get_candidates_cast() - frozenset(candidates)
    if undeclared_cands:
        raise TransformContractError(
            f"transform_fn returned candidates {set(undeclared_cands)} not in the profile."
            " Declare them in the profile's candidates before perturbing."
        )


def _validate_perturbed_scores(
    perturbed_matrix: ScoreMatrix, candidates: tuple[Candidate, ...]
) -> None:
    """
    Checks that perturbed scores only score the profile's candidates.

    Args:
        perturbed_matrix (ScoreMatrix): Perturbed voters.
        candidates (tuple[Candidate, ...]): Candidates of the profile to perturb.

    Raises:
        TransformContractError: A candidate outside the profile's candidates is scored.
    """
    undeclared_cands = perturbed_matrix.get_candidates_cast() - frozenset(candidates)
    if undeclared_cands:
        raise TransformContractError(
            f"transform_fn returned scores for candidates {set(undeclared_cands)} not in the"
            " profile. Declare them in the profile's candidates before perturbing."
        )


def _stack_score_matrices(
    unperturbed_matrix: ScoreMatrix,
    perturbed_matrix: ScoreMatrix,
    candidates: tuple[Candidate, ...],
) -> ScoreMatrix:
    """
    Stacks unperturbed and perturbed voters, which share the source's score columns.

    Args:
        unperturbed_matrix (ScoreMatrix): Unperturbed voters.
        perturbed_matrix (ScoreMatrix): Perturbed voters, with the same score columns.
        candidates (tuple[Candidate, ...]): Candidates of the profile to perturb.

    Returns:
        ScoreMatrix: Unperturbed rows followed by perturbed rows, with reset labels.
    """
    return ScoreMatrix(
        voter_weight_data=_concatenate_voter_weight_data(
            [unperturbed_matrix.voter_weight_data, perturbed_matrix.voter_weight_data]
        ),
        scores=np.concatenate([unperturbed_matrix.scores, perturbed_matrix.scores]),
        score_columns=unperturbed_matrix.score_columns,
        candidates=candidates,
    )
