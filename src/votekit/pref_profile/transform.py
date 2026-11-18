from collections.abc import Callable
from dataclasses import dataclass, field, replace
from fractions import Fraction
from numbers import Real
from typing import (
    Any,
    Protocol,
    TypeVar,
    overload,
    runtime_checkable,
)

import numpy as np
from numpy.typing import NDArray

from votekit.constants import (
    EMPTY_SET_ID,
    END_OF_RANKING_ID,
    END_OF_RANKING_SET,
)
from votekit.exceptions import TransformContractError
from votekit.pref_profile import ProfileError, RankProfile, ScoreProfile
from votekit.pref_profile.profile_matrix import RankMatrix, ScoreMatrix, WeightKind
from votekit.pref_profile.profile_matrix.rank_matrix import _concatenate_rank_matrices
from votekit.pref_profile.profile_matrix.shared_data import (
    ProfileVoterWeightData,
    _concatenate_voter_weight_data,
)
from votekit.types import Candidate

RankingRow = NDArray[np.object_]
ScoresRow = NDArray[np.float64]

MatrixT = TypeVar("MatrixT", RankMatrix, ScoreMatrix)
ProfileT = TypeVar("ProfileT", RankProfile, ScoreProfile)


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
    A per-voter transform function lifted to the ProfileTransform contract.

    ``transform_matrix`` splits each row of the selected voters' matrix into one row per unit of
    weight, calls the function once per unit on a writeable copy of that unit's ballot, checks
    each result, and constructs a matrix of the same kind from the results. The output has one row
    of weight 1 per unit of input weight, so total weight is conserved.

    The RankMatrix ballots are transformed from IDs to candidate sets for the transform_fn.
    """

    transform_vote_fn: Callable[[NDArray], Any]

    def transform_matrix(self, matrix: MatrixT) -> MatrixT:
        unit_row_labels, unit_data = _expand_to_units(matrix)

        ballot_index = matrix.voter_weight_data.ballot_index
        unit_rows = _decode_rows(matrix, unit_row_labels)

        for unit_idx, source_row_label in enumerate(unit_row_labels):
            perturbed_row = self.transform_vote_fn(unit_rows[unit_idx])
            ballot_label = (
                source_row_label if ballot_index is None else ballot_index[source_row_label]
            )
            unit_rows[unit_idx] = _check_row(perturbed_row, matrix, ballot_label)

        if isinstance(matrix, RankMatrix):
            _check_declared_candidates(unit_rows, unit_row_labels, matrix)
            return RankMatrix.from_array(
                unit_rows,
                voter_weight_data=unit_data,
                ranking_columns=matrix.ranking_columns,
                candidates=matrix.candidates,
            )
        else:
            return ScoreMatrix(
                voter_weight_data=unit_data,
                scores=unit_rows,
                score_columns=matrix.score_columns,
                candidates=matrix.candidates,
            )


def _expand_to_units(
    matrix: RankMatrix | ScoreMatrix,
) -> tuple[NDArray[np.intp], ProfileVoterWeightData]:
    """
    Splits each row of the matrix into one row per unit of weight.

    If a row has weight greater than 1 and a non-empty voter set, the set must be split. Each unit
    is assigned one voter from its row's voter set, in sorted order. Voters were already sampled
    without replacement when the profile was masked. Else, the voter set is left untouched.

    A non-empty voter set must have one voter per unit of weight. ``perturb`` checks this for
    every row when masking, before any rows are selected, so it is not checked again here.

    Args:
        matrix (RankMatrix | ScoreMatrix): Matrix with integral weights.

    Returns:
        tuple[NDArray[np.intp], ProfileVoterWeightData]: Source row label of each unit, and
            voter weight data with weight 1 and at most one voter per unit.

    Raises:
        ValueError: The matrix has non-integral weights. The row cannot be split per unit of weight.
    """
    matrix_data = matrix.voter_weight_data
    weights = _integral_weights(matrix_data)
    # every row is already one voter, and a non-empty voter set already has its one voter
    if np.all(weights == 1):
        return np.arange(len(matrix)), matrix_data
    unit_row_labels = np.repeat(np.arange(len(matrix)), weights)

    unit_voter_sets = None
    if matrix_data.voter_sets is not None:
        unit_voter_sets = np.empty(len(unit_row_labels), dtype=object)
        unit_idx = 0
        for voter_set, weight in zip(matrix_data.voter_sets, weights):
            end_unit = unit_idx + weight
            if voter_set == frozenset():
                unit_voter_sets[unit_idx:end_unit] = voter_set
            else:
                unit_voter_sets[unit_idx:end_unit] = [
                    frozenset({voter}) for voter in sorted(voter_set, key=str)
                ]
            unit_idx = end_unit

    # weights stay float64 until perturb restores the profile's weight kind at the end
    unit_matrix_data = ProfileVoterWeightData(
        weights=np.ones(len(unit_row_labels), dtype=np.float64),
        voter_sets=unit_voter_sets,
        weight_kind=WeightKind.FLOAT,
    )
    return unit_row_labels, unit_matrix_data


def _decode_rows(matrix: RankMatrix | ScoreMatrix, rows: NDArray[np.intp]) -> NDArray:
    """
    Writeable copy of the given rows: candidate sets for rankings, floats for scores.

    Args:
        matrix (RankMatrix | ScoreMatrix): Source matrix.
        rows (NDArray[np.intp]): Rows to decode, repeats allowed.

    Returns:
        NDArray: One decoded row per entry of ``rows``.
    """
    if isinstance(matrix, RankMatrix):
        return matrix.get_ranking_array(rows)
    return matrix.scores[rows]


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


def _check_row(perturbed_row: Any, matrix: RankMatrix | ScoreMatrix, ballot_label: Any) -> NDArray:
    """
    Converts one row returned by a per-voter transform function to a 1D array and checks it.

    A list or tuple is converted with ``np.asarray``; an array is used as is, without a copy.
    Ranking entries may be sets or frozensets, and sets are converted to frozensets as
    ``RankProfile`` does. Score entries must be real numbers.

    Args:
        perturbed_row (Any): Row returned by the function.
        matrix (RankMatrix | ScoreMatrix): Matrix the function's input was decoded from.
        ballot_label (Any): Ballot Index of the input row in the profile, as shown by
            ``profile.to_df()``.

    Returns:
        NDArray: The row as a 1D array.

    Raises:
        TransformContractError: The row is not a 1D array of the matrix's width, a ranking row
            holds entries other than sets or frozensets, or a score row holds non-numeric or
            negative scores.
    """
    row = np.asarray(perturbed_row)

    width = (
        matrix.num_ranking_positions
        if isinstance(matrix, RankMatrix)
        else len(matrix.score_columns)
    )
    if row.shape != (width,):
        raise TransformContractError(
            f"transform_fn must return a 1D array of length {width}, one entry per"
            f" {'ranking position' if isinstance(matrix, RankMatrix) else 'score column'}."
            f" Returned {perturbed_row} for Ballot Index {ballot_label}."
        )

    if isinstance(matrix, RankMatrix):
        for position, cand_set in enumerate(row):
            if not isinstance(cand_set, (set, frozenset)):
                raise TransformContractError(
                    "transform_fn must return an array of sets or frozensets of candidates,"
                    " padding the end with frozenset({'~'}). Returned"
                    f" {perturbed_row} for Ballot Index {ballot_label}."
                )
            # as RankProfile does, sets become frozensets; frozensets are kept without a copy
            if isinstance(cand_set, set):
                row[position] = frozenset(cand_set)
        return row

    # ScoreMatrix
    if row.dtype.kind not in "iuf":
        raise TransformContractError(
            f"transform_fn returned a non-numeric score. Returned {perturbed_row} for Ballot"
            f" Index {ballot_label}."
        )
    if np.any(row < 0):
        raise TransformContractError(
            f"transform_fn returned a negative score. Returned {perturbed_row} for Ballot Index"
            f" {ballot_label}."
        )
    return row


def _check_declared_candidates(
    unit_rows: NDArray[np.object_], unit_row_labels: NDArray[np.intp], matrix: RankMatrix
) -> None:
    """
    Checks once, over every perturbed ranking, that only the profile's candidates are ranked.

    Args:
        unit_rows (NDArray[np.object_]): Perturbed rankings of candidate sets, one row per unit.
        unit_row_labels (NDArray[np.intp]): Row of ``matrix`` each unit was decoded from.
        matrix (RankMatrix): Matrix the units were decoded from.

    Raises:
        TransformContractError: A ranking ranks a candidate outside the profile's candidates.
    """
    # a Python set keeps one copy of each distinct candidate set, so the union is over few sets
    distinct_cand_sets = set(unit_rows.ravel())
    cands_cast = frozenset().union(*distinct_cand_sets) - END_OF_RANKING_SET
    undeclared_cands = cands_cast - frozenset(matrix.candidates)
    if not undeclared_cands:
        return

    # only on failure: find the first ranking with an undeclared candidate to name its ballot
    unit_idx = next(
        idx
        for idx, row in enumerate(unit_rows)
        if any(cand_set & undeclared_cands for cand_set in row)
    )
    source_row = unit_row_labels[unit_idx]
    ballot_index = matrix.voter_weight_data.ballot_index
    ballot_label = source_row if ballot_index is None else ballot_index[source_row]
    raise TransformContractError(
        f"transform_fn returned candidates {set(undeclared_cands)} not in the profile's"
        " candidates. Declare them in the profile's candidates before perturbing. Returned"
        f" {unit_rows[unit_idx]} for Ballot Index {ballot_label}."
    )


def _lift_to_operation(transform_fn: Any) -> ProfileTransform:
    """
    Lifts the transform function to the ProfileTransform interface.

    A ProfileTransform is returned unchanged. Any other callable is a per-voter transform function
    and is wrapped so that it is called once per unit of weight. Lifting does not run the
    transformation.

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


@overload
def perturb(
    profile: RankProfile,
    transform_fn: ProfileTransform[RankMatrix] | Callable[[RankingRow], RankingRow],
    per_voter_probability: float | Callable[[RankingRow], float] = 1.0,
    *,
    group_ballots: bool = False,
    rng_seed: int | None = None,
) -> RankProfile: ...


@overload
def perturb(
    profile: ScoreProfile,
    transform_fn: ProfileTransform[ScoreMatrix] | Callable[[ScoresRow], ScoresRow],
    per_voter_probability: float | Callable[[ScoresRow], float] = 1.0,
    *,
    group_ballots: bool = False,
    rng_seed: int | None = None,
) -> ScoreProfile: ...


def perturb(
    profile: ProfileT,
    transform_fn: Any,
    per_voter_probability: Any = 1.0,
    *,
    group_ballots: bool = False,
    rng_seed: int | None = None,
) -> ProfileT:
    """
    Perturb voters in the profile, each independently with some probability.

    Each unit of ballot weight is treated as one voter and is selected independently with
    probability p, so a row of weight w has Binomial(w, p) of its weight perturbed. A row with only
    part of its weight selected is split into an unperturbed row with the remaining weight and a
    selected row with the selected weight. Set ``group_ballots`` to regroup the output by ranking
    or scores.

    A ``ProfileTransform`` receives the matrix of selected rows, each weighted by its number of
    selected voters. A per-voter transform function is called once per selected voter, so any
    randomness in it is independent across voters, and each selected voter becomes its own row of
    weight 1.

    A per-voter transform function receives one voter's ballot as a writeable 1D array copy. It
    may modify the array in place and return it, or return a new 1D array, list, or tuple of the
    same length. For a RankProfile, the array holds one frozenset of candidates per ranking
    position, as in the ranking columns of ``profile.to_df()``: ``frozenset()`` marks an explicit
    empty position and ``frozenset({"~"})`` marks trailing unranked positions. Frozensets cannot be
    edited in place, so replace an entry to change it; a replacement may be a set or frozenset,
    and sets are converted to frozensets as ``RankProfile`` does. A function that shortens a
    ranking writes ``frozenset({"~"})`` into the positions it frees. For a ScoreProfile, the array
    holds the voter's scores in the order of the score columns of ``profile.to_df()``, with NaN
    meaning unscored. Keep in mind that score columns can include undeclared candidates beyond
    ``profile.candidates`` that have a score of 0 or NaN.

    Perturbed ballots may only use the profile's candidates. To add a candidate, declare it in the
    profile's candidates before perturbing.

    User functions only receive the ballot, not its weight or row index. To pass extra arguments,
    curry them first, e.g. ``functools.partial(fn, candidate="A")``. If a per-voter transform
    function, a ``ProfileTransform``, or a probability function is random, it is responsible for
    its own randomness: give it an ``np.random.Generator``, not a seed, since re-seeding on each
    call gives every voter the same outcome. ``rng_seed`` only seeds voter selection and voter set
    sampling. A per-voter transform function should not depend on the order of its calls: a row's
    selected voters are assigned to its calls in sorted order.

    If a row has a non-empty voter set, its size must equal the row's weight. Selected voters are
    sampled without replacement from the row's voter set, and the remaining voters stay with the
    unperturbed row.

    Args:
        profile (RankProfile | ScoreProfile): Profile to perturb.
        transform_fn (ProfileTransform | Callable[[NDArray], NDArray]): A ``ProfileTransform``
            operates on the matrix of all selected rows at once and is the fast path for large
            profiles. A per-voter transform function takes one voter's ranking or scores and
            returns the perturbed ones; it is called once per selected voter.
        per_voter_probability (float | Callable[[NDArray], float], optional): Probability that
            each voter is perturbed. A float applies to all voters. A callable is evaluated once
            per row on a read-only array of that row's ballot, in the same form a per-voter
            transform function receives. If the callable is random, the drawn probability is
            shared by all voters in that row, so results can depend on how the profile is grouped.
            Values must lie in [0, 1]. Defaults to 1.0.
        group_ballots (bool, optional): Whether to group the output ballots by ranking or scores,
            summing weights and joining voter sets. Defaults to False.
        rng_seed (int, optional): Seed for voter selection and voter set sampling. Defaults to
            None.

    Returns:
        RankProfile | ScoreProfile: The perturbed profile, including unperturbed voters.

    Raises:
        TypeError: The profile is not a RankProfile or ScoreProfile.
        TypeError: A probability is not a real number.
        ValueError: A probability lies outside [0, 1].
        ValueError: The profile has non-integral weights.
        ProfileError: A row's non-empty voter set does not match its weight.
        TransformContractError: ``transform_fn`` is a class, or neither callable nor a
            ``ProfileTransform``.
        TransformContractError: The transformation returns an invalid ranking or scores, uses a
            candidate not declared on the profile, does not conserve weight, or changes the score
            columns.
    """
    # TODO: remove once its an instance method of the Profile class. Self calls it
    if not isinstance(profile, (RankProfile, ScoreProfile)):
        raise TypeError(f"profile must be a RankProfile or ScoreProfile, received {type(profile)}.")

    # validate functions and lift to the same interface
    transform = _lift_to_operation(transform_fn)

    if len(profile._matrix) == 0:
        return profile

    # mask matrix with probability function
    rng = np.random.default_rng(rng_seed)
    matrix_data = profile._matrix.voter_weight_data
    total_weights = _integral_weights(matrix_data)
    _check_voter_sets_match_weights(matrix_data, total_weights)
    probabilities = _row_probabilities(per_voter_probability, profile._matrix)
    weight_selected = rng.binomial(total_weights, probabilities)

    selected_voter_sets, remaining_voter_sets = _split_voter_sets(
        matrix_data.voter_sets, total_weights, weight_selected, rng
    )

    remaining_rows = (weight_selected == 0) | (weight_selected < total_weights)
    selected_rows = weight_selected > 0
    unperturbed_matrix = _select_rows(
        profile._matrix, remaining_rows, total_weights - weight_selected, remaining_voter_sets
    )
    selected_matrix = _select_rows(
        profile._matrix, selected_rows, weight_selected, selected_voter_sets
    )

    perturbed_matrix = _apply_profile_transform(transform, selected_matrix)

    # combine perturbed and unperturbed matrices
    if isinstance(profile, RankProfile):
        assert isinstance(unperturbed_matrix, RankMatrix) and isinstance(
            perturbed_matrix, RankMatrix
        )
        _validate_perturbed_rankings(perturbed_matrix, profile)
        result_matrix = _concatenate_rank_matrices(
            [unperturbed_matrix, perturbed_matrix], profile.candidates, profile.max_ranking_length
        )
    else:
        assert isinstance(unperturbed_matrix, ScoreMatrix) and isinstance(
            perturbed_matrix, ScoreMatrix
        )
        _validate_perturbed_scores(perturbed_matrix, profile)
        result_matrix = _stack_score_matrices(
            unperturbed_matrix, perturbed_matrix, profile.candidates
        )

    # group matrix if set
    if group_ballots:
        result_matrix = result_matrix.group_ballots()

    if matrix_data.weight_kind is WeightKind.FRACTION:
        result_matrix = _as_fraction_weights(result_matrix)

    # create profile from matrix
    if isinstance(profile, RankProfile):
        assert isinstance(result_matrix, RankMatrix)
        return RankProfile._from_matrix(
            result_matrix, max_ranking_length=profile.max_ranking_length
        )
    assert isinstance(result_matrix, ScoreMatrix)
    return ScoreProfile._from_matrix(result_matrix)


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
    return np.array([int(weight) for weight in data.weights], dtype=np.int64)


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
    if isinstance(matrix, RankMatrix):
        return matrix.with_weights(fractions)
    return ScoreMatrix(
        voter_weight_data=matrix.voter_weight_data.with_weights(fractions),
        scores=matrix.scores,
        score_columns=matrix.score_columns,
        candidates=matrix.candidates,
    )


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
            probability = per_voter_probability(row)
            _validate_probability(probability, f"per_voter_probability({row})")
            probabilities.append(float(probability))
        return np.asarray(probabilities, dtype=np.float64)

    _validate_probability(per_voter_probability, "per_voter_probability")
    return np.full(len(matrix), float(per_voter_probability))


def _validate_probability(probability: Any, source: str) -> None:
    if not isinstance(probability, Real):
        raise TypeError(f"{source} must be a real number, received {type(probability)}.")
    if not 0.0 <= float(probability) <= 1.0:
        raise ValueError(f"{source} must lie in [0, 1], received {probability}.")


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
        row_label = mismatched[0]
        raise ProfileError(
            "Cannot perturb a profile whose voter sets do not match their weights. Ballot at index"
            f" {row_label} has a voter set of size {voter_set_sizes[row_label]} but weight"
            f" {weights[row_label]}."
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
    for row_label in np.flatnonzero(has_voters):
        voter_set, perturb_weight = voter_sets[row_label], weight_selected[row_label]
        if perturb_weight == 0:
            selected_voter_set[row_label] = frozenset()
        elif perturb_weight == weights[row_label]:
            remaining_voter_set[row_label] = frozenset()
        else:
            voters = sorted(voter_set, key=str)
            selected_voters = set(
                rng.choice(len(voters), size=perturb_weight, replace=False).tolist()
            )
            selected_voter_set[row_label] = frozenset(
                v for i, v in enumerate(voters) if i in selected_voters
            )
            remaining_voter_set[row_label] = frozenset(
                v for i, v in enumerate(voters) if i not in selected_voters
            )

    return selected_voter_set, remaining_voter_set


def _select_rows(
    matrix: MatrixT,
    mask: NDArray[np.bool_],
    weights: NDArray[np.int64],
    voter_sets: NDArray[np.object_] | None,
) -> MatrixT:
    """
    Masks the matrix's rows and replaces their weights and voter sets.

    Rows keep the source's Ballot Index labels, so errors can name the profile's row. Weights
    are stored as float64; perturb restores the profile's weight kind at the end.

    Args:
        matrix (RankMatrix | ScoreMatrix): Source matrix.
        mask (NDArray[np.bool_]): One flag per source row; True rows are kept in source order.
        weights (NDArray[np.int64]): New integer weight of each source row.
        voter_sets (Optional[NDArray[np.object_]]): New voter set of each source row, or None
            when the profile has no voter data.

    Returns:
        RankMatrix | ScoreMatrix: Matrix of the kept rows with their new weights and voter sets.
    """
    data = matrix.voter_weight_data
    labels = data.ballot_index if data.ballot_index is not None else np.arange(len(matrix))
    masked_data = ProfileVoterWeightData(
        weights=weights[mask].astype(np.float64),
        voter_sets=None if voter_sets is None else voter_sets[mask],
        weight_kind=WeightKind.FLOAT,
        ballot_index=labels[mask],
    )
    if isinstance(matrix, RankMatrix):
        return replace(matrix, voter_weight_data=masked_data, rankings=matrix.rankings[mask])
    return ScoreMatrix(
        voter_weight_data=masked_data,
        scores=matrix.scores[mask],
        score_columns=matrix.score_columns,
        candidates=matrix.candidates,
    )


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
            not conserve weight, or changes the ranking or score columns.
    """
    result = transform.transform_matrix(selected_matrix)
    if not isinstance(result, type(selected_matrix)):
        raise TransformContractError(
            f"transform_fn.transform_matrix must return a {type(selected_matrix).__name__}, got"
            f" {type(result)}."
        )
    received = sum(selected_matrix.voter_weight_data.weights)
    returned = sum(result.voter_weight_data.weights)
    if returned != received:
        raise TransformContractError(
            "transform_fn.transform_matrix must conserve weight. Received total weight"
            f" {received}, returned {returned}."
        )
    if isinstance(selected_matrix, RankMatrix) and isinstance(result, RankMatrix):
        if result.ranking_columns != selected_matrix.ranking_columns:
            raise TransformContractError(
                "transform_fn.transform_matrix must keep the ranking columns"
                f" {selected_matrix.ranking_columns}, one per position up to the profile's"
                f" max_ranking_length, returned {result.ranking_columns}."
            )
    if isinstance(selected_matrix, ScoreMatrix) and isinstance(result, ScoreMatrix):
        if result.score_columns != selected_matrix.score_columns:
            raise TransformContractError(
                "transform_fn.transform_matrix must keep the score columns"
                f" {selected_matrix.score_columns}, returned {result.score_columns}."
            )
    return result


def _validate_perturbed_rankings(perturbed_matrix: RankMatrix, profile: RankProfile) -> None:
    """
    Checks that perturbed rankings fit the profile they are returned to.

    The ranking columns were already checked against the selected matrix's, so the rankings
    have the profile's max_ranking_length positions.

    Args:
        perturbed_matrix (RankMatrix): Perturbed voters.
        profile (RankProfile): Profile to perturb.

    Raises:
        TransformContractError: The rankings rank more candidates than the profile's
            max_ranking_length allows, or rank candidates outside the profile's candidates.
    """

    if perturbed_matrix.has_ties().any():
        max_ranked = perturbed_matrix.max_candidates_ranked()
        if max_ranked > profile.max_ranking_length:
            raise TransformContractError(
                f"transform_fn returned a ranking of {max_ranked} candidates, more than the"
                f" profile's max_ranking_length {profile.max_ranking_length}."
            )

    undeclared_cands = perturbed_matrix.get_candidates_cast() - frozenset(profile.candidates)
    if undeclared_cands:
        raise TransformContractError(
            f"transform_fn returned candidates {set(undeclared_cands)} not in the profile."
            " Declare them in the profile's candidates before perturbing."
        )


def _validate_perturbed_scores(perturbed_matrix: ScoreMatrix, profile: ScoreProfile) -> None:
    """
    Checks that perturbed scores only score the profile's candidates.

    Args:
        perturbed_matrix (ScoreMatrix): Perturbed voters.
        profile (ScoreProfile): Profile to perturb.

    Raises:
        TransformContractError: A candidate outside the profile's candidates is scored.
    """
    undeclared_cands = perturbed_matrix.get_candidates_cast() - frozenset(profile.candidates)
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


@dataclass
class SwapCandidates:
    """
    ProfileTransform class to swap two candidates on every ballot.

    Every candidate set holding one of the two candidates has it replaced by the other, so a tie
    with one becomes a tie with the other. A set holding both, or neither, is unchanged. The swap
    is a mapping over the matrix's candidate-set table, applied to all ballots at once,
    regardless of the candidates' distance or order. Swapped sets that are not already in the
    table are appended to it, and sets that are no longer used stay in the table.

    Attributes:
        candidate_a (Candidate): Candidate to swap with candidate_b.
        candidate_b (Candidate): Candidate to swap with candidate_a.

    Raises:
        ValueError: A ballot ranks candidate_a or candidate_b in more than one position, so the
            swap is ambiguous.
    """

    candidate_a: Candidate
    candidate_b: Candidate

    def transform_matrix(self, matrix: RankMatrix) -> RankMatrix:
        cand_sets = matrix.cand_sets
        has_a = np.fromiter((self.candidate_a in s for s in cand_sets), bool, len(cand_sets))
        has_b = np.fromiter((self.candidate_b in s for s in cand_sets), bool, len(cand_sets))

        # fail before swapping if a ballot ranks either candidate in more than one position
        slots = matrix.rankings - EMPTY_SET_ID
        repeated = (np.count_nonzero(has_a[slots], axis=1) > 1) | (
            np.count_nonzero(has_b[slots], axis=1) > 1
        )
        if np.any(repeated):
            row = int(np.flatnonzero(repeated)[0])
            ballot_index = matrix.voter_weight_data.ballot_index
            ballot_label = row if ballot_index is None else ballot_index[row]
            raise ValueError(
                f"Cannot swap {self.candidate_a} and {self.candidate_b}: Ballot Index"
                f" {ballot_label} ranks one of them in more than one position."
            )

        # identity map from each candidate-set ID to its swapped ID; only sets holding exactly
        # one of the two candidates change, and new swapped sets are appended to the table
        id_mapping = np.arange(EMPTY_SET_ID, EMPTY_SET_ID + len(cand_sets), dtype=np.intp)
        cand_set_to_id = {
            cand_set: cand_set_id for cand_set_id, cand_set in matrix.iter_candidate_sets()
        }
        swap = {self.candidate_a: self.candidate_b, self.candidate_b: self.candidate_a}
        for slot in np.flatnonzero(has_a != has_b):
            swapped_set = frozenset(swap.get(candidate, candidate) for candidate in cand_sets[slot])
            id_mapping[slot] = cand_set_to_id.setdefault(
                swapped_set, len(cand_set_to_id) + EMPTY_SET_ID
            )

        return replace(
            matrix,
            rankings=id_mapping[slots],
            cand_sets=tuple(cand_set_to_id),  # insertion order is ID order
        )


@dataclass
class SwapAdjacentRankPositions:
    """
    ProfileTransform class to swap one random pair of adjacent ranking positions on each ballot.

    Follows the rank jitter in ``JitterTest`` of
    https://github.com/peterrrock2/divij-rticle-4/blob/main/sensitivity.py. For each ballot, a
    position is drawn uniformly from the profile's ranking positions and a direction, left or
    right, is drawn uniformly. The ballot's candidate sets at that position and its neighbor are
    swapped. If the neighbor falls off either end of the ranking positions, or either position is
    unranked padding, the ballot is unchanged. An explicit empty ranking position is swapped like
    any other position.

    Use with ``perturb``, whose ``per_voter_probability`` selects the ballots to jitter. Each
    selected voter receives its own swap, so rows are split into one row per unit of weight.

    The generator is created once from ``rng_seed``, so repeated calls, such as repeated
    ``perturb`` runs with the same transform, continue its random stream instead of repeating
    the same swaps.

    Attributes:
        rng_seed (int, optional): Seed for the positions and directions. Pass a seed for
            reproducible swaps. Defaults to None, which creates an unseeded generator.
    """

    rng_seed: int | None = None
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.rng_seed)

    def transform_matrix(self, matrix: RankMatrix) -> RankMatrix:
        # each voter draws its own swap
        unit_rows, unit_data = _expand_to_units(matrix)
        ranking_arr = matrix.rankings[unit_rows]
        num_rows, num_positions = ranking_arr.shape

        rows = np.arange(num_rows)
        positions = self._rng.integers(0, max(num_positions, 1), size=num_rows)
        neighbors = positions + self._rng.choice([-1, 1], size=num_rows)

        in_range = (neighbors >= 0) & (neighbors < num_positions)
        rows, positions, neighbors = rows[in_range], positions[in_range], neighbors[in_range]
        # swapping a ranked position with trailing padding leaves the ranking unchanged
        ranked = (ranking_arr[rows, positions] != END_OF_RANKING_ID) & (
            ranking_arr[rows, neighbors] != END_OF_RANKING_ID
        )
        rows, positions, neighbors = rows[ranked], positions[ranked], neighbors[ranked]

        ranking_arr[rows, positions], ranking_arr[rows, neighbors] = (
            ranking_arr[rows, neighbors],
            ranking_arr[rows, positions],
        )

        return replace(matrix, voter_weight_data=unit_data, rankings=ranking_arr)
