from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction
from numbers import Integral, Real
from typing import Any, cast

import numpy as np

from votekit.pref_profile import RankProfile
from votekit.pref_profile.profile_matrix import EMPTY_SET_ID, END_OF_RANKING_ID
from votekit.types import Candidate, Numeric


def to_exact_fraction_weight(value: object) -> Fraction:
    """
    Convert an exact STV ballot weight to a nonnegative ``Fraction``.

    Warning:
        Exact STV operations assume every ranking position contains at most one candidate. This
        helper does not validate that profile-level precondition.

    Args:
        value (object): Ballot weight to convert. Integral real values and ``Fraction`` values
            are accepted.

    Returns:
        Fraction: Exact ballot weight.

    Raises:
        TypeError: If the weight is not a real number.
        ValueError: If the weight is negative, non-finite, or a non-integral float.
    """
    if isinstance(value, Fraction):
        if value < 0:
            raise ValueError("Ballot weight cannot be negative.")
        return value
    if isinstance(value, Integral):
        if value < 0:
            raise ValueError("Ballot weight cannot be negative.")
        return Fraction(int(value))
    if isinstance(value, Real):
        try:
            integer_value = int(cast(Any, value))
        except (OverflowError, ValueError) as error:
            raise ValueError("Ballot weight must be finite.") from error
        if value < 0:
            raise ValueError("Ballot weight cannot be negative.")
        if value != integer_value:
            raise ValueError(
                "Exact STV requires integral float weights; pass Fraction(numerator, denominator) "
                "for rational weights."
            )
        return Fraction(integer_value)
    raise TypeError("Exact STV ballot weights must be real numbers.")


def convert_profile_weights(
    profile: RankProfile,
    converter: Callable[[Any], Numeric],
) -> RankProfile:
    """
    Return a copy of a rank profile with converted ballot weights.

    Warning:
        Exact STV operations assume every ranking position contains at most one candidate. This
        helper does not validate that precondition.

    Args:
        profile (RankProfile): Profile whose weights are converted.
        converter (Callable[[Any], Numeric]): Function applied to each ballot weight.

    Returns:
        RankProfile: Profile with converted weights and unchanged rankings and candidates.
    """
    weights = profile._matrix.voter_weight_data.weights
    converted_weights = np.empty(len(weights), dtype=object)

    for idx, weight in enumerate(weights):
        converted_weights[idx] = converter(weight)

    return profile._with_weights(converted_weights)


def exact_first_place_votes(profile: RankProfile) -> dict[Candidate, Fraction]:
    """
    Compute exact first-place totals for an exact STV profile.

    Empty and tilde first positions contribute no votes, matching the float STV scorer.

    Warning:
        The profile must already have passed STV validation. Tied ranking positions are not
        supported.

    Args:
        profile (RankProfile): Profile with singleton rankings and ``Fraction`` weights.

    Returns:
        dict[Candidate, Fraction]: First-place total for every candidate, including zero totals.
    """
    scores = {candidate: Fraction(0) for candidate in profile.candidates_cast}
    if profile._matrix.rankings.size == 0:
        return scores

    for candidate_set_id, weight in zip(
        profile._matrix.rankings[:, 0], profile._matrix.voter_weight_data.weights
    ):
        if candidate_set_id in (EMPTY_SET_ID, END_OF_RANKING_ID):
            continue
        candidate_set = profile._matrix.get_cand_set_from_id(candidate_set_id)
        assert len(candidate_set) == 1
        candidate = next(iter(candidate_set))
        assert isinstance(weight, Fraction)
        scores[candidate] += weight
    return scores


def exact_borda_scores(profile: RankProfile) -> dict[Candidate, Fraction]:
    """
    Compute exact Borda scores for an exact STV profile.

    Warning:
        The profile must already have passed STV validation. Tied ranking positions are not
        supported.

    Args:
        profile (RankProfile): Profile with singleton rankings and ``Fraction`` weights.

    Returns:
        dict[Candidate, Fraction]: Exact Borda score for every candidate.

    Raises:
        ValueError: A ballot has an empty ranking position, including on zero-weight ballots.
    """
    if (profile._matrix.rankings == EMPTY_SET_ID).any():
        raise ValueError("Cannot compute Borda scores for a ballot with an empty ranking position.")

    scores = {candidate: Fraction(0) for candidate in profile.candidates_cast}
    if profile._matrix.rankings.size == 0:
        return scores

    candidates_by_set_id = {}
    for candidate_set_id, candidate_set in profile._matrix.iter_candidate_sets():
        if candidate_set_id in (EMPTY_SET_ID, END_OF_RANKING_ID):
            continue
        assert len(candidate_set) == 1
        candidates_by_set_id[candidate_set_id] = next(iter(candidate_set))

    for candidate_set_ids, weight in zip(
        profile._matrix.rankings, profile._matrix.voter_weight_data.weights
    ):
        assert isinstance(weight, Fraction)
        for rank, candidate_set_id in enumerate(candidate_set_ids):
            if candidate_set_id == END_OF_RANKING_ID:
                break
            scores[candidates_by_set_id[candidate_set_id]] += weight * (
                profile.max_ranking_length - rank
            )
    return scores
