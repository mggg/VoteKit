from fractions import Fraction
from numbers import Integral, Real
from typing import Any, Callable, cast

from votekit.pref_profile import RankProfile
from votekit.types import Candidate, Numeric


def to_exact_fraction_weight(value: object) -> Fraction:
    """
    Convert an exact STV ballot weight to a nonnegative ``Fraction``.

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

    Args:
        profile (RankProfile): Profile whose weights are converted.
        converter (Callable[[Any], Numeric]): Function applied to each ballot weight.

    Returns:
        RankProfile: Profile with converted weights and unchanged rankings and candidates.
    """
    converted_df = profile.df.copy()
    converted_df["Weight"] = converted_df["Weight"].map(converter)
    converted_profile = RankProfile(
        df=converted_df,
        candidates=profile.candidates,
        max_ranking_length=profile.max_ranking_length,
    )
    assert isinstance(converted_profile, RankProfile)
    return converted_profile


def exact_first_place_votes(profile: RankProfile) -> dict[Candidate, Fraction]:
    """
    Compute exact first-place totals for an exact STV profile.

    Args:
        profile (RankProfile): Profile with singleton rankings and ``Fraction`` weights.

    Returns:
        dict[Candidate, Fraction]: First-place total for every candidate, including zero totals.
    """
    scores = {candidate: Fraction(0) for candidate in profile.candidates_cast}
    if profile._df.empty:
        return scores

    for candidate_id, weight in zip(profile._df["Ranking_1"], profile._df["Weight"]):
        if candidate_id == -1:
            continue
        candidate_set = profile.id_candidate_map[candidate_id]
        assert len(candidate_set) == 1
        candidate = next(iter(candidate_set))
        assert isinstance(weight, Fraction)
        scores[candidate] += weight
    return scores


def exact_borda_scores(profile: RankProfile) -> dict[Candidate, Fraction]:
    """
    Compute exact Borda scores for an exact STV profile.

    Args:
        profile (RankProfile): Profile with singleton rankings and ``Fraction`` weights.

    Returns:
        dict[Candidate, Fraction]: Exact Borda score for every candidate.
    """
    scores = {candidate: Fraction(0) for candidate in profile.candidates_cast}
    assert profile.max_ranking_length is not None

    for ballot in profile.ballots:
        assert ballot.ranking is not None
        assert isinstance(ballot.weight, Fraction)
        for rank, candidate_set in enumerate(ballot.ranking):
            if candidate_set == frozenset({"~"}):
                break
            assert len(candidate_set) == 1
            candidate = next(iter(candidate_set))
            scores[candidate] += ballot.weight * (profile.max_ranking_length - rank)
    return scores
