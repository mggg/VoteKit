from fractions import Fraction
from functools import partial
from typing import Callable, Union, overload, Literal

from .pref_profile import PreferenceProfile
from .ballot import Ballot


@overload
def clean_profile(
    profile: PreferenceProfile,
    clean_ballot_func: Callable[[Ballot], Ballot],
    return_adjusted_count: Literal[False],
    remove_empty_ballots: bool = True,
    remove_zero_weight_ballots: bool = True,
) -> PreferenceProfile: ...
@overload
def clean_profile(
    profile: PreferenceProfile,
    clean_ballot_func: Callable[[Ballot], Ballot],
    return_adjusted_count: Literal[True],
    remove_empty_ballots: bool = True,
    remove_zero_weight_ballots: bool = True,
) -> tuple[PreferenceProfile, Fraction]: ...


def clean_profile(
    profile: PreferenceProfile,
    clean_ballot_func: Callable[[Ballot], Ballot],
    return_adjusted_count: Literal[True] | Literal[False] = False,
    remove_empty_ballots: bool = True,
    remove_zero_weight_ballots: bool = True,
) -> PreferenceProfile | tuple[PreferenceProfile, Fraction]:
    """
    Allows user-defined cleaning rules for PreferenceProfile. Input function
    that applies modification to a single ballot.

    Args:
        profile (PreferenceProfile): A PreferenceProfile to clean.
        clean_ballot_func (Callable[[Ballot], Ballot]): Function that
            takes a ``Ballot`` and returns a cleaned ``Ballot``.
        return_adjusted_count (bool, optional): Whether or not to return the number of ballots
            adjusted. An adjusted ballot is a ballot that was not spoiled by the cleaning rule,
            but did have some adjustment. Count is computed using original ballot weights.
            Defaults to False. If True, function returns a tuple (cleaned_profile, count).
        remove_empty_ballots (bool, optional): Whether or not to remove ballots that have no
            ranking or scores as a result of cleaning. Defaults to True.
        remove_zero_weight_ballots (bool, optional): Whether or not to remove ballots that have no
            weight as a result of cleaning. Defaults to True.

    Returns:
        Union[PreferenceProfile, tuple[PreferenceProfile, Fraction]]:
            A cleaned ``PreferenceProfile``. If return_adjusted_count, function returns the profile
            and the adjusted count as a tuple (cleaned_profile, count).
    """
    new_ballots = [Ballot()] * len(profile.ballots)

    num_adjusted = Fraction(0)
    for i, b in enumerate(profile.ballots):
        new_b = clean_ballot_func(b)

        if new_b != b and (new_b.ranking or new_b.scores) and new_b.weight > 0:
            num_adjusted += b.weight

        new_ballots[i] = new_b

    if remove_empty_ballots:
        new_ballots = [b for b in new_ballots if b.ranking or b.scores]

    if remove_zero_weight_ballots:
        new_ballots = [b for b in new_ballots if b.weight > 0]

    if return_adjusted_count:
        return PreferenceProfile(ballots=tuple(new_ballots)), num_adjusted
    else:
        return PreferenceProfile(ballots=tuple(new_ballots))


def remove_repeated_candidates_from_ballot(
    ballot: Ballot,
) -> Ballot:
    """
    Given a ballot, if a candidate appears multiple times on a ballot, keep the first instance,
    remove any further instances, and condense any empty rankings as as result.
    Only works on ranking ballots, not score ballots.

    Args:
        ballot (Ballot]): Ballot to remove repeated candidates from.

    Returns:
        Ballot: Ballot with duplicate candidate(s) removed.

    Raises:
        TypeError: Ballot must only have rankings, not scores.
        TypeError: Ballot must have rankings.
    """

    if not ballot.ranking:
        raise TypeError(f"Ballot must have rankings: {ballot}")
    elif ballot.scores:
        raise TypeError(f"Ballot must only have rankings, not scores: {ballot}")

    dedup_ranking = []
    seen_cands = []

    for cand_set in ballot.ranking:
        new_position = []
        for cand in cand_set:
            if cand not in seen_cands:
                new_position.append(cand)
                seen_cands.append(cand)

        if len(new_position) > 0:
            dedup_ranking.append(frozenset(new_position))

    new_ballot = Ballot(
        id=ballot.id,
        weight=Fraction(ballot.weight),
        ranking=tuple(dedup_ranking),
        voter_set=ballot.voter_set,
    )

    return new_ballot


@overload
def remove_repeated_candidates(
    profile: PreferenceProfile,
    return_adjusted_count: Literal[False],
    remove_empty_ballots: bool = True,
    remove_zero_weight_ballots: bool = True,
) -> PreferenceProfile: ...
@overload
def remove_repeated_candidates(
    profile: PreferenceProfile,
    return_adjusted_count: Literal[True],
    remove_empty_ballots: bool = True,
    remove_zero_weight_ballots: bool = True,
) -> tuple[PreferenceProfile, Fraction]: ...
def remove_repeated_candidates(
    profile: PreferenceProfile,
    return_adjusted_count: Literal[True] | Literal[False] = False,
    remove_empty_ballots: bool = True,
    remove_zero_weight_ballots: bool = True,
) -> PreferenceProfile | tuple[PreferenceProfile, Fraction]:
    """
    Given a profile, if a candidate appears multiple times on a ballot, keep the first instance,
    remove any further instances, and condense any empty rankings as as result.
    Only works on ranking ballots, not score ballots.

    Args:
        profile (PreferenceProfile): Profile to remove repeated candidates from.
        return_adjusted_count (bool, optional): Whether or not to return the number of ballots
            adjusted. An adjusted ballot is a ballot that was not spoiled by the cleaning rule,
            but did have some adjustment. Count is computed using original ballot weights.
            Defaults to False. If True, function returns a tuple (cleaned_profile, count).
        remove_empty_ballots (bool, optional): Whether or not to remove ballots that have no
            ranking or scores as a result of cleaning. Defaults to True.
        remove_zero_weight_ballots (bool, optional): Whether or not to remove ballots that have no
            weight as a result of cleaning. Defaults to True.

    Returns:
        Union[PreferenceProfile, tuple[PreferenceProfile, Fraction]]:
            A cleaned ``PreferenceProfile``. If return_adjusted_count, function returns the profile
            and the adjusted count as a tuple (cleaned_profile, count).

    Raises:
        TypeError: Ballots must only have rankings, not scores.
        TypeError: Ballots must have rankings.
    """

    return clean_profile(
        profile,
        remove_repeated_candidates_from_ballot,
        return_adjusted_count,
        remove_empty_ballots,
        remove_zero_weight_ballots,
    )


def remove_cand_from_ballot(
    removed: Union[str, list],
    ballot: Ballot,
) -> Ballot:
    """
    Removes specified candidate(s) from ballot. When a candidate is
    removed from a ballot, lower ranked candidates are moved up.

    Args:
        removed (Union[str, list]): Candidate or list of candidates to be removed.
        ballot (Ballot): Ballot to remove candidates from.

    Returns:
        Ballot: Ballot with candidate(s) removed.
    """
    if isinstance(removed, str):
        removed = [removed]

    new_ranking = []
    if ballot.ranking:
        for s in ballot.ranking:
            new_s = []
            for c in s:
                if c not in removed:
                    new_s.append(c)
            if len(new_s) > 0:
                new_ranking.append(frozenset(new_s))

    new_scores = {}
    if ballot.scores:
        new_scores = {
            c: score for c, score in ballot.scores.items() if c not in removed
        }

    new_ballot = Ballot(
        ranking=tuple(new_ranking) if len(new_ranking) else None,
        weight=ballot.weight,
        scores=new_scores if len(new_scores) else None,
        id=ballot.id,
        voter_set=ballot.voter_set,
    )

    return new_ballot


@overload
def remove_cand(
    removed: Union[str, list],
    profile: PreferenceProfile,
    return_adjusted_count: Literal[False],
    remove_empty_ballots: bool = True,
    remove_zero_weight_ballots: bool = True,
) -> PreferenceProfile: ...


@overload
def remove_cand(
    removed: Union[str, list],
    profile: PreferenceProfile,
    return_adjusted_count: Literal[True],
    remove_empty_ballots: bool = True,
    remove_zero_weight_ballots: bool = True,
) -> PreferenceProfile | tuple[PreferenceProfile, Fraction]: ...


def remove_cand(
    removed: Union[str, list],
    profile: PreferenceProfile,
    return_adjusted_count: Literal[True] | Literal[False] = False,
    remove_empty_ballots: bool = True,
    remove_zero_weight_ballots: bool = True,
) -> PreferenceProfile | tuple[PreferenceProfile, Fraction]:
    """
    Given a profile, remove the given candidate(s) from the ballots. Any ranking left empty
    as a result is condensed. Removes candidates from score dictionary as well.

    Args:
        removed (Union[str, list]): Candidate or list of candidates to be removed.
        profile (PreferenceProfile): Profile to remove repeated candidates from.
        return_adjusted_count (bool, optional): Whether or not to return the number of ballots
            adjusted. An adjusted ballot is a ballot that was not spoiled by the cleaning rule,
            but did have some adjustment. Count is computed using original ballot weights.
            Defaults to False. If True, function returns a tuple (cleaned_profile, count).
        remove_empty_ballots (bool, optional): Whether or not to remove ballots that have no
            ranking or scores as a result of cleaning. Defaults to True.
        remove_zero_weight_ballots (bool, optional): Whether or not to remove ballots that have no
            weight as a result of cleaning. Defaults to True.

    Returns:
        Union[PreferenceProfile, tuple[PreferenceProfile, Fraction]]:
            A cleaned ``PreferenceProfile``. If return_adjusted_count, function returns the profile
            and the adjusted count as a tuple (cleaned_profile, count).
    """

    return clean_profile(
        profile,
        partial(remove_cand_from_ballot, removed),
        return_adjusted_count,
        remove_empty_ballots,
        remove_zero_weight_ballots,
    )
