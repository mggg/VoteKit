from fractions import Fraction
from functools import reduce
from typing import Callable, Union, TypeVar, cast

from .pref_profile import PreferenceProfile
from .ballot import Ballot


def remove_empty_ballots(
    pp: PreferenceProfile, keep_candidates: bool = False
) -> PreferenceProfile:
    """
    Removes empty ballots from a PreferenceProfile.

    Args:
        pp (PreferenceProfile): A PreferenceProfile to clean.
        keep_candidates (bool, optional): If True, keep all of the candidates
            from the original PreferenceProfile in the returned PreferenceProfile, even if
            they got no votes. Defaults to False.

    Returns:
        PreferenceProfile: A cleaned PreferenceProfile.
    """

    ballots_nonempty = tuple([ballot for ballot in pp.ballots if ballot.ranking])
    if keep_candidates:
        old_cands = pp.candidates
        pp_clean = PreferenceProfile(ballots=ballots_nonempty, candidates=old_cands)
    else:
        pp_clean = PreferenceProfile(ballots=ballots_nonempty)
    return pp_clean


def clean_profile(
    pp: PreferenceProfile, clean_ballot_func: Callable[[Ballot], Ballot]
) -> PreferenceProfile:
    """
    Allows user-defined cleaning rules for PreferenceProfile. Input function
    that applies modification to a single ballot.

    Args:
        pp (PreferenceProfile): A PreferenceProfile to clean.
        clean_ballot_func (Callable[[Ballot], Ballot]): Function that
            takes a ``Ballot`` and returns a cleaned ``Ballot``.

    Returns:
        PreferenceProfile: A cleaned ``PreferenceProfile``.
    """
    cleaned = map(clean_ballot_func, pp.ballots)

    return PreferenceProfile(ballots=tuple(cleaned)).condense_ballots()


def merge_ballots(ballots: list[Ballot]) -> Ballot:
    """
    Takes a list of ballots with the same ranking and merges them into one ballot.

    Args:
        ballots (list[Ballot]): A list of ballots to deduplicate.

    Returns:
        Ballot: A ballot with the same ranking and aggregated weight and voters.
    """
    weight = sum(b.weight for b in ballots)
    ranking = ballots[0].ranking
    voters_to_merge = [b.voter_set for b in ballots if b.voter_set]
    voter_set = None
    if len(voters_to_merge) > 0:
        voter_set = reduce(lambda b1, b2: b1.union(b2), voters_to_merge)
        voter_set = set(voter_set)
    return Ballot(ranking=ranking, voter_set=voter_set, weight=Fraction(weight))


COB = TypeVar("COB", PreferenceProfile, tuple[Ballot, ...], Ballot)


def remove_repeated_candidates(
    profile_or_ballots: COB,
) -> COB:
    """
    Given a collection of ballots (a profile, a tuple, or a single ballot), if a candidate
    appears multiple times on a ballot, keep the first instance, remove any further instances,
    and condense any empty rankings as as result. Only works on ranking ballots, not score ballots.

    Args:
        profile_or_ballots (Union[PreferenceProfile, tuple[Ballot,...], Ballot]): Collection
            of ballots to remove repeated candidates from.

    Returns:
        Union[PreferenceProfile, tuple[Ballot,...],Ballot]:
            Updated collection of ballots with duplicate candidate(s) removed.

    Raises:
        TypeError: All ballots must only have rankings, not scores.
        TypeError: All ballots must have rankings.
    """

    if isinstance(profile_or_ballots, PreferenceProfile):
        ballots = profile_or_ballots.ballots
    elif isinstance(profile_or_ballots, Ballot):
        ballots = (profile_or_ballots,)
    else:
        ballots = profile_or_ballots[:]

    scrubbed_ballots = [Ballot()] * len(ballots)

    for i, ballot in enumerate(ballots):
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

        scrubbed_ballots[i] = new_ballot

    if isinstance(profile_or_ballots, PreferenceProfile):
        return cast(
            COB,
            PreferenceProfile(
                ballots=tuple(scrubbed_ballots),
                candidates=profile_or_ballots.candidates,
                max_ballot_length=profile_or_ballots.max_ballot_length,
            ),
        )

    elif isinstance(profile_or_ballots, Ballot):
        return cast(COB, scrubbed_ballots[0])

    else:
        return cast(COB, tuple(scrubbed_ballots))


def remove_cand(
    removed: Union[str, list],
    profile_or_ballots: COB,
    condense: bool = True,
    leave_zero_weight_ballots: bool = False,
) -> COB:
    """
    Removes specified candidate(s) from profile, ballot, or list of ballots. When a candidate is
    removed from a ballot, lower ranked candidates are moved up.
    Automatically condenses any ballots that match as result of scrubbing.

    Args:
        removed (Union[str, list]): Candidate or list of candidates to be removed.
        profile_or_ballots (Union[PreferenceProfile, tuple[Ballot,...], Ballot]): Collection
            of ballots to remove candidates from.
        condense (bool, optional):  Whether or not to return a condensed collection of ballots,
            where they are grouped by multiplicity. Defaults to True.
        leave_zero_weight_ballots (bool, optional): Whether or not to leave ballots with zero
            weight in the collection. Defaults to False.

    Returns:
        Union[PreferenceProfile, tuple[Ballot,...],Ballot]:
            Updated collection of ballots with candidate(s) removed.
    """
    if isinstance(removed, str):
        removed = [removed]

    # map to tuple of ballots
    if isinstance(profile_or_ballots, PreferenceProfile):
        ballots = profile_or_ballots.ballots
    elif isinstance(profile_or_ballots, Ballot):
        ballots = (profile_or_ballots,)
    else:
        ballots = profile_or_ballots[:]

    scrubbed_ballots = [Ballot()] * len(ballots)
    for i, ballot in enumerate(ballots):
        new_ranking = []
        new_scores = {}
        if ballot.ranking:
            for s in ballot.ranking:
                new_s = []
                for c in s:
                    if c not in removed:
                        new_s.append(c)
                if len(new_s) > 0:
                    new_ranking.append(frozenset(new_s))

        if ballot.scores:
            new_scores = {
                c: score for c, score in ballot.scores.items() if c not in removed
            }

        if len(new_ranking) > 0 and len(new_scores) > 0:
            scrubbed_ballots[i] = Ballot(
                ranking=tuple(new_ranking), weight=ballot.weight, scores=new_scores
            )
        elif len(new_ranking) > 0:
            scrubbed_ballots[i] = Ballot(
                ranking=tuple(new_ranking), weight=ballot.weight
            )

        elif len(new_scores) > 0:
            scrubbed_ballots[i] = Ballot(weight=ballot.weight, scores=new_scores)

        # else ballot exhausted
        else:
            scrubbed_ballots[i] = Ballot(weight=Fraction(0))

    # return matching input data type
    if isinstance(profile_or_ballots, PreferenceProfile):
        clean_profile = PreferenceProfile(
            ballots=tuple([b for b in scrubbed_ballots if b.weight > 0]),
            candidates=tuple(
                [c for c in profile_or_ballots.candidates if c not in removed]
            ),
        )

        if leave_zero_weight_ballots:
            clean_profile = PreferenceProfile(
                ballots=tuple(scrubbed_ballots),
                candidates=tuple(
                    [c for c in profile_or_ballots.candidates if c not in removed]
                ),
            )

        if condense:
            clean_profile = clean_profile.condense_ballots()

        return cast(COB, clean_profile)

    elif isinstance(profile_or_ballots, Ballot):
        clean_profile = None

        if leave_zero_weight_ballots:
            clean_profile = PreferenceProfile(
                ballots=tuple(scrubbed_ballots),
            )
        else:
            clean_profile = PreferenceProfile(
                ballots=tuple([b for b in scrubbed_ballots if b.weight > 0]),
            )

        if condense:
            clean_profile = clean_profile.condense_ballots()

        return cast(COB, clean_profile.ballots[0])
    else:
        clean_profile = None

        if leave_zero_weight_ballots:
            clean_profile = PreferenceProfile(
                ballots=tuple(scrubbed_ballots),
            )
        else:
            clean_profile = PreferenceProfile(
                ballots=tuple([b for b in scrubbed_ballots if b.weight > 0]),
            )

        if condense:
            clean_profile = clean_profile.condense_ballots()

        return cast(COB, clean_profile.ballots)
