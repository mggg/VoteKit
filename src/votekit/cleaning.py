from copy import deepcopy
from fractions import Fraction
from functools import reduce
from itertools import groupby
from typing import Callable

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
        (PreferenceProfile): A cleaned PreferenceProfile.
    """

    ballots_nonempty = [
        deepcopy(ballot) for ballot in pp.get_ballots() if ballot.ranking
    ]
    if keep_candidates:
        old_cands = deepcopy(pp.get_candidates())
        pp_clean = PreferenceProfile(ballots=ballots_nonempty, candidates=old_cands)
    else:
        pp_clean = PreferenceProfile(ballots=ballots_nonempty)
    return pp_clean


def clean_profile(
    pp: PreferenceProfile, clean_ballot_func: Callable[[Ballot], Ballot]
) -> PreferenceProfile:
    """
    Allows user-defined cleaning rules for PreferenceProfile. Input function
    that applies modification or rule to a single ballot.

    Args:
        pp (PreferenceProfile): A PreferenceProfile to clean.
        clean_ballot_func (Callable[[Ballot], Ballot]): Function that 
            takes a list of ballots and cleans each ballot.

    Returns:
        (PreferenceProfile): A cleaned PreferenceProfile.
    """

    # apply cleaning function to clean all ballots
    if clean_ballot_func is not None:
        cleaned = map(clean_ballot_func, pp.ballots)
    # group ballots that have the same ranking after cleaning
    grouped_ballots = [
        list(result)
        for key, result in groupby(cleaned, key=lambda ballot: ballot.ranking)
    ]
    # merge ballots in the same groups
    new_ballots = [merge_ballots(b) for b in grouped_ballots]
    return PreferenceProfile(ballots=new_ballots)


def merge_ballots(ballots: list[Ballot]) -> Ballot:
    """
    Takes a list of ballots with the same ranking and merge them into one ballot.

    Args:
        ballots (list[Ballot]): A list of ballots to deduplicate.

    Returns:
        (Ballot): A ballot with the same ranking and aggregated weight and voters.
    """
    weight = sum(b.weight for b in ballots)
    ranking = ballots[0].ranking
    voters_to_merge = [b.voter_set for b in ballots if b.voter_set]
    voter_set = None
    if len(voters_to_merge) > 0:
        voter_set = reduce(lambda b1, b2: b1.union(b2), voters_to_merge)
        voter_set = set(voter_set)
    return Ballot(ranking=ranking, voter_set=voter_set, weight=Fraction(weight))


def deduplicate_profiles(pp: PreferenceProfile) -> PreferenceProfile:
    """
    Given a PreferenceProfile, deduplicates its ballots.

    Args:
        pp (PreferenceProfile): A PreferenceProfile to clean.

    Returns:
        (PreferenceProfile): A cleaned PreferenceProfile without duplicates.
    """

    def deduplicate_ballots(ballot: Ballot) -> Ballot:
        """
        Takes a ballot and deduplicates its rankings.

        Args:
            ballot (Ballot): a ballot with duplicates in its ranking.

        Returns:
            Ballot: a ballot without duplicates.
        """
        ranking = ballot.ranking
        dedup_ranking = []
        for cand in ranking:
            if cand in ranking and cand not in dedup_ranking:
                # dedup_ranking.append({None})
                dedup_ranking.append(cand)
        new_ballot = Ballot(
            id=ballot.id,
            weight=Fraction(ballot.weight),
            ranking=dedup_ranking,
            voter_set=ballot.voter_set,
        )
        return new_ballot

    pp_clean = clean_profile(pp=pp, clean_ballot_func=deduplicate_ballots)
    return pp_clean


def remove_noncands(
    profile: PreferenceProfile, non_cands: list[str]
) -> PreferenceProfile:
    """
    Removes user-assigned non-candidates from ballots, deletes ballots
    that are empty as a result of the removal.

    Args:
        profile (PreferenceProfile): A PreferenceProfile to clean.
        non_cands (list[str]): A list of non-candidates to be removed.

    Returns:
        (PreferenceProfile): A profile with non-candidates removed.
    """

    def remove_from_ballots(ballot: Ballot, non_cands: list[str]) -> Ballot:
        """
        Removes non-candidiates from ballot objects.

        Args:
            ballot (Ballot): a ballot to be cleaned.
            non_cands (list[str]): a list of candidates to remove.

        Returns:
            Ballot: returns a cleaned Ballot.
        """
        # TODO: adjust so string and list of strings are acceptable inputes

        to_remove = []
        for item in non_cands:
            to_remove.append({item})

        ranking = ballot.ranking
        clean_ranking = []
        for cand in ranking:
            if cand not in to_remove and cand not in clean_ranking:
                clean_ranking.append(cand)

        clean_ballot = Ballot(
            id=ballot.id,
            ranking=clean_ranking,
            weight=Fraction(ballot.weight),
            voter_set=ballot.voter_set,
        )

        return clean_ballot

    cleaned = [
        remove_from_ballots(ballot, non_cands)
        for ballot in profile.ballots
        if remove_from_ballots(ballot, non_cands).ranking
    ]
    grouped_ballots = [
        list(result)
        for key, result in groupby(cleaned, key=lambda ballot: ballot.ranking)
    ]
    # merge ballots in the same groups
    new_ballots = [merge_ballots(b) for b in grouped_ballots]
    return PreferenceProfile(ballots=new_ballots)
