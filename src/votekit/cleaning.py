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
    Removes empty ballots from a preference profile.
    :param pp: :class:`PreferenceProfile`
    :param keep_candidates: optional :class:`bool` If True, keep all of the
    candidates from the original preference profile in the returned preference
    profile, even if some no longer appear in the returned ballots.
    :return: A preference profile without empty ballots.
    :rtype: :class:`PreferenceProfile`
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


def _clean(
    pp: PreferenceProfile, clean_ballot_func: Callable[[Ballot], Ballot]
) -> PreferenceProfile:
    """
    General cleaning function that takes a preference profile and applies a
    cleaning function to each ballot and merges the ballots with the same ranking.
    This is used primarily when only the ballot ranking needs to be cleaned.
    :param pp: :class:`PreferenceProfile` The preference profile to be cleaned
    :param clean_ballot_func: :class:`Callable`[:class:`list`[:class:`Ballot`], :class:`Ballot`]
    Function that takes a list of ballots and cleans them
    :return: A cleaned preference profile
    :rtype: :class:`PreferenceProfile`
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
    :param ballots: :class:`list`[:class:`Ballot`] Where each ballot has the same
    ranking.
    :return: A ballot with the same ranking and aggregated weight and voters
    :rtype: :class:`Ballot`
    """
    weight = sum(b.weight for b in ballots)
    ranking = ballots[0].ranking
    voters_to_merge = [b.voters for b in ballots if b.voters]
    voters = None
    if len(voters_to_merge) > 0:
        voters = reduce(lambda b1, b2: b1.union(b2), voters_to_merge)
        voters = set(voters)
    return Ballot(ranking=ranking, voters=voters, weight=Fraction(weight))


# TODO: Brenda will replace this function with the function she wrote,
# TODO: change to keep ranks so that we’ll have None
def deduplicate_profiles(pp: PreferenceProfile) -> PreferenceProfile:
    """
    Given a preference profile, deduplicates its ballots.
    :param pp: :class:`PreferenceProfile`
    :return: A preference profile without duplicates
    :rtype: :class:`PreferenceProfile`
    """

    def deduplicate_ballots(ballot: Ballot) -> Ballot:
        """
        Takes a ballot and deduplicates its rankings
        :param ballot: :class:`Ballot` with duplicates in its ranking
        :return: A ballot without duplicates
        :rtype: :class:`Ballot`
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
            voters=ballot.voters,
        )
        return new_ballot

    pp_clean = _clean(pp=pp, clean_ballot_func=deduplicate_ballots)
    return pp_clean


def remove_from_ballots(ballot: Ballot, non_cands: list[str]) -> Ballot:
    """
    Removes non-candidiates from ballot objects
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
        voters=ballot.voters,
    )

    return clean_ballot


## TODO: typehints so that non_cands inputs is str or list of strs
def remove_noncands(
    profile: PreferenceProfile, non_cands: list[str]
) -> PreferenceProfile:
    """
    Removes user-assigned non-candidates from ballots, deletes ballots
    that are empty as a result of the removal.
    :param profile: :class:`PreferenceProfile` Uncleaned preference profile.
    :param non_cands: :class:`list`[:class:`str`] Non-candidate items to be removed.
    :return: A profile with non-candidates removed.
    :rtype: :class:`PreferenceProfile`
    """
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
