from votekit.profile import PreferenceProfile
from copy import deepcopy
from votekit.ballot import Ballot
from typing import Callable
from itertools import groupby
from functools import reduce


def remove_empty_ballots(
    pp: PreferenceProfile, keep_candidates: bool = False
) -> PreferenceProfile:
    """
    Returns a preference profile which is the input pp without empty ballots.
    keep_candidates: use old set of candidates, even if some no longer appear
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

def clean(
    pp: PreferenceProfile, clean_ballot_func: Callable[[Ballot], Ballot] = None
) -> PreferenceProfile:
    """
    General cleaning function that takes a preference profile and applies a
    cleaning function to each ballot and merges the ballots with the same ranking
    used primarily when only the ballot ranking needs to be cleaned
    Args:
        pp (PreferenceProfile): preference profile to be cleaned
        clean_ballot_func (Callable[[list[Ballot]], list[Ballot]]): function that
        takes a list of ballots and cleans them

    Returns:
        PreferenceProfile: a cleaned preference profile
    """

    # apply cleaning function to clean all ballots

    cleaned = pp.ballots
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
    takes a list of ballots and merge them
    Args:
        ballots (list[Ballot]): a list of ballots with the same ranking
    Returns:
        Ballot: a ballot with the same ranking and aggregated weight and voters
    """
    weight = sum(b.weight for b in ballots)
    ranking = ballots[0].ranking
    voters_to_merge = [b.voters for b in ballots if b.voters]
    voters = None
    if len(voters_to_merge) > 0:
        voters = reduce(lambda b1, b2: b1.union(b2), voters_to_merge)
        voters = set(voters)
    return Ballot(ranking=ranking, voters=voters, weight=float(weight))


def overvote(
    pp: PreferenceProfile
) -> PreferenceProfile:
    """
    Returns a preference profile which truncates overvotes then recommutes
    total votes for each ballot permutation. Examples:
    ['A','B','B'] -> ['A', 'B']
    ['A', 'B', 'A'] -> ['A', 'B']
    """
    updated_BL =[]
    for ballot in deepcopy(pp.get_ballots()):
        updated_ranking = []
        for rank in ballot.ranking:
            if rank not in updated_ranking:
                updated_ranking.append(rank)
        
        updated_ballotRank = Ballot(ranking = updated_ranking, weight = ballot.weight)
        updated_BL.append(updated_ballotRank)
    pp_clean = PreferenceProfile(ballots=updated_BL)
    return pp_clean