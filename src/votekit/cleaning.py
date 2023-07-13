from .profile import PreferenceProfile
from .ballot import Ballot
from copy import deepcopy
from typing import Callable
from itertools import groupby
from functools import reduce

def remove_empty_ballots(pp: PreferenceProfile, keep_candidates: bool = False) -> PreferenceProfile:
    """
    Returns a preference profile which is the input pp without empty ballots.
    keep_candidates: use old set of candidates, even if some no longer appear
    """
    ballots_nonempty = [deepcopy(ballot) for ballot in pp.get_ballots() if ballot.ranking]

    if keep_candidates:
        old_cands = deepcopy(pp.get_candidates())
        pp_clean = PreferenceProfile(ballots=ballots_nonempty, candidates=old_cands)
    else:
        pp_clean = PreferenceProfile(ballots=ballots_nonempty)

    return pp_clean

def clean(pp: PreferenceProfile, clean_ballot_func: Callable[[list[Ballot]], list[Ballot]]) -> PreferenceProfile:
    # apply cleaning function to clean all ballots
    cleaned = map(clean_ballot_func, pp.ballots)
    # group ballots that have the same ranking after cleaning
    grouped_ballots = [list(result) for key, result in groupby(cleaned, key=lambda ballot: ballot.ranking)]
    print(grouped_ballots)
    # merge ballots in the same groups
    new_ballots = [merge_ballots(b) for b in grouped_ballots]
    return PreferenceProfile(ballots=new_ballots)

def merge_ballots(ballots: list[Ballot]) -> Ballot:
    weight = reduce(lambda b1, b2: b1.weight + b2.weight, ballots)
    voters = reduce(lambda b1, b2: b1.voters.union(b2.voters), ballots)
    ranking = ballots[0].ranking
    return Ballot(ranking=ranking, voters=voters, weight=weight)


def deduplicate_ballots(ballots: list[Ballot]) -> list[Ballot]:

    def _deduplicate_ranking(ranking: list[set]) -> list[set]:
        # note: can ignore dups in ties because it's a set
        dedup_ranking = []
        for cand in ranking:
            if cand in ranking and cand in dedup_ranking:
                dedup_ranking.append({None})
            elif cand in ranking:
                dedup_ranking.append(cand)
        return dedup_ranking

    dedup_ballots = []

    for ballot in ballots:
        dedup_ranking = _deduplicate_ranking(ballot.ranking)
        new_ballot = Ballot(id=ballot.id, weight=ballot.weight, ranking=dedup_ranking, voters=ballot.voters)
        dedup_ballots.append(new_ballot)
    
    return dedup_ballots

def deduplicate_profiles(pp: PreferenceProfile) -> PreferenceProfile:
    pp_clean = clean(pp=pp, clean_ballot_func=deduplicate_ballots)
    return pp_clean
        
    """
    removes duplicates in a voter's ranking of candidates
    ex. ['c1', 'c1', 'c2] -> ['c1', '', 'c2']

    Args:
        ranking (list of string): the candidates ordered by voter's ranking

    Returns:
        a list of string: the ranking of candidates without duplicates
    """
    

    # ranking_without_dups = []
    # for cand in ranking:
    #     if cand in ranking and cand in ranking_without_dups:
    #         ranking_without_dups.append('')
    #     elif cand in ranking:
    #         ranking_without_dups.append(cand)
    # return ranking_without_dups
