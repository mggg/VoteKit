from collections import namedtuple
from copy import deepcopy
from fractions import Fraction
import random
from typing import Union, Iterable

from .ballot import Ballot
from .pref_profile import PreferenceProfile


COLOR_LIST = [
    (0.55, 0.71, 0.0),
    (0.82, 0.1, 0.26),
    (0.44, 0.5, 0.56),
    (1.0, 0.75, 0.0),
    (1.0, 0.77, 0.05),
    (0.0, 0.42, 0.24),
    (0.13, 0.55, 0.13),
    (0.9, 0.13, 0.13),
    (0.08, 0.38, 0.74),
    (0.41, 0.21, 0.61),
    (1.0, 0.72, 0.77),
    (1.0, 0.66, 0.07),
    (1.0, 0.88, 0.21),
    (0.55, 0.82, 0.77),
]

# Election Helper Functions
CandidateVotes = namedtuple("CandidateVotes", ["cand", "votes"])


def compute_votes(candidates: list, ballots: list[Ballot]) -> list[CandidateVotes]:
    """
    Computes first place votes for all candidates in a preference profile
    """

    votes = {}
    for candidate in candidates:
        weight = Fraction(0)
        for ballot in ballots:
            if not ballot.ranking:
                continue
            if len(ballot.ranking[0]) == 1:
                if ballot.ranking[0] == {candidate}:
                    weight += ballot.weight
            else:
                if candidate in ballot.ranking[0]:  # ties
                    weight += ballot.weight / len(ballot.ranking[0])
        votes[candidate] = weight

    ordered = [
        CandidateVotes(cand=key, votes=value)
        for key, value in sorted(votes.items(), key=lambda x: x[1], reverse=True)
    ]

    return ordered


def fractional_transfer(
    winner: str, ballots: list[Ballot], votes: dict, threshold: int
) -> list[Ballot]:
    """
    Calculates fractional transfer from winner, then removes winner
    from the list of ballots
    """
    transfer_value = (votes[winner] - threshold) / votes[winner]

    for ballot in ballots:
        if ballot.ranking and ballot.ranking[0] == {winner}:
            ballot.weight = ballot.weight * transfer_value

    return remove_cand(winner, ballots)


def random_transfer(
    winner: str, ballots: list[Ballot], votes: dict, threshold: int
) -> list[Ballot]:
    """
    Cambridge/Cincinnati-style transfer where transfer ballots are selected randomly
    """

    # turn all of winner's ballots into (multiple) ballots of weight 1
    weight_1_ballots = []
    for ballot in ballots:
        if ballot.ranking and ballot.ranking[0] == {winner}:
            # note: under random transfer, weights should always be integers
            for _ in range(int(ballot.weight)):
                weight_1_ballots.append(
                    Ballot(
                        id=ballot.id,
                        ranking=ballot.ranking,
                        weight=Fraction(1),
                        voters=ballot.voters,
                    )
                )

    # remove winner's ballots
    ballots = [
        ballot
        for ballot in ballots
        if not (ballot.ranking and ballot.ranking[0] == {winner})
    ]

    surplus_ballots = random.sample(weight_1_ballots, int(votes[winner]) - threshold)
    ballots += surplus_ballots

    transfered = remove_cand(winner, ballots)

    return transfered


def seqRCV_transfer(
    winner: str, ballots: list[Ballot], votes: dict, threshold: int
) -> list[Ballot]:
    """
    Useful for a Sequential RCV election which does not use a transfer method ballots \n
    ballots: list of ballots \n
    output: same ballot list
    """
    return ballots


def remove_cand(removed: Union[str, Iterable], ballots: list[Ballot]) -> list[Ballot]:
    """
    Removes candidate from ballots
    """
    if isinstance(removed, str):
        remove_set = {removed}
    elif isinstance(removed, Iterable):
        remove_set = set(removed)

    update = deepcopy(ballots)
    for n, ballot in enumerate(update):
        new_ranking = []
        for s in ballot.ranking:
            new_s = s.difference(remove_set)
            if len(new_s) > 0:
                new_ranking.append(new_s)
        update[n].ranking = new_ranking

    return update


def order_candidates_by_borda(candidate_set, candidate_borda):
    # Sort the candidates in candidate_set based on their Borda values
    ordered_candidates = sorted(
        candidate_set, key=lambda candidate: (-candidate_borda[candidate], candidate)
    )
    return ordered_candidates


# Summmary Stat functions
def first_place_votes(profile: PreferenceProfile) -> dict:
    """
    Wrapper for compute_votes to call on PreferenceProfile
    """
    cands = profile.get_candidates()
    ballots = profile.get_ballots()

    return {cand: float(votes) for cand, votes in compute_votes(cands, ballots)}


def mentions(profile: PreferenceProfile) -> dict:
    """
    Calculates total mentions for a candidates
    """
    mentions: dict[str, float] = {}

    ballots = profile.get_ballots()
    for ballot in ballots:
        for rank in ballot.ranking:
            for cand in rank:
                if cand not in mentions:
                    mentions[cand] = 0
                if len(rank) > 1:
                    mentions[cand] += (1 / len(rank)) * int(
                        ballot.weight
                    )  # split mentions for candidates that are tied
                else:
                    mentions[cand] += float(ballot.weight)

    return mentions


def borda_scores(
    profile: PreferenceProfile, ballot_length=None, score_vector=None
) -> dict:
    candidates = profile.get_candidates()
    if ballot_length is None:
        ballot_length = max([len(ballot.ranking) for ballot in profile.ballots])
    if score_vector is None:
        score_vector = list(range(ballot_length, 0, -1))

    candidate_borda = {c: Fraction(0) for c in candidates}
    for ballot in profile.ballots:
        current_ind = 0
        candidates_covered = []
        for s in ballot.ranking:
            position_size = len(s)
            local_score_vector = score_vector[current_ind : current_ind + position_size]
            borda_allocation = sum(local_score_vector) / position_size
            for c in s:
                candidate_borda[c] += Fraction(borda_allocation) * ballot.weight
            current_ind += position_size
            candidates_covered += list(s)

        # If ballot was incomplete, evenly allocation remaining points
        if current_ind < len(score_vector):
            remainder_cands = set(candidates).difference(set(candidates_covered))
            remainder_score_vector = score_vector[current_ind:]
            remainder_borda_allocation = sum(remainder_score_vector) / len(
                remainder_cands
            )
            for c in remainder_cands:
                candidate_borda[c] += (
                    Fraction(remainder_borda_allocation) * ballot.weight
                )

    return candidate_borda
