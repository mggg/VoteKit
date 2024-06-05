from collections import namedtuple
from fractions import Fraction
import numpy as np
from typing import Union, Iterable, Optional, Any
from itertools import permutations
import math
import warnings

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


def ballots_by_first_cand(candidates: list[str], ballots: list[Ballot]) -> dict:
    """
    Partitions the ballots by first place candidate.

    Args:
        candidates (list[str]): A list of candidate strings.
        ballots (list[Ballot]): A list of ballots.

    Returns:
        dict: A dictionary whose keys are candidates and values are lists of ballots that
            have that candidate first.
    """
    cand_dict = {c: [] for c in candidates}  # type: dict

    for b in ballots:
        if b.ranking:
            # find first place candidate, ensure there is only one
            first_cand = list(b.ranking[0])
            if len(first_cand) > 1:
                raise ValueError(f"Ballot {b} has a tie for first.")
            else:
                first_cand = first_cand[0]

            cand_dict[first_cand].append(b)

    return cand_dict


def compute_votes(
    candidates: list[str],
    ballots: list[Ballot],
) -> tuple[list[CandidateVotes], dict]:
    """
    Computes first place votes for all candidates in a ``PreferenceProfile``.

    Args:
        candidates (list[str]): List of all candidates in a ``PreferenceProfile``.
        ballots (list[Ballot]): List of Ballot objects.

    Returns:
        tuple: A tuple (ordered, votes) where ordered is a list of tuples (cand, first place votes)
            ordered by decreasing first place votes and votes is a dictionary whose keys are
            candidates and values are first place votes.
    """
    votes = {cand: Fraction(0) for cand in candidates}

    for ballot in ballots:
        if not ballot.ranking:
            continue
        first_place_cand = unset(ballot.ranking[0])
        if isinstance(first_place_cand, list):
            for cand in first_place_cand:
                votes[cand] += ballot.weight / len(first_place_cand)
        else:
            votes[first_place_cand] += ballot.weight

    ordered = [
        CandidateVotes(cand=key, votes=value)
        for key, value in sorted(votes.items(), key=lambda x: x[1], reverse=True)
    ]

    return ordered, votes


def remove_cand(removed: Union[str, Iterable], ballots: list[Ballot]) -> list[Ballot]:
    """
    Removes specified candidate(s) from ballots.

    Args:
        removed (Union[str, Iterable]): Candidate or group of candidates to be removed.
        ballots (list[Ballot]): List of ballots to remove candidate(s) from.

    Returns:
        list[Ballot]: Updated list of ballots with candidate(s) removed.
    """

    if isinstance(removed, str):
        remove_set = {removed}
    elif isinstance(removed, Iterable):
        remove_set = set(removed)

    update = []
    for ballot in ballots:
        new_ranking = []
        if len(remove_set) == 1 and remove_set in ballot.ranking:
            for s in ballot.ranking:
                new_s = s.difference(remove_set)
                if new_s:
                    new_ranking.append(new_s)
            update.append(
                Ballot(
                    id=ballot.id,
                    ranking=tuple(new_ranking),
                    weight=ballot.weight,
                    voter_set=ballot.voter_set,
                )
            )
        elif len(remove_set) > 1:
            for s in ballot.ranking:
                new_s = s.difference(remove_set)
                if new_s:
                    new_ranking.append(new_s)
            update.append(
                Ballot(
                    id=ballot.id,
                    ranking=tuple(new_ranking),
                    weight=ballot.weight,
                    voter_set=ballot.voter_set,
                )
            )
        else:
            update.append(ballot)

    return update


# Summmary Stat functions
def first_place_votes(profile: PreferenceProfile, to_float: bool = False) -> dict:
    """
    Calculates first-place votes for a ``PreferenceProfile``.

    Args:
        profile (PreferenceProfile): Inputed PreferenceProfile of ballots.
        to_float (bool): If True, compute first place votes as floats instead of Fractions.
            Defaults to False.

    Returns:
        dict: Dictionary of candidates (keys) and first place vote totals (values).
    """
    cands = profile.get_candidates()
    ballots = profile.get_ballots()

    _, votes_dict = compute_votes(cands, ballots)

    if to_float:
        votes_dict = {k: float(v) for k, v in votes_dict.items()}
        return votes_dict
    else:
        return votes_dict


def mentions(profile: PreferenceProfile) -> dict:
    """
    Calculates total mentions for a ``PreferenceProfile``.

    Args:
        profile (PreferenceProfile): Inputed PreferenceProfile of ballots.

    Returns:
        dict: Dictionary of candidates (keys) and mention totals (values).
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
    profile: PreferenceProfile,
    ballot_length: Optional[int] = None,
    score_vector: Optional[list] = None,
) -> dict:
    """
    Calculates Borda scores for a ``PreferenceProfile``.

    Args:
        profile (PreferenceProfile): ``PreferenceProfile`` of ballots.
        ballot_length (int, optional): Length of a ballot. If None, length of longest ballot is
            used. Defaults to None.
        score_vector (list, optional): List of Borda weights. If None, vector is assigned
            :math:`(n,n-1,\dots,1)`. Defaults to None.

    Returns:
        dict: Dictionary of candidates (keys) and Borda scores (values).
    """
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


def unset(input_set: Union[set, frozenset]) -> Any:
    """
    Removes object from set or frozenset.

    Args:
        input_set (Union[set, frozenset]): Input (frozen)set.

    Returns:
        Any: If (frozen)set has length one, return the object. Else return a list.
    """
    rv = list(input_set)

    if len(rv) == 1:
        return rv[0]

    return rv


def candidate_position_dict(ranking: list[set[str]]) -> dict:
    """
    Creates a dictionary with the integer ranking of candidates given a set ranking
    i.e. A > {B, C} > D returns {A: 1, B: 2, C: 2, D: 4}.

    Args:
        ranking (list[set[str]]): A list-of-sets ranking of candidates.

    Returns:
        dict: Dictionary of candidates (keys) and integer rankings (values).
    """
    candidate_positions = {}
    position = 0

    for tie_set in ranking:
        for candidate in tie_set:
            candidate_positions[candidate] = position
        position += len(tie_set)

    return candidate_positions


def tie_broken_ranking(
    ranking: list[set[str]], profile: PreferenceProfile, tiebreak: str = "none"
) -> list[set[str]]:
    """
    Breaks ties in a list-of-sets ranking according to a given scheme.

    Args:
        ranking (list[set[str]]): A list-of-set ranking of candidates.
        profile (PreferenceProfile): PreferenceProfile.
        tiebreak (str, optional): Method of tiebreak, currently supports 'none', 'random', 'borda',
            'firstplace'. Defaults to 'none'.

    Returns:
        list[set[str]]: A list-of-set ranking of candidates (broken down to one candidate sets
        unless tiebreak = 'none').
    """

    new_ranking = []
    if tiebreak == "none":
        new_ranking = ranking
    elif tiebreak == "random":
        for s in ranking:
            shuffled_s = list(np.random.permutation(list(s)))
            new_ranking += [{c} for c in shuffled_s]
    elif tiebreak == "firstplace":
        tiebreak_scores = first_place_votes(profile)
        for s in ranking:
            ordered_set = scores_into_set_list(tiebreak_scores, s)
            new_ranking += ordered_set
    elif tiebreak == "borda":
        tiebreak_scores = borda_scores(profile)
        for s in ranking:
            ordered_set = scores_into_set_list(tiebreak_scores, s)
            new_ranking += ordered_set
    else:
        raise ValueError("Invalid tiebreak code was provided")

    if tiebreak != "none" and any(len(s) > 1 for s in new_ranking):
        print("Initial tiebreak was unsuccessful, performing random tiebreak")
        new_ranking = tie_broken_ranking(
            ranking=new_ranking, profile=profile, tiebreak="random"
        )

    return new_ranking


def scores_into_set_list(
    score_dict: dict, candidate_subset: Union[list[str], set[str], None] = None
) -> list[set[str]]:
    """
    Sorts candidates based on a scoring dictionary (i.e. Borda, First-Place).

    Args:
        score_dict (dict): Dictionary between candidates (key) and their score (value).
        candidate_subset (Union[list[str], set[str], None], optional): Relevant candidates to sort.
            Defaults to None.

    Returns:
        list[set[str]]: Candidate rankings in a list-of-sets form.
    """
    if isinstance(candidate_subset, list):
        candidate_subset = set(candidate_subset)

    tier_dict: dict = {}
    for k, v in score_dict.items():
        if v in tier_dict.keys():
            tier_dict[v].add(k)
        else:
            tier_dict[v] = {k}
    tier_list = [tier_dict[k] for k in sorted(tier_dict.keys(), reverse=True)]
    if candidate_subset is not None:
        tier_list = [
            t & candidate_subset for t in tier_list if len(t & candidate_subset) > 0
        ]
    return tier_list


def compute_scores_from_vector(
    profile: PreferenceProfile, score_vector: list[float]
) -> dict:
    """
    Computes the scores received by each candidate given the score vector and the profile.

    Args:
        profile (PreferenceProfile): ``PreferenceProfile`` to compute scores for.
        score_vector (list[float]): List of floats where :math:`i`th position denotes points given
            to candidates in position :math:`i`.

    Returns:
        dict: A dictionary whose keys are candidates and values are scores.
    """
    # check for valid score vector
    validate_score_vector(score_vector)

    candidates_to_scores = {c: 0.0 for c in profile.get_candidates()}

    for ballot in profile.ballots:
        for i, s in enumerate(ballot.ranking):
            # for each candidate in position i, give them points as determined by score_vector
            for c in s:
                try:
                    candidates_to_scores[c] += score_vector[i] * ballot.weight
                except IndexError:
                    warnings.warn(
                        f"Tried to access index {i} of score vector,"
                        f"but vector only length {len(score_vector)}. "
                        "Assigned candidate 0 points.",
                        UserWarning,
                    )

    return candidates_to_scores


def validate_score_vector(score_vector: list[float]):
    """
    Validator function for score vectors. Vectors should be non-increasing and non-negative.

    Args:
        score_vector (list[float]): Score vector.

    Raises:
        ValueError: If any score is negative.
        ValueError: If score vector is increasing at any point.

    """

    for i, score in enumerate(score_vector):
        # if score is negative
        if score < 0:
            raise ValueError("Score vector must be non-negative.")

        if i > 0:
            # if the current score is bigger than prev
            if score > score_vector[i - 1]:
                raise ValueError("Score vector must be non-increasing.")


def elect_cands_from_set_ranking(
    ranking: list[set[str]], seats: int
) -> tuple[list[set[str]], list[set[str]]]:
    """
    Splits a ranking into elected and eliminated based on seats,
    and if a tie set overlaps the desired number of seats raises a ValueError.

    Args:
        ranking (list[set[str]]): A list-of-set ranking of candidates.
        seats (int): Number of seats to fill.

    Returns:
        tuple[list[set[str]], list[set[str]]]: A list-of-sets of elected candidates, a list-of-sets
        of eliminated candidates.
    """
    cands_elected = 0
    elected = []
    eliminated = []

    for i, s in enumerate(ranking):
        if cands_elected + len(s) <= seats:
            cands_elected += len(s)
            elected.append(s)
        else:
            eliminated = ranking[i:]
            break

    if cands_elected != seats:
        raise ValueError(
            "Cannot elect correct number of candidates without breaking ties."
        )

    return elected, eliminated


# helper functions for Election base class
def recursively_fix_ties(ballot_lst: list[Ballot], num_ties: int) -> list[Ballot]:
    """
    Recursively fixes ties in a ballot in the case there is more then one tie.

    Args:
        ballot_lst (list): List of Ballot objects.
        num_ties (int):  Number of ties to resolve.

    Returns:
        list[Ballot]: A list of Ballots with ties resolved.
    """
    # base case, if only one tie to resolved return the list of already
    # resolved ballots
    if num_ties == 1:
        return ballot_lst

    # in the event multiple positions have ties
    else:
        update = set()
        for ballot in ballot_lst:
            update.update(set(fix_ties(ballot)))

        return recursively_fix_ties(list(update), num_ties - 1)


def fix_ties(ballot: Ballot) -> list[Ballot]:
    """
    Helper function for ``recursively_fix_ties``. Resolves the first appearing
    tied rank in the input ballot.

    Args:
        ballot (Ballot): A Ballot.

    Returns:
        list[Ballot]: List of Ballots that are permutations of the tied ballot.
    """

    ballots = []
    for idx, rank in enumerate(ballot.ranking):
        if len(rank) > 1:
            for order in permutations(rank):
                resolved = []
                for cand in order:
                    resolved.append(frozenset(cand))
                ballots.append(
                    Ballot(
                        id=ballot.id,
                        ranking=ballot.ranking[:idx]
                        + tuple(resolved)
                        + ballot.ranking[idx + 1 :],
                        weight=ballot.weight / math.factorial(len(rank)),
                        voter_set=ballot.voter_set,
                    )
                )

    return ballots
