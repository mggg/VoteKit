from fractions import Fraction
from typing import Union, Sequence
from itertools import permutations
import math
import random
from .ballot import Ballot
from .pref_profile import PreferenceProfile


def ballots_by_first_cand(profile: PreferenceProfile) -> dict[str, list[Ballot]]:
    """
    Partitions the profile by first place candidate. Assumes there are no ties within first place
    positions of ballots.

    Args:
        profile (PreferenceProfile): Profile to partititon.

    Returns:
        dict[str, list[Ballot]]:
            A dictionary whose keys are candidates and values are lists of ballots that
            have that candidate first.
    """
    cand_dict: dict[str, list[Ballot]] = {c: [] for c in profile.get_candidates()}

    for b in profile.ballots:
        # find first place candidate, ensure there is only one
        first_cand = list(b.ranking[0])
        if len(first_cand) > 1:
            raise ValueError(f"Ballot {b} has a tie for first.")

        cand_dict[first_cand[0]].append(b)

    return cand_dict


def remove_cand(
    removed: Union[str, list], profile: PreferenceProfile
) -> PreferenceProfile:
    """
    Removes specified candidate(s) from profile. When a candidate is removed from a ballot, lower
    ranked candidates are moved up. Automatically condenses the profile.

    Args:
        removed (Union[str, list]): Candidate or list of candidates to be removed.
        profile (PreferenceProfile): Profile to remove candidates from.

    Returns:
        PreferenceProfile: Updated profile of ballots with candidate(s) removed.
    """
    if isinstance(removed, str):
        removed = [removed]

    scrubbed_ballots: list[Union[int, Ballot]] = [-1] * len(profile.ballots)
    for i, ballot in enumerate(profile.ballots):
        new_ranking = []
        for s in ballot.ranking:
            new_s = []
            for c in s:
                if c not in removed:
                    new_s.append(c)
            if len(new_s) > 0:
                new_ranking.append(frozenset(new_s))
        if len(new_ranking) > 0:
            scrubbed_ballots[i] = Ballot(
                ranking=tuple(new_ranking), weight=ballot.weight
            )

    return PreferenceProfile(
        ballots=[b for b in scrubbed_ballots if isinstance(b, Ballot)]
    ).condense_ballots()


def add_missing_cands(profile: PreferenceProfile) -> PreferenceProfile:
    """
    Add any candidates from `profile.get_candidates()` that are not listed on a ballot
    as tied in last place. Helper function for scoring profiles. Automatically
    condenses profile.

    Args:
        profile (PreferenceProfile): Input profile.

    Returns:
        PreferenceProfile
    """

    new_ballots = [Ballot()] * len(profile.ballots)
    candidates = set(profile.get_candidates())

    for i, ballot in enumerate(profile.ballots):
        b_cands = [c for s in ballot.ranking for c in s]
        missing_cands = candidates.difference(b_cands)

        new_ranking = (
            list(ballot.ranking) + [missing_cands]
            if len(missing_cands) > 0
            else ballot.ranking
        )

        new_ballots[i] = Ballot(
            id=ballot.id,
            weight=ballot.weight,
            voter_set=ballot.voter_set,
            ranking=tuple([frozenset(s) for s in new_ranking]),
        )

    return PreferenceProfile(ballots=new_ballots).condense_ballots()


def validate_score_vector(score_vector: Sequence[Union[float, int, Fraction]]):
    """
    Validator function for score vectors. Vectors should be non-increasing and non-negative.

    Args:
        score_vector (Sequence[Union[float, int, Fraction]]): Score vector.

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


def score_profile(
    profile: PreferenceProfile,
    score_vector: Sequence[Union[float, int, Fraction]],
    to_float: bool = False,
) -> Union[dict[str, Fraction], dict[str, float]]:
    """
    Score the candidates based on a score vector. For example, the vector (1,0,...) would
    return the first place votes for each candidate. Vectors should be non-increasing and
    non-negative. Vector should be as long as the number of candidates. If it is shorter,
    we add 0s. Candidates tied in a position receive an average of the points they would have
    received had it been untied. Any candidates not listed on a ballot are considered tied in last
    place (and thus receive an average of any remaining points).


    Args:
        profile (PreferenceProfile): Profile to score.
        score_vector (Sequence[Union[float, int, Fraction]]): Score vector. Should be
            non-increasing and non-negative. Vector should be as long as the number of candidates.
            If it is shorter, we add 0s.
        to_float (bool, optional): If True, compute scores as floats instead of Fractions.
            Defaults to False.

    Returns:
        Union[dict[str, Fraction], dict[str, float]]: Dictionary mapping candidates to scores.
    """
    validate_score_vector(score_vector)

    max_length = len(profile.get_candidates())
    if len(score_vector) < max_length:
        score_vector = list(score_vector) + [0] * (max_length - len(score_vector))

    profile = add_missing_cands(profile)

    scores = {c: Fraction(0) for c in profile.get_candidates()}
    for ballot in profile.ballots:
        current_ind = 0
        for s in ballot.ranking:
            position_size = len(s)
            local_score_vector = score_vector[current_ind : current_ind + position_size]
            allocation = sum(local_score_vector) / position_size
            for c in s:
                scores[c] += Fraction(allocation) * ballot.weight
            current_ind += position_size

    if to_float:
        return {c: float(v) for c, v in scores.items()}
    return scores


def first_place_votes(
    profile: PreferenceProfile, to_float: bool = False
) -> Union[dict[str, Fraction], dict[str, float]]:
    """
    Computes first place votes for all candidates in a ``PreferenceProfile``.

    Args:
        profile (PreferenceProfile): The profile to compute first place votes for.
        to_float (bool): If True, compute first place votes as floats instead of Fractions.
            Defaults to False.

    Returns:
        Union[dict[str, Fraction],dict[str, float]]:
            Dictionary mapping candidates to number of first place votes.
    """
    # equiv to score vector of (1,0,0,...)
    return score_profile(profile, [1] + [0] * len(profile.get_candidates()), to_float)


def mentions(
    profile: PreferenceProfile, to_float: bool = False
) -> Union[dict[str, Fraction], dict[str, float]]:
    """
    Calculates total mentions for a ``PreferenceProfile``.

    Args:
        profile (PreferenceProfile): PreferenceProfile of ballots.
        to_float (bool): If True, compute mention as floats instead of Fractions.
            Defaults to False.

    Returns:
        Union[dict[str, Fraction], dict[str, float]]:
            Dictionary mapping candidates to mention totals (values).
    """
    mentions = {c: Fraction(0) for c in profile.get_candidates()}

    for ballot in profile.ballots:
        for s in ballot.ranking:
            for cand in s:
                mentions[cand] += ballot.weight
    if to_float:
        return {c: float(v) for c, v in mentions.items()}
    return mentions


def borda_scores(
    profile: PreferenceProfile,
    to_float: bool = False,
) -> Union[dict[str, Fraction], dict[str, float]]:
    """
    Calculates Borda scores for a ``PreferenceProfile``. The Borda vector is :math:`(n,n-1,\dots,1)`
    where :math:`n` is the number of candidates.

    Args:
        profile (PreferenceProfile): ``PreferenceProfile`` of ballots.
        to_float (bool): If True, compute Borda as floats instead of Fractions.
            Defaults to False.

    Returns:
        Union[dict[str, Fraction], dict[str, float]]:
            Dictionary mapping candidates to Borda scores.
    """
    score_vector = list(range(len(profile.get_candidates()), 0, -1))

    return score_profile(profile, score_vector, to_float)


def tie_broken_ranking(
    ranking: tuple[frozenset[str], ...],
    profile: PreferenceProfile,
    tiebreak: str = "none",
) -> tuple[frozenset[str], ...]:
    """
    Breaks ties in a list-of-sets ranking according to a given scheme.

    Args:
        ranking (list[set[str]]): A list-of-set ranking of candidates.
        profile (PreferenceProfile): PreferenceProfile.
        tiebreak (str, optional): Method of tiebreak, currently supports 'none', 'random', 'borda',
            'firstplace'. Defaults to 'none'.

    Returns:
        tuple[frozenset[str], ...]: A list-of-set ranking of candidates (broken down to one
        candidate sets unless tiebreak = 'none').
    """
    if tiebreak == "none":
        return ranking
    elif tiebreak == "random":
        new_ranking = tuple(
            frozenset({c}) for s in ranking for c in random.sample(list(s), k=len(s))
        )
    elif tiebreak == "firstplace":
        tiebreak_scores = first_place_votes(profile)
        new_ranking = score_dict_to_ranking(tiebreak_scores)
    elif tiebreak == "borda":
        tiebreak_scores = borda_scores(profile)
        new_ranking = score_dict_to_ranking(tiebreak_scores)
    else:
        raise ValueError("Invalid tiebreak code was provided")

    if tiebreak != "none" and any(len(s) > 1 for s in new_ranking):
        print("Initial tiebreak was unsuccessful, performing random tiebreak")
        new_ranking = tie_broken_ranking(
            ranking=new_ranking, profile=profile, tiebreak="random"
        )

    return new_ranking


def score_dict_to_ranking(
    score_dict: Union[dict[str, Fraction], dict[str, float]], sort_high_low: bool = True
) -> tuple[frozenset[str], ...]:
    """
    Sorts candidates into a tuple of frozensets ranking based on a scoring dictionary.

    Args:
        score_dict (Union[dict[str, Fraction],dict[str, float]]): Dictionary between candidates
            and their score.
        sort_high_low (bool, optional): How to sort candidates based on scores. True sorts
            from high to low. Defaults to True.


    Returns:
        tuple[frozenset[str],...]: Candidate rankings in a list-of-sets form.
    """

    score_to_cand: dict[Union[float, Fraction], list[str]] = {
        s: [] for s in score_dict.values()
    }
    for c, score in score_dict.items():
        score_to_cand[score].append(c)

    return tuple(
        [
            frozenset(c_list)
            for _, c_list in sorted(
                score_to_cand.items(), key=lambda x: x[0], reverse=sort_high_low
            )
        ]
    )


def elect_cands_from_set_ranking(
    ranking: tuple[frozenset[str]], m: int
) -> tuple[tuple[frozenset[str], ...], tuple[frozenset[str], ...]]:
    """
    Given a ranking, elect the top m candidates in the ranking.
    If a tie set overlaps the desired number of seats it raises a ValueError.
    Returns a tuple of elected candidates, remaining candidates.

    Args:
        ranking (tuple[frozenset[str]]): A list-of-set ranking of candidates.
        m (int): Number of seats to elect.

    Returns:
        tuple[tuple[frozenset[str]]], list[tuple[frozenset[str]]]:
            A list-of-sets of elected candidates, a list-of-sets of remaining candidates.
    """
    if m < 1:
        raise ValueError("m must be strictly positive")

    num_elected = 0
    elected = []
    i = 0

    while num_elected < m:
        elected.append(ranking[i])
        num_elected += len(ranking[i])
        i += 1

    if num_elected > m:
        raise ValueError(
            "Cannot elect correct number of candidates without breaking ties."
        )

    return (tuple(elected), ranking[i:])


def expand_tied_ballot(ballot: Ballot) -> list[Ballot]:
    """
    Fix tie(s) in a ballot by returning all possible permutations of the tie(s), and divide the
    weight of the original ballot equally among the new ballots.

    Args:
        ballot (Ballot): Ballot to expand tie sets on.

    Returns:
        list[Ballot]: All possible permutations of the tie(s).

    """

    if all(len(s) == 1 for s in ballot.ranking):
        return [ballot]

    else:
        for i, s in enumerate(ballot.ranking):
            if len(s) > 1:
                new_ballots = [
                    Ballot(
                        weight=ballot.weight / math.factorial(len(s)),
                        id=ballot.id,
                        voter_set=ballot.voter_set,
                        ranking=tuple(ballot.ranking[:i])
                        + tuple([frozenset({c}) for c in order])
                        + tuple(ballot.ranking[(i + 1) :]),
                    )
                    for order in permutations(s)
                ]

                return [b for new_b in new_ballots for b in expand_tied_ballot(new_b)]

        assert False  # mypy


def resolve_profile_ties(profile: PreferenceProfile) -> PreferenceProfile:
    """
    Takes in a PeferenceProfile with potential ties in ballots. Replaces
    ballots with ties with fractionally weighted ballots corresponding to
    all permutations of the tied ranking. Automatically condenses the ballots.

    Args:
        profile (PreferenceProfile): Input profile with potentially tied rankings.

    Returns:
        PreferenceProfile: A PreferenceProfile with resolved ties.
    """

    new_ballots = [b for ballot in profile.ballots for b in expand_tied_ballot(ballot)]
    return PreferenceProfile(ballots=new_ballots).condense_ballots()
