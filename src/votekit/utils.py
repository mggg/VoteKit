from fractions import Fraction
from typing import Union, Sequence, Optional, TypeVar, cast
from itertools import permutations
import math
import random
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
    cand_dict: dict[str, list[Ballot]] = {c: [] for c in profile.candidates}

    for b in profile.ballots:
        if not b.ranking:
            raise TypeError("Ballots must have rankings.")
        else:
            # find first place candidate, ensure there is only one
            first_cand = list(b.ranking[0])
            if len(first_cand) > 1:
                raise ValueError(f"Ballot {b} has a tie for first.")

            cand_dict[first_cand[0]].append(b)

    return cand_dict


COB = TypeVar("COB", PreferenceProfile, tuple[Ballot, ...], Ballot)


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
        condense (bool, optional): Whether or not to return a condensed profile. Defaults to True.
        leave_zero_weight_ballots (bool, optional): Whether or not to leave ballots with zero
            weight in the PreferenceProfile. Defaults to False.

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


def add_missing_cands(profile: PreferenceProfile) -> PreferenceProfile:
    """
    Add any candidates from `profile.candidates` that are not listed on a ballot
    as tied in last place. Helper function for scoring profiles. Automatically
    condenses profile.

    Args:
        profile (PreferenceProfile): Input profile.

    Returns:
        PreferenceProfile
    """

    new_ballots = [Ballot()] * len(profile.ballots)
    candidates = set(profile.candidates)

    for i, ballot in enumerate(profile.ballots):
        if not ballot.ranking:
            raise TypeError("Ballots must have rankings.")
        else:
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

    return PreferenceProfile(
        ballots=tuple(new_ballots), candidates=tuple(candidates)
    ).condense_ballots()


def validate_score_vector(score_vector: Sequence[Union[float, Fraction]]):
    """
    Validator function for score vectors. Vectors should be non-increasing and non-negative.

    Args:
        score_vector (Sequence[Union[float, Fraction]]): Score vector.

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


def score_profile_from_rankings(
    profile: PreferenceProfile,
    score_vector: Sequence[Union[float, Fraction]],
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
        score_vector (Sequence[Union[float, Fraction]]): Score vector. Should be
            non-increasing and non-negative. Vector should be as long as the number of candidates.
            If it is shorter, we add 0s.
        to_float (bool, optional): If True, compute scores as floats instead of Fractions.
            Defaults to False.

    Returns:
        Union[dict[str, Fraction], dict[str, float]]:
            Dictionary mapping candidates to scores.
    """
    validate_score_vector(score_vector)

    max_length = len(profile.candidates)
    if len(score_vector) < max_length:
        score_vector = list(score_vector) + [0] * (max_length - len(score_vector))

    profile = add_missing_cands(profile)

    scores = {c: Fraction(0) for c in profile.candidates}
    for ballot in profile.ballots:
        current_ind = 0
        if not ballot.ranking:
            raise TypeError("Ballots must have rankings.")
        else:
            for s in ballot.ranking:
                position_size = len(s)
                if len(s) == 0:
                    raise TypeError(f"Ballot {ballot} has an empty ranking position.")
                local_score_vector = score_vector[
                    current_ind : current_ind + position_size
                ]
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
    return score_profile_from_rankings(
        profile, [1] + [0] * len(profile.candidates), to_float
    )


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
    mentions = {c: Fraction(0) for c in profile.candidates}

    for ballot in profile.ballots:
        if not ballot.ranking:
            raise TypeError("Ballots must have rankings.")
        else:
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
    r"""
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
    score_vector = list(range(len(profile.candidates), 0, -1))

    return score_profile_from_rankings(profile, score_vector, to_float)


def tiebreak_set(
    r_set: frozenset[str],
    profile: Optional[PreferenceProfile] = None,
    tiebreak: str = "random",
) -> tuple[frozenset[str], ...]:
    """
    Break a single set of candidates into multiple sets each with a single candidate according
    to a tiebreak rule. Rule 1: random. Rule 2: first-place votes; break the tie based on
    first-place votes in the profile. Rule 3: borda; break the tie based on Borda points in the
    profile.

    Args:
        r_set (frozenset[str]): Set of candidates on which to break tie.
        profile (PreferenceProfile, optional): Profile used to break ties in first-place votes or
            Borda setting. Defaults to None, which implies a random tiebreak.
        tiebreak (str, optional): Tiebreak method to use. Options are "random", "first_place", and
            "borda". Defaults to "random".

    Returns:
        tuple[frozenset[str],...]: tiebroken ranking
    """
    if tiebreak == "random":
        new_ranking = tuple(
            frozenset({c}) for c in random.sample(list(r_set), k=len(r_set))
        )
    elif (tiebreak == "first_place" or tiebreak == "borda") and profile:
        if tiebreak == "borda":
            tiebreak_scores = borda_scores(profile)
        else:
            tiebreak_scores = first_place_votes(profile)
        tiebreak_scores = {
            c: Fraction(score) for c, score in tiebreak_scores.items() if c in r_set
        }
        new_ranking = score_dict_to_ranking(tiebreak_scores)

    elif not profile:
        raise ValueError("Method of tiebreak requires profile.")
    else:
        raise ValueError("Invalid tiebreak code was provided")

    if any(len(s) > 1 for s in new_ranking):
        print("Initial tiebreak was unsuccessful, performing random tiebreak")
        new_ranking, _ = tiebroken_ranking(
            new_ranking, profile=profile, tiebreak="random"
        )

    return new_ranking


def tiebroken_ranking(
    ranking: tuple[frozenset[str], ...],
    profile: Optional[PreferenceProfile] = None,
    tiebreak: str = "random",
) -> tuple[
    tuple[frozenset[str], ...], dict[frozenset[str], tuple[frozenset[str], ...]]
]:
    """
    Breaks ties in a list-of-sets ranking according to a given scheme.

    Args:
        ranking (list[set[str]]): A list-of-set ranking of candidates.
        profile (PreferenceProfile, optional): Profile used to break ties in first-place votes or
            Borda setting. Defaults to None, which implies a random tiebreak.
        tiebreak (str, optional): Method of tiebreak, currently supports 'random', 'borda',
            'first_place'. Defaults to random.

    Returns:
        tuple[tuple[frozenset[str], ...], dict[frozenset[str], tuple[frozenset[str],...]]]:
            The first entry of the tuple is a list-of-set ranking of candidates (broken down to one
            candidate sets). The second entry is a dictionary that maps tied sets to their
            resolution.
    """
    new_ranking: list[frozenset[str]] = [frozenset()] * len(
        [c for s in ranking for c in s]
    )

    i = 0
    tied_dict = {}
    for s in ranking:
        if len(s) > 1:
            tiebroken = tiebreak_set(s, profile, tiebreak)
            tied_dict[s] = tiebroken
        else:
            tiebroken = (s,)
        new_ranking[i : (i + len(tiebroken))] = tiebroken
        i += len(tiebroken)

    return (tuple(new_ranking), tied_dict)


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

    if len(score_to_cand) > 0:
        return tuple(
            [
                frozenset(c_list)
                for _, c_list in sorted(
                    score_to_cand.items(), key=lambda x: x[0], reverse=sort_high_low
                )
            ]
        )
    else:
        return (frozenset(),)


def elect_cands_from_set_ranking(
    ranking: tuple[frozenset[str], ...],
    m: int,
    profile: Optional[PreferenceProfile] = None,
    tiebreak: Optional[str] = None,
) -> tuple[
    tuple[frozenset[str], ...],
    tuple[frozenset[str], ...],
    Optional[tuple[frozenset[str], tuple[frozenset[str], ...]]],
]:
    """
    Given a ranking, elect the top m candidates in the ranking.
    If a tie set overlaps the desired number of seats, it breaks the tie with the provided
    method or raises a ValueError if tiebreak is set to None.
    Returns a tuple of elected candidates, remaining candidates, and a tuple whose first entry
    is a tie set and whose second entry is the resolution of the tie.

    Args:
        ranking (tuple[frozenset[str],...]): A list-of-set ranking of candidates.
        m (int): Number of seats to elect.
        profile (PreferenceProfile, optional): Profile used to break ties in first-place votes or
            Borda setting. Defaults to None, which implies a random tiebreak.
        tiebreak (str, optional): Method of tiebreak, currently supports 'random', 'borda',
            'first_place'. Defaults to None, which does not break ties.

    Returns:
        tuple[tuple[frozenset[str]]], list[tuple[frozenset[str]],
            Optional[tuple[frozenset[str], tuple[frozenset[str], ...]]]:
            A list-of-sets of elected candidates, a list-of-sets of remaining candidates,
            and a tuple whose first entry is a tie set and whose second entry is the resolution of
            the tie. If no ties were broken, the tuple returns None.
    """
    if m < 1:
        raise ValueError("m must be strictly positive")

    # if there are more seats than candidates
    if m > len([c for s in ranking for c in s]):
        raise ValueError("m must be no more than the number of candidates.")

    num_elected = 0
    elected = []
    i = 0
    tiebreak_ranking = None

    while num_elected < m:
        elected.append(ranking[i])
        num_elected += len(ranking[i])
        if num_elected > m:
            if not tiebreak:
                raise ValueError(
                    "Cannot elect correct number of candidates without breaking ties."
                )
            else:
                elected.pop(-1)
                num_elected -= len(ranking[i])
                tiebroken_ranking = tiebreak_set(ranking[i], profile, tiebreak)
                elected += tiebroken_ranking[: (m - num_elected)]
                remaining = list(tiebroken_ranking[(m - num_elected) :])
                if i < len(ranking):
                    remaining += list(ranking[(i + 1) :])

                return (
                    tuple(elected),
                    tuple(remaining),
                    (ranking[i], tiebroken_ranking),
                )

        i += 1

    return (tuple(elected), ranking[i:], tiebreak_ranking)


def expand_tied_ballot(ballot: Ballot) -> list[Ballot]:
    """
    Fix tie(s) in a ballot by returning all possible permutations of the tie(s), and divide the
    weight of the original ballot equally among the new ballots.

    Args:
        ballot (Ballot): Ballot to expand tie sets on.

    Returns:
        list[Ballot]: All possible permutations of the tie(s).

    """
    if not ballot.ranking:
        raise TypeError("Ballot must have ranking.")
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

    new_ballots = tuple(
        [b for ballot in profile.ballots for b in expand_tied_ballot(ballot)]
    )
    return PreferenceProfile(ballots=new_ballots).condense_ballots()


def score_profile_from_ballot_scores(
    profile: PreferenceProfile,
    to_float: bool = False,
) -> Union[dict[str, Fraction], dict[str, float]]:
    """
    Score the candidates based on the ``scores`` parameter of the ballots.
    All ballots must have a ``scores`` parameter; note that a ``scores`` dictionary
    with no non-zero scores will raise the same error.

    Args:
        profile (PreferenceProfile): Profile to score.
        to_float (bool, optional): If True, compute scores as floats instead of Fractions.
            Defaults to False.

    Returns:
        Union[dict[str, Fraction], dict[str, float]]:
            Dictionary mapping candidates to scores.
    """
    scores = {c: Fraction(0) for c in profile.candidates}
    for ballot in profile.ballots:
        if not ballot.scores:
            raise TypeError(f"Ballot {ballot} has no scores.")
        else:
            for c, score in ballot.scores.items():
                scores[c] += score * ballot.weight

    if to_float:
        return {c: float(v) for c, v in scores.items()}
    return scores
