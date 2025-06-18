from typing import Sequence, Optional, Literal
from itertools import permutations
import math
import random
from .ballot import Ballot
from .pref_profile import PreferenceProfile, ProfileError
import pandas as pd
import numpy as np

COLOR_LIST = [
    "#0099cd",
    "#ffca5d",
    "#00cd99",
    "#99cd00",
    "#cd0099",
    "#9900cd",
    "#8dd3c7",
    "#bebada",
    "#fb8072",
    "#80b1d3",
    "#fdb462",
    "#b3de69",
    "#fccde5",
    "#bc80bd",
    "#ccebc5",
    "#ffed6f",
    "#ffffb3",
    "#a6cee3",
    "#1f78b4",
    "#b2df8a",
    "#33a02c",
    "#fb9a99",
    "#e31a1c",
    "#fdbf6f",
    "#ff7f00",
    "#cab2d6",
    "#6a3d9a",
    "#b15928",
    "#64ffda",
    "#00B8D4",
    "#A1887F",
    "#76FF03",
    "#DCE775",
    "#B388FF",
    "#FF80AB",
    "#D81B60",
    "#26A69A",
    "#FFEA00",
    "#6200EA",
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
    if not profile.contains_rankings:
        raise TypeError("Ballots must have rankings.")

    df = profile.df
    ranking_cols = [f"Ranking_{i}" for i in range(1, profile.max_ranking_length + 1)]

    rank_arr = df[ranking_cols].to_numpy()
    weights = df["Weight"].to_numpy()

    cand_dict: dict[str, list[Ballot]] = {c: [] for c in profile.candidates}
    tilde = frozenset({"~"})

    for row, w in zip(rank_arr, weights):
        first = row[0]

        if len(first) > 1:
            raise ValueError(
                f"Ballot "
                f"{Ballot(ranking=tuple(c_set for c_set in row if c_set != tilde), weight=w)} "
                "has a tie for first."
            )

        cand = next(iter(first))

        if cand == "~":
            continue

        clean_ranking = tuple(s for s in row if s != tilde)

        cand_dict[cand].append(Ballot(ranking=clean_ranking, weight=w))

    return cand_dict


def add_missing_cands(profile: PreferenceProfile) -> PreferenceProfile:
    """
    Add any candidates from `profile.candidates` that are not listed on a ballot
    as tied in last place.

    Args:
        profile (PreferenceProfile): Input profile.

    Returns:
        PreferenceProfile
    """

    new_ballots = [Ballot()] * len(profile.ballots)
    candidates = set(profile.candidates)

    for i, ballot in enumerate(profile.ballots):
        if ballot.ranking is None:
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
                weight=ballot.weight,
                voter_set=ballot.voter_set,
                ranking=tuple([frozenset(s) for s in new_ranking]),
            )

    return PreferenceProfile(ballots=tuple(new_ballots), candidates=tuple(candidates))


def validate_score_vector(score_vector: Sequence[float]):
    """
    Validator function for score vectors. Vectors should be non-increasing and non-negative.

    Args:
        score_vector (Sequence[float]): Score vector.

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


def _score_dict_from_rankings_df_no_ties(
    profile: PreferenceProfile,
    score_vector: Sequence[float],
) -> dict[str, float]:

    validate_score_vector(score_vector)

    if profile.contains_scores:
        raise ProfileError("Profile must only contain ranked ballots.")

    max_len = profile.max_ranking_length
    if len(score_vector) < max_len:
        score_vector = list(score_vector) + [0] * (max_len - len(score_vector))

    df = profile.df

    cand_frznst = [frozenset({c}) for c in profile.candidates_cast]
    all_frznst = cand_frznst + [frozenset({"~"}), frozenset()]
    n_buckets = len(all_frznst)
    idx_of_empty = all_frznst.index(frozenset())

    # Pull out weights and score_vector as NumPy arrays:
    weights = df["Weight"].to_numpy(dtype=float)
    score_arr = np.array(score_vector, dtype=float).reshape(1, -1)

    rank_cols = [f"Ranking_{i}" for i in range(1, max_len + 1)]
    arr = df[rank_cols].to_numpy(dtype=object)
    flat_arr = arr.ravel()

    # Slick way of converting frozensets to integer codes:
    codes_flat = pd.Categorical(flat_arr, categories=all_frznst).codes.astype(np.int64)

    # Take care of error codes (-1)
    if (codes_flat == -1).any():
        codes_flat = np.where(codes_flat == -1, idx_of_empty, codes_flat)

    weight_matrix = weights[:, None] * score_arr

    weights_flat = weight_matrix.ravel()
    bucket_sums = np.bincount(codes_flat, weights=weights_flat, minlength=n_buckets)

    return {next(iter(k)): bucket_sums[idx] for idx, k in enumerate(cand_frznst)}


def score_profile_from_rankings(
    profile: PreferenceProfile,
    score_vector: Sequence[float],
    tie_convention: Literal["high", "average", "low"] = "low",
) -> dict[str, float]:
    """
    Score the candidates based on a score vector. For example, the vector (1,0,...) would
    return the first place votes for each candidate. Vectors should be non-increasing and
    non-negative. Vector should be as long as ``max_ranking_length`` in the profile.
    If it is shorter, we add 0s. Candidates who are not mentioned in any ranking do not appear
    in the dictionary.


    Args:
        profile (PreferenceProfile): Profile to score.
        score_vector (Sequence[float]): Score vector. Should be
            non-increasing and non-negative. Vector should be as long as ``max_ranking_length`` in
            the profile. If it is shorter, we add 0s.
        tie_convention (Literal["high", "average", "low"], optional): How to award points for
            tied rankings. Defaults to "low", where any candidates tied receive the lowest possible
            points for their position, eg three people tied for 3rd would each receive the points
            for 5th. "high" awards the highest possible points, so in the previous example, they
            would each receive the points for 3rd. "average" averages the points, so they would each
            receive the points for 4th place.

    Returns:
        dict[str, float]:
            Dictionary mapping candidates to scores.
    """
    validate_score_vector(score_vector)

    if profile.contains_scores is True:
        raise ProfileError("Profile must only contain ranked ballots.")
    max_length = profile.max_ranking_length
    if len(score_vector) < max_length:
        score_vector = list(score_vector) + [0] * (max_length - len(score_vector))

    scores = {c: 0.0 for c in profile.candidates_cast}

    try:
        ranking_cols = [f"Ranking_{i}" for i in range(1, max_length + 1)]
        ranking_mat = profile.df[ranking_cols].to_numpy()
    except KeyError as e:
        raise TypeError("Ballots must have rankings.") from e

    weights = profile.df["Weight"].to_numpy(dtype=float)

    if tie_convention not in ["high", "average", "low"]:
        raise ValueError(
            (
                "tie_convention must be one of 'high', 'low', 'average', "
                f"not {tie_convention}"
            )
        )

    tilde = frozenset({"~"})
    for idx in range(len(ranking_mat)):
        current_ind = 0
        ranking = ranking_mat[idx]
        wt = weights[idx]
        for s in ranking:
            position_size = len(s)
            if position_size == 0:
                raise TypeError(
                    f"Ballot {Ballot(ranking=ranking.tolist(), weight=wt)} has an empty ranking "
                    "position."
                )
            if s == tilde:
                continue

            local_score_vector = score_vector[current_ind : current_ind + position_size]

            if tie_convention == "high":
                allocation = max(local_score_vector)
            elif tie_convention == "low":
                allocation = min(local_score_vector)
            else:
                allocation = sum(local_score_vector) / position_size

            for c in s:
                scores[c] += allocation * wt
            current_ind += position_size

    return scores


def _first_place_votes_from_df_no_ties(
    profile: PreferenceProfile,
) -> dict[str, float]:
    """
    Computes first place votes for all candidates_cast in a ``PreferenceProfile``.
    Intended to be much faster than first_place_votes, but does not handle ties in ballots.

    Args:
        profile (PreferenceProfile): The profile to compute first place votes for.

    Returns:
        dict[str, float]:
            Dictionary mapping candidates to number of first place votes.
    """
    # equiv to score vector of (1,0,0,...)
    return _score_dict_from_rankings_df_no_ties(
        profile, [1] + [0] * (profile.max_ranking_length - 1)
    )


def first_place_votes(
    profile: PreferenceProfile,
    tie_convention: Literal["high", "average", "low"] = "average",
) -> dict[str, float]:
    """
    Computes first place votes for all candidates_cast in a ``PreferenceProfile``.

    Args:
        profile (PreferenceProfile): The profile to compute first place votes for.
        tie_convention (Literal["high", "average", "low"], optional): How to award points
            for tied first place votes. Defaults to "average", where if n candidates are tied for
            first, each receives 1/n points. "high" would award them each one point, and "low" 0.

    Returns:
        dict[str, float]:
            Dictionary mapping candidates to number of first place votes.
    """
    # equiv to score vector of (1,0,0,...)
    return score_profile_from_rankings(
        profile, [1] + [0] * (profile.max_ranking_length - 1), tie_convention
    )


def mentions(
    profile: PreferenceProfile,
) -> dict[str, float]:
    """
    Calculates total mentions for all candidates in a ``PreferenceProfile``.

    Args:
        profile (PreferenceProfile): PreferenceProfile of ballots.

    Returns:
        dict[str, float]:
            Dictionary mapping candidates to mention totals (values).
    """
    mentions = {c: 0.0 for c in profile.candidates}

    for ballot in profile.ballots:
        if ballot.ranking is None:
            raise TypeError("Ballots must have rankings.")
        else:
            for s in ballot.ranking:
                for cand in s:
                    mentions[cand] += ballot.weight
    return mentions


def borda_scores(
    profile: PreferenceProfile,
    borda_max: Optional[int] = None,
    tie_convention: Literal["high", "average", "low"] = "low",
) -> dict[str, float]:
    r"""
    Calculates Borda scores for candidates_cast in a ``PreferenceProfile``. The Borda vector is
    :math:`(n,n-1,\dots,1)` where :math:`n` is the ``borda_max`.

    Args:
        profile (PreferenceProfile): ``PreferenceProfile`` of ballots.
        borda_max (int, optional): The maximum value of the Borda vector. Defaults to
            the length of the longest allowable ballot in the profile.
        tie_convention (Literal["high", "average", "low"], optional): How to award points for
            tied rankings. Defaults to "low", where any candidates tied receive the lowest possible
            points for their position, eg three people tied for 3rd would each receive the points
            for 5th. "high" awards the highest possible points, so in the previous example, they
            would each receive the points for 3rd. "average" averages the points, so they would each
            receive the points for 4th place.

    Returns:
        dict[str, float]:
            Dictionary mapping candidates to Borda scores.
    """
    if borda_max is None:
        borda_max = profile.max_ranking_length

    score_vector = list(range(borda_max, 0, -1))

    return score_profile_from_rankings(profile, score_vector, tie_convention)


def tiebreak_set(
    r_set: frozenset[str],
    profile: Optional[PreferenceProfile] = None,
    tiebreak: str = "random",
    scoring_tie_convention: Literal["high", "average", "low"] = "low",
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
        scoring_tie_convention (Literal["high", "average", "low"], optional): How to award points
            for tied rankings. Defaults to "low", where any candidates tied receive the lowest
            possible points for their position, eg three people tied for 3rd would each receive the
            points for 5th. "high" awards the highest possible points, so in the previous example,
            they would each receive the points for 3rd. "average" averages the points, so they would
            each receive the points for 4th place.

    Returns:
        tuple[frozenset[str],...]: tiebroken ranking
    """
    if tiebreak == "random":
        new_ranking = tuple(
            frozenset({c}) for c in random.sample(list(r_set), k=len(r_set))
        )
    elif (tiebreak == "first_place" or tiebreak == "borda") and profile:
        if tiebreak == "borda":
            tiebreak_scores = borda_scores(
                profile, tie_convention=scoring_tie_convention
            )
        else:
            tiebreak_scores = first_place_votes(
                profile, tie_convention=scoring_tie_convention
            )
        tiebreak_scores = {
            c: score for c, score in tiebreak_scores.items() if c in r_set
        }
        new_ranking = score_dict_to_ranking(tiebreak_scores)

    elif profile is None:
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
    score_dict: dict[str, float], sort_high_low: bool = True
) -> tuple[frozenset[str], ...]:
    """
    Sorts candidates into a tuple of frozensets ranking based on a scoring dictionary.

    Args:
        score_dict (dict[str, float]): Dictionary between candidates
            and their score.
        sort_high_low (bool, optional): How to sort candidates based on scores. True sorts
            from high to low. Defaults to True.


    Returns:
        tuple[frozenset[str],...]: Candidate rankings in a list-of-sets form.
    """

    score_to_cand: dict[float, list[str]] = {s: [] for s in score_dict.values()}
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
            if tiebreak is None:
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
    if ballot.ranking is None:
        raise TypeError("Ballot must have ranking.")
    if all(len(s) == 1 for s in ballot.ranking):
        return [ballot]

    else:
        for i, s in enumerate(ballot.ranking):
            if len(s) > 1:
                new_ballots = [
                    Ballot(
                        weight=ballot.weight / math.factorial(len(s)),
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
    all permutations of the tied ranking.

    Args:
        profile (PreferenceProfile): Input profile with potentially tied rankings.

    Returns:
        PreferenceProfile: A PreferenceProfile with resolved ties.
    """

    new_ballots = tuple(
        [b for ballot in profile.ballots for b in expand_tied_ballot(ballot)]
    )
    return PreferenceProfile(ballots=new_ballots)


def score_profile_from_ballot_scores(
    profile: PreferenceProfile,
) -> dict[str, float]:
    """
    Score the candidates based on the ``scores`` parameter of the ballots.
    All ballots must have a ``scores`` parameter; note that a ``scores`` dictionary
    with no non-zero scores will raise the same error.

    Args:
        profile (PreferenceProfile): Profile to score.

    Returns:
        dict[str, float]:
            Dictionary mapping candidates to scores.
    """
    scores = {c: 0.0 for c in profile.candidates}
    for ballot in profile.ballots:
        if ballot.scores is None:
            raise TypeError(f"Ballot {ballot} has no scores.")
        else:
            for c, score in ballot.scores.items():
                scores[c] += score * ballot.weight

    return scores


def ballot_lengths(profile: PreferenceProfile) -> dict[int, float]:
    """
    Compute the frequency of ballot lengths in the profile.
    Includes all lengths from 1 to ``max_ranking_length`` as keys.
    Ballots must have rankings.

    Args:
        profile (PreferenceProfile): Profile to compute ballot lengths.

    Returns:
        dict[int, float]: Dictionary of ballot length frequency.

    Raises:
        TypeError: All ballots must have rankings.
    """

    ballot_lengths = {i: 0.0 for i in range(1, profile.max_ranking_length + 1)}

    for ballot in profile.ballots:
        if ballot.ranking is None:
            raise TypeError("All ballots must have rankings.")

        length = len(ballot.ranking)
        ballot_lengths[length] += ballot.weight

    return ballot_lengths
