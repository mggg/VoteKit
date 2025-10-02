from typing import Sequence, Optional, Literal, Union
from itertools import permutations
import math
import random
from votekit.ballot import Ballot, RankBallot
from votekit.pref_profile import RankProfile, ScoreProfile
import pandas as pd
import numpy as np
from numpy.typing import NDArray

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


def ballots_by_first_cand(profile: RankProfile) -> dict[str, list[RankBallot]]:
    """
    Partitions the profile by first place candidate. Assumes there are no ties within first place
    positions of ballots.

    Args:
        profile (RankProfile): Profile to partititon.

    Returns:
        dict[str, list[RankBallot]]:
            A dictionary whose keys are candidates and values are lists of ballots that
            have that candidate first.
    """
    if not isinstance(profile, RankProfile):
        raise TypeError("Ballots must have rankings.")

    df = profile.df
    assert profile.max_ranking_length is not None
    ranking_cols = [f"Ranking_{i}" for i in range(1, profile.max_ranking_length + 1)]

    rank_arr = df[ranking_cols].to_numpy()
    weights = df["Weight"].to_numpy()
    voter_sets = df["Voter Set"].to_numpy().astype(object)

    cand_dict: dict[str, list[RankBallot]] = {c: [] for c in profile.candidates}
    tilde = frozenset({"~"})

    for row, w, voter_set in zip(rank_arr, weights, voter_sets):
        first = row[0]

        if len(first) > 1:
            raise ValueError(
                f"Ballot "
                f"{RankBallot(ranking=tuple(c_set for c_set in row if c_set != tilde), weight=float(w))} "
                "has a tie for first."
            )

        cand = next(iter(first))

        if cand == "~":
            continue

        clean_ranking = tuple(s for s in row if s != tilde)

        cand_dict[cand].append(
            RankBallot(ranking=clean_ranking, weight=float(w), voter_set=voter_set)
        )

    return cand_dict


def add_missing_cands(profile: RankProfile) -> RankProfile:
    """
    Add any candidates from `profile.candidates` that are not listed on a ballot
    as tied in last place.

    Args:
        profile (RankProfile): Input profile.

    Returns:
        RankProfile
    """
    if not isinstance(profile, RankProfile):
        raise TypeError("Profile must be of type RankProfile.")
    new_ballots = [RankBallot()] * len(profile.ballots)
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

            new_ballots[i] = RankBallot(
                weight=ballot.weight,
                voter_set=ballot.voter_set,
                ranking=tuple([frozenset(s) for s in new_ranking]),
            )

    return RankProfile(ballots=tuple(new_ballots), candidates=tuple(candidates))


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
    profile: RankProfile,
    score_vector: Sequence[float],
) -> dict[str, float]:

    validate_score_vector(score_vector)

    if not isinstance(profile, RankProfile):
        raise TypeError("Profile must only contain ranked ballots.")

    assert profile.max_ranking_length is not None
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
    codes_flat: NDArray[np.int64] = pd.Categorical(
        flat_arr, categories=all_frznst
    ).codes.astype(np.int64)

    # Take care of error codes (-1)
    if (codes_flat == -1).any():
        codes_flat = np.where(codes_flat == -1, idx_of_empty, codes_flat)

    weight_matrix = weights[:, None] * score_arr

    weights_flat = weight_matrix.ravel()
    bucket_sums = np.bincount(codes_flat, weights=weights_flat, minlength=n_buckets)

    return {next(iter(k)): bucket_sums[idx] for idx, k in enumerate(cand_frznst)}


def score_dict_from_score_vector(
    profile: RankProfile,
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
        profile (RankProfile): Profile to score.
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

    if not isinstance(profile, RankProfile):
        raise TypeError("Profile must only contain ranked ballots.")
    assert profile.max_ranking_length is not None
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

            local_score_vector: Sequence[float | int] = score_vector[
                current_ind : current_ind + position_size
            ]

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
    profile: RankProfile,
) -> dict[str, float]:
    """
    Computes first place votes for all candidates_cast in a ``RankProfile``.
    Intended to be much faster than first_place_votes, but does not handle ties in ballots.

    Args:
        profile (RankProfile): The profile to compute first place votes for.

    Returns:
        dict[str, float]:
            Dictionary mapping candidates to number of first place votes.
    """
    # equiv to score vector of (1,0,0,...)
    assert profile.max_ranking_length is not None
    return _score_dict_from_rankings_df_no_ties(
        profile, [1] + [0] * (profile.max_ranking_length - 1)
    )


def first_place_votes(
    profile: RankProfile,
    tie_convention: Literal["high", "average", "low"] = "average",
) -> dict[str, float]:
    """
    Computes first place votes for all candidates_cast in a ``RankProfile``.

    Args:
        profile (RankProfile): The profile to compute first place votes for.
        tie_convention (Literal["high", "average", "low"], optional): How to award points
            for tied first place votes. Defaults to "average", where if n candidates are tied for
            first, each receives 1/n points. "high" would award them each one point, and "low" 0.

    Returns:
        dict[str, float]:
            Dictionary mapping candidates to number of first place votes.
    """
    # equiv to score vector of (1,0,0,...)
    if not isinstance(profile, RankProfile):
        raise TypeError("Profile must be of type RankProfile.")
    assert profile.max_ranking_length is not None
    return score_dict_from_score_vector(
        profile, [1] + [0] * (profile.max_ranking_length - 1), tie_convention
    )


def mentions(
    profile: RankProfile,
) -> dict[str, float]:
    """
    Calculates total mentions for all candidates in a ``RankProfile``.

    Args:
        profile (RankProfile): RankProfile of ballots.

    Returns:
        dict[str, float]:
            Dictionary mapping candidates to mention totals (values).
    """
    mentions = {c: 0.0 for c in profile.candidates}
    if not isinstance(profile, RankProfile):
        raise TypeError("Profile must be of type RankProfile.")
    for ballot in profile.ballots:
        if ballot.ranking is None:
            raise TypeError("Ballots must have rankings.")
        else:
            for s in ballot.ranking:
                for cand in s:
                    mentions[cand] += ballot.weight
    return mentions


def borda_scores(
    profile: RankProfile,
    borda_max: Optional[int] = None,
    tie_convention: Literal["high", "average", "low"] = "low",
) -> dict[str, float]:
    r"""
    Calculates Borda scores for candidates_cast in a ``RankProfile``. The Borda vector is
    :math:`(n,n-1,\dots,1)` where :math:`n` is the ``borda_max`.

    Args:
        profile (RankProfile): ``RankProfile`` of ballots.
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
    if not isinstance(profile, RankProfile):
        raise TypeError("Profile must be of type RankProfile.")
    if borda_max is None:
        assert profile.max_ranking_length is not None
        borda_max = profile.max_ranking_length

    score_vector = list(range(borda_max, 0, -1))

    return score_dict_from_score_vector(profile, score_vector, tie_convention)


def tiebreak_set(
    r_set: frozenset[str],
    profile: Optional[RankProfile] = None,
    tiebreak: str = "random",
    scoring_tie_convention: Literal["high", "average", "low"] = "low",
    backup_tiebreak_convention: Optional[str] = None,
) -> tuple[frozenset[str], ...]:
    """
    Break a single set of candidates into multiple sets each with a single candidate according
    to a tiebreak rule. Rule 1: random. Rule 2: first-place votes; break the tie based on
    first-place votes in the profile. Rule 3: borda; break the tie based on Borda points in the
    profile. Rule 4: lex/lexicographic/alph/alphabetical; break the tie alphabetically.

    Args:
        r_set (frozenset[str]): Set of candidates on which to break tie.
        profile (RankProfile, optional): Profile used to break ties in first-place votes or
            Borda setting. Defaults to None, which implies a random tiebreak.
        tiebreak (str): Tiebreak method to use. Options are "random", "first_place", and
            "borda". Defaults to "random".
        scoring_tie_convention (Literal["high", "average", "low"]): How to award points
            for tied rankings. Defaults to "low", where any candidates tied receive the lowest
            possible points for their position, eg three people tied for 3rd would each receive the
            points for 5th. "high" awards the highest possible points, so in the previous example,
            they would each receive the points for 3rd. "average" averages the points, so they would
            each receive the points for 4th place.
        backup_tiebreak_convention (str, optional): If the initial tiebreak does not resolve all
            ties,
            this convention is used to break any remaining ties. Options are "random" and
            "lex/lexicographic/alph/alphabetical". Defaults to None which sets the backup to
            "lex" if the initial tiebreak is alphabetical, and "random" otherwise.

    Returns:
        tuple[frozenset[str],...]: tiebroken ranking
    """
    if tiebreak in ["alphabetical", "lexicographic", "alph", "lex"]:
        sorted_cands = sorted([c for c in r_set])
        new_ranking = tuple(map(lambda c: frozenset({c}), sorted_cands))

    elif tiebreak == "random":
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
        if backup_tiebreak_convention is None:
            if tiebreak in [
                "alphabetical",
                "lexicographic",
                "alph",
                "lex",
            ]:
                backup_tiebreak_convention = "lex"
            else:
                backup_tiebreak_convention = "random"

        if backup_tiebreak_convention in [
            "alphabetical",
            "lexicographic",
            "alph",
            "lex",
        ]:
            print("Initial tiebreak was unsuccessful, performing alphabetic tiebreak")
            new_ranking, _ = tiebroken_ranking(
                new_ranking, profile=profile, tiebreak="lex"
            )
        elif backup_tiebreak_convention == "random":
            print("Initial tiebreak was unsuccessful, performing random tiebreak")
            new_ranking, _ = tiebroken_ranking(
                new_ranking, profile=profile, tiebreak="random"
            )
        else:
            raise ValueError("Invalid backup tiebreak code was provided")

    return new_ranking


def tiebroken_ranking(
    ranking: tuple[frozenset[str], ...],
    profile: Optional[RankProfile] = None,
    tiebreak: str = "random",
) -> tuple[
    tuple[frozenset[str], ...], dict[frozenset[str], tuple[frozenset[str], ...]]
]:
    """
    Breaks ties in a list-of-sets ranking according to a given scheme.

    Args:
        ranking (list[set[str]]): A list-of-set ranking of candidates.
        profile (RankProfile, optional): Profile used to break ties in first-place votes or
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
    ranking: Sequence[Union[frozenset[str], set[str]]],
    m: int,
    profile: Optional[RankProfile] = None,
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
        profile (RankProfile, optional): Profile used to break ties in first-place votes or
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
    if m > len([c for s in ranking for c in s]):
        raise ValueError("m must be no more than the number of candidates.")

    ranking_fs: tuple[frozenset[str], ...] = tuple(
        s if isinstance(s, frozenset) else frozenset(s) for s in ranking
    )

    num_elected = 0
    elected: list[frozenset[str]] = []
    i = 0
    tiebreak_ranking: Optional[tuple[frozenset[str], tuple[frozenset[str], ...]]] = None

    while num_elected < m:
        elected.append(ranking_fs[i])
        num_elected += len(ranking_fs[i])
        if num_elected > m:
            if tiebreak is None:
                raise ValueError(
                    "Cannot elect correct number of candidates without breaking ties."
                )
            # back out the overfill
            elected.pop()
            num_elected -= len(ranking_fs[i])

            tiebroken = tiebreak_set(frozenset(ranking_fs[i]), profile, tiebreak)
            elected += tiebroken[: (m - num_elected)]

            remaining: list[frozenset[str]] = list(tiebroken[(m - num_elected) :])
            if i < len(ranking_fs):
                remaining += list(ranking_fs[(i + 1) :])

            return (
                tuple(elected),
                tuple(remaining),
                (ranking_fs[i], tiebroken),
            )
        i += 1

    return (tuple(elected), ranking_fs[i:], tiebreak_ranking)


def expand_tied_ballot(ballot: RankBallot) -> list[RankBallot]:
    """
    Fix tie(s) in a ballot by returning all possible permutations of the tie(s), and divide the
    weight of the original ballot equally among the new ballots.

    Args:
        ballot (RankBallot): Ballot to expand tie sets on.

    Returns:
        list[v]: All possible permutations of the tie(s).

    """
    if not isinstance(ballot, RankBallot):
        raise TypeError("Ballot must be of type RankBallot.")
    if ballot.ranking is None:
        raise TypeError("Ballot must have ranking.")
    if all(len(s) == 1 for s in ballot.ranking):
        return [ballot]

    else:
        for i, s in enumerate(ballot.ranking):
            if len(s) > 1:
                new_ballots = [
                    RankBallot(
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


def resolve_profile_ties(profile: RankProfile) -> RankProfile:
    """
    Takes in a PeferenceProfile with potential ties in ballots. Replaces
    ballots with ties with fractionally weighted ballots corresponding to
    all permutations of the tied ranking.

    Args:
        profile (RankProfile): Input profile with potentially tied rankings.

    Returns:
        RankProfile: A RankProfile with resolved ties.
    """

    new_ballots = tuple(
        [b for ballot in profile.ballots for b in expand_tied_ballot(ballot)]
    )
    return RankProfile(ballots=new_ballots)


def score_profile_from_ballot_scores(
    profile: ScoreProfile,
) -> dict[str, float]:
    """
    Score the candidates based on the ``scores`` parameter of the ballots.
    All ballots must have a ``scores`` parameter; note that a ``scores`` dictionary
    with no non-zero scores will raise the same error.

    Args:
        profile (ScoreProfile): Profile to score.

    Returns:
        dict[str, float]:
            Dictionary mapping candidates to scores.
    """
    scores = {c: 0.0 for c in profile.candidates}
    if not isinstance(profile, ScoreProfile):
        raise TypeError("Profile must be of type ScoreProfile.")
    for ballot in profile.ballots:
        if ballot.scores is None:
            raise TypeError(f"Ballot {ballot} has no scores.")
        else:
            for c, score in ballot.scores.items():
                scores[c] += score * ballot.weight

    return scores


def ballot_lengths(profile: RankProfile) -> dict[int, float]:
    """
    Compute the frequency of ballot lengths in the profile.
    Includes all lengths from 1 to ``max_ranking_length`` as keys.
    Ballots must have rankings.

    Args:
        profile (RankProfile): Profile to compute ballot lengths.

    Returns:
        dict[int, float]: Dictionary of ballot length frequency.

    Raises:
        TypeError: All ballots must have rankings.
    """
    if not isinstance(profile, RankProfile):
        raise TypeError("Profile must be of type RankProfile.")
    assert profile.max_ranking_length is not None

    ballot_lengths = {i: 0.0 for i in range(1, profile.max_ranking_length + 1)}

    for ballot in profile.ballots:
        if ballot.ranking is None:
            raise TypeError("All ballots must have rankings.")

        length = len(ballot.ranking)
        ballot_lengths[length] += ballot.weight

    return ballot_lengths


def index_to_lexicographic_ballot(
    ballot_index: int,
    n_candidates: int,
    max_length: int,
    total_valid_ballots_method,
    always_use_total_valid_ballots_method: bool = True,
) -> list[int]:
    """
    Convert an index to one ballot with candidates taken from the list range(n_candidates), and
    where the ballot has length at most max_length.
    The ordering of the ballots is lexicographic, i.e., the first ballot is the
    lexicographically smallest ballot and continues in that order:

    (0,),
    (0,1),
    (0,1,2),
    ...
    (0,2),
    (0,2,1),
    ...
    (n-1, n-2, ..., n-l)

    where n is the number of candidates and l is the maximum ballot length.

    Args:
        ballot_index (int): The index to convert.
        n_candidates (int): The number of candidates.
        max_length (int): The maximum allowed ballot rank.
        total_valid_ballots_method: ((n_candidates, max_length) ->
            integer) a method which returns the total number of
            allowed ballots.
        always_use_total_valid_ballots_method (bool): a flag
            indicating whether the given total valid ballots method
            should be used when computing b_n. If
            total_valid_ballots_method is using a cache then this
            should be True for maximum runtime efficiency.

    Returns:
        list[int]: A list representing the ballot corresponding to index.
    """
    total_valid_ballots = total_valid_ballots_method(n_candidates, max_length)
    if ballot_index >= total_valid_ballots:
        raise Exception(
            f"Given ballot index {ballot_index} out of range. Max index: {total_valid_ballots}"
        )

    def chunk_size(n_cands: int, ballot_len: int) -> int:
        return total_valid_ballots_method(n_cands, ballot_len) // n_cands

    candidates = list(range(n_candidates))
    out = []
    if always_use_total_valid_ballots_method:
        bn = total_valid_ballots_method(n_candidates + 1, max_length + 1) // (
            n_candidates + 1
        )
    else:
        bn = chunk_size(n_candidates + 1, max_length + 1)

    for i in range(n_candidates, 0, -1):
        if always_use_total_valid_ballots_method:
            bn = total_valid_ballots_method(
                n_candidates=i, max_ballot_length=max_length
            ) // (i)
        else:
            bn = (bn - 1) // i
        # Perform Euclidean division of index by bn
        section = ballot_index // bn
        remaining = ballot_index % bn
        out.append(candidates.pop(section))
        if remaining == 0:
            # Cut off the ballot here
            break
        ballot_index = remaining - 1
    return out


def build_df_from_ballot_samples(
    ballots_freq_dict: dict[tuple[int, ...], int], candidates: Sequence[str]
):
    """
    Helper function which creates a pandas df to instantiate a
    RankProfile
    args:
        ballots_freq_dict: dictionary mapping ballots to
            sampled frequency. The keys should be in candidate id
            form
        candidates : list of candidates in the profile
    returns:
        pandas df
    """
    df_data = []
    n_cands = len(candidates)
    for ballot in ballots_freq_dict.keys():
        ballot_as_frozenset_entries = tuple(
            [frozenset([candidates[i]]) for i in ballot]
        )
        completed_ballot = (
            ballot_as_frozenset_entries
            + tuple(
                [frozenset(["~"]) for _ in range(n_cands - len(ballot))]
            )  # padding short ballots
            + tuple([ballots_freq_dict[ballot], set()])
        )  # weight, voter set
        df_data.append(completed_ballot)
    return pd.DataFrame(
        df_data,
        columns=[f"Ranking_{i}" for i in range(1, n_cands + 1)]
        + ["Weight", "Voter Set"],
    )
