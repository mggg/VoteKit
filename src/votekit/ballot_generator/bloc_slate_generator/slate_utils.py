import numpy as np
from typing import Sequence, Iterator
from votekit.ballot_generator.bloc_slate_generator.model import BlocSlateConfig
from votekit.pref_interval import PreferenceInterval
import pandas as pd

from votekit.pref_profile import RankProfile


def _lexicographic_symbol_tuple_iterator(
    symbol_list: list[str],
) -> Iterator[tuple[str, ...]]:
    """
    Given a set of symbols, generate all possible combinations of symbols.

    Example:
        symbol_list = ["A", "A", "B"]
        returns {("A", "A", "B"), ("A", "B", "A"), ("B", "A", "A")}

    Args:
        symbol_list (list[str]): A list of symbols.

    Returns:
        set[tuple[str]]: A set of tuples of strings.
    """
    n = len(symbol_list)
    if n == 0:
        return

    # sort lexicographically
    current_symbol_perm = sorted(symbol_list)

    # Idea here is to make the smallest possible change to the between steps that increases
    # the lexicographic cost (so we will find the next thing in the lexicographic ordering)
    while True:
        yield tuple(current_symbol_perm)

        # find the longest non-increasing suffix (fails when you have reverse lex order and returns)
        # start from the right and stop when you find a[i] < a[i+1]
        i = n - 2
        while i >= 0 and current_symbol_perm[i] >= current_symbol_perm[i + 1]:
            i -= 1
        if i < 0:
            return

        # find rightmost element greater than a[i]. This may not be a[i+1] because of duplicates
        j = n - 1
        while current_symbol_perm[j] <= current_symbol_perm[i]:
            j -= 1

        # swap pivot with successor
        current_symbol_perm[i], current_symbol_perm[j] = (
            current_symbol_perm[j],
            current_symbol_perm[i],
        )

        # reverse the suffix to get the next smallest lexicographic ordering (so the suffix should
        # b in non-decreasing order now)
        current_symbol_perm[i + 1 :] = reversed(current_symbol_perm[i + 1 :])


def _algorithm_a_sample_indices(weights: np.array, n_samples: int) -> np.ndarray:
    """
    Sample without replacement from a distribution given by the weights.
    All weights must be non-zero, and the sample will be a sorted list of indices
    corresponding to the weights.

    Algorithm A from https://doi.org/10.1016/j.ipl.2005.11.003

    Args:
        weights (np.array): The weights of the distribution.
        n_samples (int): Number of samples to generate.

    Returns:
        np.ndarray: The sampled indices, n_samples x n_cands.
    """
    n_cands = len(weights)
    uniform = np.random.uniform(0, 1, size=(n_samples, n_cands))
    uniform = uniform ** (1 / weights)
    # want the largest values to be first
    indices = np.flip(np.argsort(uniform, axis=1), axis=1)
    return indices


def _construct_slate_to_candidate_ordering_arrays(
    config: BlocSlateConfig,
    bloc: str,
    n_samples: int,
) -> dict[str, np.ndarray]:
    """
    Create candidate orderings within each slate based on preference intervals.
    Orders non-zero support candidates by sampling without replacement according to the
    preference intervals. Randomly permutes candidates with 0 support at the end of the
    ordering.

    Args:
        config (BlocSlateConfig): Configuration object containing all necessary parameters for
            working with a bloc-slate ballot generator.
        bloc (str): The name of the bloc.
        n_samples (int): Number of candidate orderings to generate.

    Returns:
        dict[str, np.ndarray]: A dictionary mapping slate names to an n_samples x n_cands matrix
            of candidate orderings.
    """
    results: dict[str, np.ndarray] = {}
    pref_intervals_by_slate_dict = config.get_preference_intervals_for_bloc(bloc)

    for slate in config.slates:
        preference_interval = pref_intervals_by_slate_dict[slate].interval
        non_zero_support_cands = pref_intervals_by_slate_dict[slate].non_zero_cands
        zero_support_cands = pref_intervals_by_slate_dict[slate].zero_cands
        cand_ordering = np.empty((n_samples, len(non_zero_support_cands) + len(zero_support_cands)), dtype=object)
        
        if len(non_zero_support_cands) != 0:
            cands_list = list(non_zero_support_cands)
            distribution = np.array([preference_interval[c] for c in list(non_zero_support_cands)])

            indices = _algorithm_a_sample_indices(distribution, n_samples)

            # TODO This is not the right size on the left 
            cand_ordering[:][:len(non_zero_support_cands)] = np.vectorize(lambda i: cands_list[i])(indices)
        
        cand_ordering[len(non_zero_support_cands):] = np.random.permutation(list(zero_support_cands))

        results[slate] = cand_ordering
        
    return results



def _convert_ballot_type_to_ranking(
    ballot_type: Sequence[str],
    cand_ordering_by_slate: dict[str, list[str]],
    #final_max_ranking_length: int = 0,
) -> list[frozenset[str]]:
    """
    Given a ballot type and a candidate ordering by slate, convert the ballot type to a ranking.

    Example:

    Given a ballot type "AABBA" and candidate ordering by slate
    {"A": ["a3", "a1", "a2"], "B": ["b2", "b1"]}, the function will return
    [frozenset({"a3"}), frozenset({"a1"}), frozenset({"b2"}), frozenset({"b1"}), frozenset({"a2"})].

    If there are not enough candidates to fill all positions, the ballot will be truncated.

    Args:
        ballot_type (Sequence[str]): A sequence of slate names representing the ballot type.
        cand_ordering_by_slate (dict[str, list[str]]): A dictionary mapping slate names to a list
            of candidate names ordered according to the sampled preference intervals.
        # final_max_ranking_length (int): The maximum length of the ranking. Defaults to 0, which
        #     sets the final max ranking length to the length of the ballot type.

    Returns:
        list[frozenset[str]]: A list of frozensets, where each frozenset contains a single
            candidate name, representing the ranking derived from the ballot type and candidate
            ordering
    """
    # if final_max_ranking_length == 0:
    #     final_max_ranking_length = len(ballot_type)

    positions = {s: 0 for s in cand_ordering_by_slate}
    ranking: list[frozenset[str]] = [frozenset("~")] * final_max_ranking_length

    fset_cache: dict[str, frozenset[str]] = {}

    # for i, slate in enumerate(ballot_type[:final_max_ranking_length]):
    for i, slate in enumerate(ballot_type):

        pos = positions[slate]

        if pos >= len(cand_ordering_by_slate[slate]):
            continue

        cand = cand_ordering_by_slate[slate][pos]
        positions[slate] = pos + 1

        fset = fset_cache.get(cand)
        if fset is None:
            fset = frozenset((cand,))
            fset_cache[cand] = fset
        ranking[i] = fset

    return ranking


def _convert_slate_ballots_to_profile(
    config, bloc, slate_ballots
):

    n_candidates = len(config.candidates)
    n_ballots = len(slate_ballots)

    # now full orderings of all candidates in each slate, zero or non-zero support
    cand_orderings_by_slate = _construct_slate_to_candidate_ordering_arrays(
        config, bloc, n_ballots
    )

    ballot_pool = np.full((n_ballots, n_candidates), frozenset("~"))
    for i, slate_ballot in enumerate(slate_ballots):
        ranking = _convert_ballot_type_to_ranking(
            ballot_type=slate_ballot,
            cand_ordering_by_slate={
                s: cand_ordering[i]
                for s, cand_ordering in cand_orderings_by_slate.items()
            },
            # TODO is this going to play nicely with zero support candidates?
            #final_max_ranking_length=n_candidates,
        )
        ballot_pool[i] = np.array(ranking)

    df = pd.DataFrame(ballot_pool)
    df.index.name = "Ballot Index"
    df.columns = [f"Ranking_{i + 1}" for i in range(n_candidates)]
    df["Weight"] = 1
    df["Voter Set"] = [frozenset()] * len(df)
    pp = RankProfile(
        candidates=config.candidates,
        df=df,
        max_ranking_length=n_candidates,
    )

    return pp
