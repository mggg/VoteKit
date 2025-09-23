import numpy as np
from typing import Sequence

from votekit.ballot_generator.bloc_slate_generator.model import BlocSlateConfig
from votekit.pref_interval import PreferenceInterval


def _make_cand_ordering_by_slate(
    config: BlocSlateConfig, pref_intervals_by_slate_dict: dict[str, PreferenceInterval]
) -> dict[str, list[str]]:
    """
    Create a candidate ordering within each slate based on the preference intervals.

    The candidate oridering is determined by sampling without replacement according to
    the preference intervals using the Plackett-Luce model (i.e. throwing a dart and removing
    that part of the interval).

    Args:
        config (BlocSlateConfig): Configuration object containing all necessary parameters for
            working with a bloc-slate ballot generator.
        pref_intervals_by_slate_dict (dict[str, PreferenceInterval]): A dictionary mapping
            slate names to their corresponding PreferenceInterval objects.

    Returns:
        dict[str, list[str]]: A dictionary mapping slate names to a list of candidate names
            ordered according to the sampled preference intervals.
    """
    cand_ordering_by_slate = {}

    for slate in config.slates:
        preference_interval = pref_intervals_by_slate_dict[slate].interval
        cands = pref_intervals_by_slate_dict[slate].non_zero_cands

        if len(cands) == 0:
            continue

        distribution = [preference_interval[c] for c in cands]

        # sample by Plackett-Luce within the bloc
        cand_ordering = np.random.choice(
            a=list(cands), size=len(cands), p=distribution, replace=False
        )

        cand_ordering_by_slate[slate] = list(cand_ordering)
    return cand_ordering_by_slate


def _convert_ballot_type_to_ranking(
    ballot_type: Sequence[str],
    cand_ordering_by_slate: dict[str, list[str]],
) -> list[frozenset[str]]:
    """
    Given a ballot type and a candidate ordering by slate, convert the ballot type to a ranking.

    Example:

    Given a ballot type "AABBA" and candidate ordering by slate
    {"A": ["a3", "a1", "a2"], "B": ["b2", "b1"]}, the function will return
    [frozenset({"a3"}), frozenset({"a1"}), frozenset({"b2"}), frozenset({"b1"}), frozenset({"a2"})].

    Args:
        ballot_type (Sequence[str]): A sequence of slate names representing the ballot type.
        cand_ordering_by_slate (dict[str, list[str]]): A dictionary mapping slate names to a list
        of candidate names ordered according to the sampled preference intervals.

    Returns:
        list[frozenset[str]]: A list of frozensets, where each frozenset contains a single
            candidate name, representing the ranking derived from the ballot type and candidate
            ordering
    """
    positions = {s: 0 for s in cand_ordering_by_slate}
    ranking: list[frozenset[str]] = [frozenset("~")] * len(ballot_type)

    fset_cache: dict[str, frozenset[str]] = {}

    for i, slate in enumerate(ballot_type):
        pos = positions[slate]
        cand = cand_ordering_by_slate[slate][pos]
        positions[slate] = pos + 1

        fset = fset_cache.get(cand)
        if fset is None:
            fset = frozenset((cand,))
            fset_cache[cand] = fset
        ranking[i] = fset

    return ranking
