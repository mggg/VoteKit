import numpy as np

from votekit.ballot_generator.bloc_slate_generator.model import BlocSlateConfig
from votekit.pref_interval import PreferenceInterval


def _make_cand_ordering_by_slate(
    config: BlocSlateConfig, pref_intervals_by_slate_dict: dict[str, PreferenceInterval]
) -> dict[str, list[str]]:
    """
    Create a candidate ordering within each slate based on the preference intervals.

    The candidate oridering is determined by sampling without replacement according to
    the preference intervals using the Plackett-Luce model (i.e. weighted coin flips).

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
        bloc_cand_pref_interval = pref_intervals_by_slate_dict[slate].interval
        cands = pref_intervals_by_slate_dict[slate].non_zero_cands

        if len(cands) == 0:
            continue

        distribution = [bloc_cand_pref_interval[c] for c in cands]

        # sample by Plackett-Luce within the bloc
        cand_ordering = np.random.choice(
            a=list(cands), size=len(cands), p=distribution, replace=False
        )

        cand_ordering_by_slate[slate] = list(cand_ordering)
    return cand_ordering_by_slate


# def _make_cand_ordering_by_slate(config, pref_intervals_by_slate_dict):
#     rng = np.random.default_rng()  # faster & better API than np.random.*
#     cand_ordering_by_bloc = {}
#     # Cache attribute lookups
#     slates = config.slates
#
#     for slate in slates:
#         entry = pref_intervals_by_slate_dict[slate]
#         cands = entry.non_zero_cands  # iterable of candidate keys
#         if not cands:  # empty or len == 0
#             continue
#
#         # Build weight vector once (float64) and normalize
#         # Using a list comp is usually fastest for Python->NumPy bridge
#         weights = np.asarray([entry.interval[c] for c in cands], dtype=np.float64)
#         wsum = weights.sum()
#         if wsum <= 0.0:
#             # Safety: all zero or negative due to upstream rounding; skip
#             continue
#         weights /= wsum
#
#         # Gumbel–top-k (samples a PL/“without replacement weighted by p” ordering)
#         # Equivalent to drawing i.i.d. Gumbels, adding log-weights, and sorting desc.
#         scores = np.log(weights) + rng.gumbel(size=weights.size)
#         order_idx = np.argsort(scores)[::-1]
#
#         # Map indices back to candidate labels; avoid list(cands) conversion if already a list/tuple
#         # If cands is a set, convert once to a tuple so indexing is stable
#         if not isinstance(cands, (list, tuple, np.ndarray)):
#             cands = tuple(cands)
#         cand_ordering_by_bloc[slate] = [cands[i] for i in order_idx]
#
#     return cand_ordering_by_bloc


def _convert_ballot_type_to_ranking(
    ballot_type, cand_ordering_by_slate
) -> list[frozenset[str]]:
    positions = {s: 0 for s in cand_ordering_by_slate}
    ranking = [frozenset()] * len(ballot_type)

    fset_cache = {}

    # Ensure sequences are indexable tuples/lists (avoid repeated conversions)
    for s, seq in cand_ordering_by_slate.items():
        if not isinstance(seq, (list, tuple)):
            cand_ordering_by_slate[s] = tuple(seq)

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
