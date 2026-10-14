from typing import Iterator, Optional

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from votekit.ballot_generator.bloc_slate_generator.config import BlocSlateConfig


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


def _fast_sample_without_replacement(
    weights: NDArray,
    n_samples: int,
    *,
    numpy_rng: Optional[Generator] = None,
) -> np.ndarray:
    """
    Sample without replacement from a distribution given by the weights.
    All weights must be non-zero, and the sample will be a sorted list of indices
    corresponding to the weights.

    Algorithm A from https://doi.org/10.1016/j.ipl.2005.11.003

    Args:
        weights (np.NDArray): The weights of the distribution.
        n_samples (int): Number of samples to generate.
        numpy_rng (Generator, optional): NumPy random number generator. Pass a seeded instance
            for reproducible results. Defaults to None for non-deterministic results.

    Returns:
        np.ndarray: The sampled indices, n_samples x n_cands.
    """
    n_cands = len(weights)
    numpy_rng = np.random.default_rng() if numpy_rng is None else numpy_rng
    uniform = numpy_rng.uniform(0, 1, size=(n_samples, n_cands))
    uniform = uniform ** (1 / weights)
    # want the largest values to be first
    indices = np.flip(np.argsort(uniform, axis=1), axis=1)
    return indices


def _construct_slate_to_candidate_ordering_arrays(
    config: BlocSlateConfig,
    bloc: str,
    n_samples: int,
    *,
    numpy_rng: Optional[Generator] = None,
) -> dict[str, np.ndarray]:
    """
    Create orderings of candidate IDs within each slate based on preference intervals.

    Orders non-zero support candidates by sampling without replacement according to the
    preference intervals. Randomly permutes candidates with 0 support at the end of the
    ordering.

    Args:
        config (BlocSlateConfig): Configuration object containing all necessary parameters for
            working with a bloc-slate ballot generator.
        bloc (str): The name of the bloc.
        n_samples (int): Number of candidate orderings to generate.
        numpy_rng (Generator, optional): NumPy random number generator. Pass a seeded instance
            for reproducible results. Defaults to None for non-deterministic results.

    Returns:
        dict[str, np.ndarray]: A dictionary mapping slate names to an n_samples x n_cands matrix
            of candidate orderings, given as indices into ``config.candidates``.
    """
    results: dict[str, np.ndarray] = {}
    pref_intervals_by_slate_dict = config.get_preference_intervals_for_bloc(bloc)
    candidate_to_index = {cand: i for i, cand in enumerate(config.candidates)}

    for slate in config.slates:
        preference_interval = pref_intervals_by_slate_dict[slate].interval
        slate_candidates = config.slate_to_candidates[slate]

        slate_cand_ids = np.array(
            [candidate_to_index[cand] for cand in slate_candidates], dtype=int
        )
        slate_cand_dist = np.array([preference_interval[cand] for cand in slate_candidates])
        indices = _fast_sample_without_replacement(
            slate_cand_dist, n_samples, numpy_rng=numpy_rng
        )  # shape

        results[slate] = slate_cand_ids[indices]

    return results


def _append_zero_slate_symbols(
    ballots: np.ndarray,
    zero_slate_set: set[str],
    n_ballots: int,
    config: BlocSlateConfig,
    *,
    numpy_rng: Optional[Generator] = None,
) -> np.ndarray:
    """
    Fills in the trailing positions of each slate ballot type with the zero cohesion slate indices.

    The zero cohesion slate indices are randomly permuted for each ballot. ``ballots`` is modified
    in place.

    Args:
        ballots (np.ndarray): Array of slate ballot types where slates are represented as indices
            into ``config.slates`` and end positions are unfilled for zero cohesion candidates.
        zero_slate_set (set[str]): Set of slates with zero cohesion for the given bloc.
        n_ballots (int): The number of ballots to generate.
        config (BlocSlateConfig): Configuration object containing all necessary parameters for
            working with a bloc-slate ballot generator.
        numpy_rng (Generator, optional): NumPy random number generator. Pass a seeded instance
            for reproducible results. Defaults to None for non-deterministic results.

    Returns:
        np.ndarray: Array of slate ballot types with zero cohesion slate symbols filled in.
    """
    zero_slate_symbols = np.array(
        [
            config.slates.index(slate)
            for slate in sorted(zero_slate_set)
            for _ in config.slate_to_candidates[slate]
        ],
        dtype=int,
    )
    numpy_rng = np.random.default_rng() if numpy_rng is None else numpy_rng
    noise = numpy_rng.random(size=(n_ballots, len(zero_slate_symbols)))
    permutation_indices = np.argsort(noise, axis=1)
    ballots[:, ballots.shape[1] - len(zero_slate_symbols) :] = zero_slate_symbols[
        permutation_indices
    ]
    return ballots


def _convert_slate_ballots_to_candidate_ids(
    config: BlocSlateConfig,
    bloc: str,
    slate_ballots: np.ndarray,
    *,
    numpy_rng: Optional[Generator] = None,
) -> np.ndarray:
    """
    Fill out slate ballot types with candidate IDs by sampling from the bloc's preference intervals.

    Args:
        config (BlocSlateConfig): Configuratioxn object containing all necessary parameters for
            working with a bloc-slate ballot generator.
        bloc (str): The name of the bloc.
        slate_ballots (np.ndarray): Array of slate ballot types where the slates are represented as
            indices into ``config.slates``.
        numpy_rng (Generator, optional): NumPy random number generator. Pass a seeded instance
            for reproducible results. Defaults to None for non-deterministic results.

    Returns:
        np.ndarray: Array of n_ballots x n_candidates with candidate IDs as indices into
            ``config.candidates``; padded with -1.
    """
    n_candidates = len(config.candidates)
    n_ballots = len(slate_ballots)

    # full orderings of all candidates as their indices into ``config.candidates`` in each slate
    cand_id_orderings_by_slate = _construct_slate_to_candidate_ordering_arrays(
        config,
        bloc,
        n_ballots,
        numpy_rng=numpy_rng,
    )

    candidate_ids = np.full((n_ballots, n_candidates), -1, dtype=int)
    for slate_index, slate in enumerate(config.slates):
        slate_ballot_mask = slate_ballots == slate_index
        ballots, candidates = np.nonzero(slate_ballot_mask)
        slate_cand_positions = (np.cumsum(slate_ballot_mask, axis=1) - 1)[ballots, candidates]
        cand_slate_orderings = cand_id_orderings_by_slate[slate]

        if (
            len(slate_cand_positions)
            and slate_cand_positions.max() >= cand_slate_orderings.shape[1]
        ):
            raise IndexError(
                f"There are insufficiently many candidates ({cand_slate_orderings.shape[1]})"
                f" in slate {slate} to complete ballot that lists the slate"
                f" {slate_cand_positions.max() + 1}."
            )
        candidate_ids[ballots, candidates] = cand_slate_orderings[ballots, slate_cand_positions]

    return candidate_ids
