from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
import pandas as pd
from numpy.random import Generator
from numpy.typing import NDArray

from votekit._df_schema import (
    BALLOT_INDEX_NAME,
    RANKING_COLUMN_PREFIX,
    VOTER_SET_COLUMN,
    WEIGHT_COLUMN,
)
from votekit.ballot_generator.bloc_slate_generator._sampling import (
    _fast_sample_without_replacement as _fast_sample_without_replacement,
)
from votekit.ballot_generator.bloc_slate_generator._sampling import (
    _fill_pl_rankings,
)
from votekit.ballot_generator.bloc_slate_generator.config import BlocSlateConfig
from votekit.pref_profile import RankProfile
from votekit.types import Candidate


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

    # Idea here is to make the smallest possible change to the between steps that increases the
    # lexicographic cost (so we will find the next thing in the lexicographic ordering)
    while True:
        yield tuple(current_symbol_perm)

        # find the longest non-increasing suffix (fails when you have reverse lex order and
        # returns) start from the right and stop when you find a[i] < a[i+1]
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


def _prepare_slate_candidate_arrays(
    config: BlocSlateConfig, bloc: str, candidate_to_id: dict[Candidate, int]
) -> tuple[tuple[NDArray[np.intp], NDArray[np.float64]], ...]:
    """
    Capture candidate IDs and supports for each slate in configuration order.

    Candidate IDs use the profile's common universe. Each pair contains independent arrays for one
    slate so batches reuse the same inputs without rereading the bloc configuration.

    Args:
        config (BlocSlateConfig): Validated slate membership and within-slate preference
            intervals.
        bloc (str): Bloc whose candidate supports are captured.
        candidate_to_id (dict[Candidate, int]): Position of every candidate in the output
            profile's declared candidate universe.

    Returns:
        tuple[tuple[NDArray[np.intp], NDArray[np.float64]], ...]: Candidate-ID and support arrays
            for each slate in configuration order. Within each pair, entries follow that slate's
            candidate order. The arrays own their storage and do not modify the configuration.
    """
    preference_intervals = config.get_preference_intervals_for_bloc(bloc)
    candidate_arrays = []
    for slate in config.slates:
        slate_candidates = config.slate_to_candidates[slate]
        candidate_ids = np.array([candidate_to_id[candidate] for candidate in slate_candidates])
        supports = np.array(
            [preference_intervals[slate].interval[candidate] for candidate in slate_candidates],
            dtype=np.float64,
        )
        candidate_arrays.append((candidate_ids, supports))

    return tuple(candidate_arrays)


def _fill_slate_candidate_rankings(
    candidate_arrays: tuple[tuple[NDArray[np.intp], NDArray[np.float64]], ...],
    slate_batches: Iterator[NDArray[np.intp]],
    rankings: NDArray[np.intp],
    rng: np.random.Generator,
) -> None:
    """
    Fill a bloc's rankings from successive batches of slate sequences.

    Each slate's candidate order follows PL on its positive supports, followed by a uniformly
    shuffled zero-support tail. Assigning that order to successive occurrences of the slate
    preserves both the slate sequence and the candidate distribution within it.

    Batches fill consecutive output rows in iterator order. Candidate sampling finishes for one
    batch before the next batch is requested, preserving the interleaving of slate and candidate
    draws when both use the same RNG. Temporary candidate-ordering buffers are released before
    advancing the iterator so they do not overlap the next batch's sequence preparation.

    Args:
        candidate_arrays (tuple[tuple[NDArray[np.intp], NDArray[np.float64]], ...]): Aligned
            candidate-ID and support arrays in slate-ID order, containing at least one slate.
        slate_batches (Iterator[NDArray[np.intp]]): Batches of slate IDs with each slate occurring
            once per candidate it contains on every row. Together the batches must contain exactly
            as many rows as the output and have the same column count.
        rankings (NDArray[np.intp]): Writable candidate-ID output for the entire bloc. May be a
            view into a larger profile's array. Every row is filled in place.
        rng (np.random.Generator): Stream advanced in slate order within each batch, interleaved
            with any draws made by the sequence iterator.
    """
    next_row = 0
    for slate_sequences in slate_batches:
        stop = next_row + len(slate_sequences)
        _fill_slate_candidate_batch(candidate_arrays, slate_sequences, rankings[next_row:stop], rng)
        next_row = stop


def _fill_slate_candidate_batch(
    candidate_arrays: tuple[tuple[NDArray[np.intp], NDArray[np.float64]], ...],
    slate_sequences: NDArray[np.intp],
    rankings: NDArray[np.intp],
    rng: np.random.Generator,
) -> None:
    """
    Fill one batch of slate positions with independently sampled candidate rankings.

    A reusable ordering buffer holds the candidate draws for one slate at a time. Its columns are
    scattered into that slate's successive positions on each ballot. All temporary arrays belong
    to this call and are released on return, before the caller prepares another batch.

    Args:
        candidate_arrays (tuple[tuple[NDArray[np.intp], NDArray[np.float64]], ...]): Candidate-ID
            and support arrays in slate-ID order, containing at least one slate.
        slate_sequences (NDArray[np.intp]): Slate IDs, with one occurrence per candidate from each
            slate on every row. Read without modification.
        rankings (NDArray[np.intp]): Writable candidate-ID output with the same shape as the
            sequences. Filled in place without retaining a reference to its storage.
        rng (np.random.Generator): Stream advanced in slate order within this batch.
    """
    max_slate_size = max(len(candidate_ids) for candidate_ids, _ in candidate_arrays)
    voter_rows = np.arange(len(rankings))[:, None]
    candidate_orderings = np.empty((len(rankings), max_slate_size), dtype=int)

    for slate_id, (candidate_ids, supports) in enumerate(candidate_arrays):
        n_candidates = len(candidate_ids)
        slate_positions = np.nonzero(slate_sequences == slate_id)[1]
        slate_positions_by_voter = slate_positions.reshape(len(rankings), n_candidates)

        slate_orderings = candidate_orderings[:, :n_candidates]
        _fill_pl_rankings(candidate_ids, supports, slate_orderings, rng)
        rankings[voter_rows, slate_positions_by_voter] = slate_orderings


def _construct_slate_to_candidate_ordering_arrays(
    config: BlocSlateConfig,
    bloc: str,
    n_samples: int,
    *,
    numpy_rng: Generator | None = None,
) -> dict[str, np.ndarray]:
    """
    Create candidate orderings within each slate based on preference intervals. Orders non-zero
    support candidates by sampling without replacement according to the preference intervals.
    Randomly permutes candidates with 0 support at the end of the ordering.

    Args:
        config (BlocSlateConfig): Configuration object containing all necessary parameters for
            working with a bloc-slate ballot generator.
        bloc (str): The name of the bloc.
        n_samples (int): Number of candidate orderings to generate.
        numpy_rng (Generator, optional): NumPy random number generator. Pass a seeded instance for
            reproducible results. Defaults to None for non-deterministic results.

    Returns:
        dict[str, np.ndarray]: A dictionary mapping slate names to an n_samples x n_cands matrix
            of candidate orderings.
    """
    results: dict[str, np.ndarray] = {}
    rng = np.random.default_rng() if numpy_rng is None else numpy_rng
    pref_intervals_by_slate_dict = config.get_preference_intervals_for_bloc(bloc)

    for slate in config.slates:
        preference_interval = pref_intervals_by_slate_dict[slate].interval
        candidates = config.slate_to_candidates[slate]
        cand_ordering = np.empty(
            (n_samples, len(candidates)),
            dtype=object,
        )

        cands_list = np.array(list(candidates), dtype=object)  # candidates can be of mixed types
        distribution = np.array([preference_interval[c] for c in candidates], dtype=np.float64)
        indices = np.empty((n_samples, len(candidates)), dtype=int)
        _fill_pl_rankings(np.arange(len(candidates)), distribution, indices, rng)
        cand_ordering[:] = cands_list[indices]

        results[slate] = cand_ordering

    return results


def _convert_slate_ballot_type_to_ranking(
    ballot_type: Sequence[str],
    cand_ordering_by_slate: dict[str, Sequence[Candidate]],
    final_max_ranking_length,
) -> list[frozenset[Candidate]]:
    """
    Given a ballot type and a candidate ordering by slate, convert the ballot type to a ranking.

    Example:

    Given a ballot type "AABBA" and candidate ordering by slate
    {"A": ["a3", "a1", "a2"], "B": ["b2", "b1"]}, the function will return
    [frozenset({"a3"}), frozenset({"a1"}), frozenset({"b2"}), frozenset({"b1"}), frozenset({"a2"})].

    If there are not enough candidates to fill all positions, the ballot will be truncated.

    Args:
        ballot_type (Sequence[str]): A sequence of slate names representing the ballot type.
        cand_ordering_by_slate (dict[str, list[Candidate]]): A dictionary mapping slate names to a
            list of candidate names ordered according to the sampled preference intervals.
            Candidates can be strings, integers, or mix of both.
        final_max_ranking_length (int): The maximum length of the ranking.

    Returns:
        list[frozenset[Candidate]]: A list of frozensets, where each frozenset contains a single
            candidate name, representing the ranking derived from the ballot type and candidate
            ordering. Candidates can be strings, integers, or mix of both.
    """

    positions = {s: 0 for s in cand_ordering_by_slate}
    ranking: list[frozenset[Candidate]] = [frozenset("~")] * final_max_ranking_length

    fset_cache: dict[Candidate, frozenset[Candidate]] = {}
    rank_position = 0
    for slate in ballot_type[:final_max_ranking_length]:
        pos = positions[slate]

        try:
            cand = cand_ordering_by_slate[slate][pos]
        except IndexError:
            raise IndexError(
                f"There are insufficiently many candidates ({len(cand_ordering_by_slate[slate])}) "
                "in slate "
                f"{slate} to complete ballot of type {ballot_type[:final_max_ranking_length]}"
            )
        except Exception as e:
            raise e
        positions[slate] = pos + 1

        fset = fset_cache.get(cand)
        if fset is None:
            fset = frozenset((cand,))
            fset_cache[cand] = fset
        ranking[rank_position] = fset
        rank_position += 1

    return ranking


def _convert_slate_ballots_to_profile(
    config: BlocSlateConfig,
    bloc: str,
    slate_ballots: Sequence[Sequence[str]],
    *,
    numpy_rng: Generator | None = None,
) -> RankProfile:
    """
    Convert slate ballot types to a preference profile, filling out candidate orderings for each
    slate.

    Args:
        config (BlocSlateConfig): Configuratioxn object containing all necessary parameters for
            working with a bloc-slate ballot generator.
        bloc (str): The name of the bloc.
        slate_ballots (Sequence[Sequence[str]]): List of slate ballot types.
        rng (random.Random, optional): Standard library random number generator. Pass a seeded
            instance for reproducible results. Defaults to None for non-deterministic results.

    Returns:
        RankProfile: A preference profile.
    """
    n_candidates = len(config.candidates)
    n_ballots = len(slate_ballots)

    # full orderings of all candidates in each slate
    cand_orderings_by_slate = _construct_slate_to_candidate_ordering_arrays(
        config,
        bloc,
        n_ballots,
        numpy_rng=numpy_rng,
    )

    ballot_pool = np.full((n_ballots, n_candidates), frozenset("~"))
    for i, slate_ballot in enumerate(slate_ballots):
        ranking = _convert_slate_ballot_type_to_ranking(
            ballot_type=slate_ballot,
            cand_ordering_by_slate={
                s: cand_ordering[i] for s, cand_ordering in cand_orderings_by_slate.items()
            },
            final_max_ranking_length=n_candidates,
        )
        ballot_pool[i] = np.array(ranking)

    df = pd.DataFrame(ballot_pool)
    df.index.name = BALLOT_INDEX_NAME
    df.columns = [f"{RANKING_COLUMN_PREFIX}{i + 1}" for i in range(n_candidates)]
    df[WEIGHT_COLUMN] = 1
    df.insert(
        len(df.columns),
        VOTER_SET_COLUMN,
        pd.Series([frozenset()] * len(df), dtype=object, index=df.index),
    )
    pp = RankProfile(
        candidates=config.candidates,
        df=df,
        max_ranking_length=n_candidates,
    )

    return pp


def _append_zero_slate_symbols(
    ballots: list[tuple[str, ...]],
    zero_slate_set: set[str],
    n_ballots: int,
    config: BlocSlateConfig,
    *,
    numpy_rng: Generator | None = None,
) -> list[tuple[str, ...]]:
    """
    Append zero cohesion slate symbols to the end of slate ballot types. Randomly permutes the
    zero cohesion slate symbols for each ballot.

    Args:
        ballots (list[tuple[str, ...]]): List of slate ballot types.
        zero_slate_set (set[str]): Set of slates with zero cohesion for the given bloc.
        n_ballots (int): The number of ballots to generate.
        config (BlocSlateConfig): Configuration object containing all necessary parameters for
            working with a bloc-slate ballot generator.
        numpy_rng (Generator, optional): NumPy random number generator. Pass a seeded instance for
            reproducible results. Defaults to None for non-deterministic results.

    Returns:
        list[list[str, ...]]: List of slate ballot types with zero cohesion slate symbols
            appended.
    """
    zero_slate_symbols = [
        slate for slate in sorted(zero_slate_set) for c in config.slate_to_candidates[slate]
    ]
    numpy_rng = np.random.default_rng() if numpy_rng is None else numpy_rng
    noise = numpy_rng.random(size=(n_ballots, len(zero_slate_symbols)))
    permutation_indices = np.argsort(noise, axis=1)
    zero_slate_orderings = np.array(list(zero_slate_symbols), dtype=object)[permutation_indices]

    ballots = [ballot + tuple(zero_slate_orderings[i]) for i, ballot in enumerate(ballots)]
    return ballots
