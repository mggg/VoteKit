"""
Generate ranked preference profiles using the slate-Plackett-Luce model.

The main API functions in this module are:

- `slate_pl_profile_generator`: Generates a single preference profile using the slate-Plackett-Luce
    model.
- `slate_pl_profiles_by_bloc_generator`: Generates preference profiles by bloc using the
    slate-Plackett-Luce model.
"""

from numbers import Real
from typing import Optional

import apportionment.methods as apportion
import numpy as np
import pandas as pd
from numpy.random import Generator

from votekit.ballot_generator.bloc_slate_generator.config import BlocSlateConfig
from votekit.ballot_generator.bloc_slate_generator.slate_utils import (
    _append_zero_slate_symbols,
    _convert_slate_ballots_to_candidate_ids,
)
from votekit.pref_profile import RankProfile

# ====================================================
# ================= Helper Functions =================
# ====================================================


def _sample_pl_slate_ballots(
    config: BlocSlateConfig,
    num_ballots: int,
    bloc: str,
    non_zero_slate_set: set[str],
    *,
    numpy_rng: Optional[Generator] = None,
) -> np.ndarray:
    """
    Returns an array of slate ballots; each row contains the indexes into ``config.slates`` in the
    order the slates appear on that ballot. Only fills positions for slates that have non-zero
    cohesion for the given bloc; remaining positions are filled with -1.


    Args:
        config (BlocSlateConfig): Configuration object containing all necessary parameters for
            working with a bloc-slate ballot generator.
        num_ballots (int): The number of ballots to generate.
        bloc (str): The name of the bloc for which to generate slate ballots.
        non_zero_slate_set (set[str]): Set of slates with non-zero cohesion for the given bloc.
        numpy_rng (Generator, optional): NumPy random number generator seeded with a known value for
            reproducible results. By default, seeded with None to generate different results each
            time.

    Returns:
        np.ndarray: An array of shape (num_ballots, len(config.candidates)), where each row contains
            the slate indexes in the order they appear on the ballot. Padded with -1.
    """
    slates = list(sorted(non_zero_slate_set))
    num_candidates_per_slate = np.array([len(config.slate_to_candidates[s]) for s in slates])

    num_candidates = int(num_candidates_per_slate.sum())

    numpy_rng = np.random.default_rng() if numpy_rng is None else numpy_rng
    rand_unif_seqs = numpy_rng.uniform(size=(num_ballots, num_candidates))

    slates = list(sorted(non_zero_slate_set))
    cohesion_values_og: list[float] = []
    for slate in slates:
        cohesion_value = config.cohesion_df.loc[bloc, slate]
        if not isinstance(cohesion_value, Real) or isinstance(cohesion_value, bool):
            raise TypeError(
                f"Cohesion must be numeric. Found {type(cohesion_value)!r} for slate {slate!r}."
            )
        cohesion_values_og.append(float(cohesion_value))

    cohesion_values = np.tile(cohesion_values_og, (num_ballots, 1))
    slate_counts = np.zeros(cohesion_values.shape, dtype=int)
    ballot_types = np.empty((num_ballots, num_candidates), dtype=int)
    ballots = np.arange(num_ballots)
    for ranking_idx in range(num_candidates):
        bin_right = np.cumsum(cohesion_values, axis=1)
        bin_left = np.zeros_like(bin_right)
        bin_left[:, 1:] = bin_right[:, :-1]
        ranking_idx_samples = rand_unif_seqs[:, ranking_idx, None]  # shape (num_ballots, 1)
        within_bin = (bin_left < ranking_idx_samples) & (ranking_idx_samples <= bin_right)
        slate_indices = np.where(within_bin.any(axis=1), within_bin.argmax(axis=1), len(slates) - 1)

        ballot_types[:, ranking_idx] = slate_indices
        slate_counts[ballots, slate_indices] += 1

        # check if any slate has reached its max count and set its cohesion to 0 for those ballots
        # check across all ballots for only the slate added in this ranking index
        if ranking_idx < num_candidates - 1:
            full_slate_mask = ballots[
                slate_counts[ballots, slate_indices] == num_candidates_per_slate[slate_indices]
            ]
            cohesion_values[full_slate_mask, slate_indices[full_slate_mask]] = 0
            cohesion_values[full_slate_mask] /= cohesion_values[full_slate_mask].sum(
                axis=1, keepdims=True
            )

    # pad ballot types with -1 for zero cohesion slate candidates
    slate_ballots = np.full((num_ballots, len(config.candidates)), -1, dtype=int)
    config_slate_index = np.array([config.slates.index(slate) for slate in slates], dtype=int)
    slate_ballots[:, :num_candidates] = config_slate_index[ballot_types]

    return slate_ballots


# ===========================================================
# ================= Interior Work Functions =================
# ===========================================================


def _inner_slate_plackett_luce(
    config: BlocSlateConfig,
    *,
    rng_seed: Optional[int] = None,
) -> dict[str, np.ndarray]:
    """
    Inner function to generate preference profiles by bloc using the slate-Plackett-Luce model.

    This model first samples a ballot type by flipping a cohesion parameter weighted coin.
    It then fills out the ballot type via sampling without replacement from the interval
    (i.e. according to the name-Plackett-Luce model).

    Slates with zero cohesion for the given bloc are randomly permuted at the end of the ballot.
    Candidates with zero support are randomly permuted at the end of the candidate ordering.

    Args:
        config (BlocSlateConfig): Configuration object containing all necessary parameters for
            working with a bloc-slate ballot generator.
        ballot_length (Optional[int]): Number of votes allowed per ballot. If None, this is
            set to the total number of candidates in the configuration. Defaults to None.
        rng_seed (optional[int]): Seed for random number generator. An integer seed produces the
            same output given identical inputs; By default, seed is None which gives
            non-deterministic results.

    Returns:
        dict[str, np.ndarray]: Dictionary whose keys are bloc strings and values are arrays of
            shape (n_ballots, n_candidates) of indices into ``config.candidates`` for that bloc.
    """
    bloc_lst = config.blocs

    bloc_counts = apportion.compute(
        "huntington", list(config.bloc_proportions.values()), config.n_voters
    )
    if not isinstance(bloc_counts, list):
        if not isinstance(bloc_counts, int):
            raise TypeError(f"Unexpected type from apportionment got {type(bloc_counts)}")

        bloc_counts = [bloc_counts]

    ballots_per_bloc = {bloc: bloc_counts[i] for i, bloc in enumerate(bloc_lst)}

    candidate_ids_by_bloc: dict[str, np.ndarray] = {}
    numpy_rng = np.random.default_rng(seed=rng_seed)

    for bloc in bloc_lst:
        n_ballots = ballots_per_bloc[bloc]

        non_zero_slate_set = {
            slate for slate in config.slates if config.cohesion_df.loc[bloc][slate] > 0
        }

        zero_slate_set = set(config.slates) - non_zero_slate_set

        slate_ballots = _sample_pl_slate_ballots(
            config=config,
            num_ballots=n_ballots,
            bloc=bloc,
            non_zero_slate_set=non_zero_slate_set,
            numpy_rng=numpy_rng,
        )

        if len(zero_slate_set) != 0:
            slate_ballots = _append_zero_slate_symbols(
                slate_ballots,
                zero_slate_set,
                n_ballots,
                config,
                numpy_rng=numpy_rng,
            )
        candidate_ids_by_bloc[bloc] = _convert_slate_ballots_to_candidate_ids(
            config, bloc, slate_ballots, numpy_rng=numpy_rng
        )

    return candidate_ids_by_bloc


# =================================================
# ================= API Functions =================
# =================================================


def slate_pl_profile_generator(
    config: BlocSlateConfig,
    *,
    group_ballots: bool = True,
    rng_seed: Optional[int] = None,
) -> RankProfile:
    """
    Generates a merged preference profile using the slate-Plackett-Luce model.

    This model first samples a ballot type by flipping a cohesion parameter weighted coin.
    It then fills out the ballot type via sampling without replacement from the preference
    interval (i.e. according to the name-Plackett-Luce model).

    Slates with zero cohesion for the given bloc are randomly permuted at the end of the ballot.
    Candidates with zero support are randomly permuted at the end of the candidate ordering.

    Args:
        config (BlocSlateConfig): Configuration object containing all necessary parameters for
            working with a bloc-slate ballot generator.
        group_ballots (bool): If True, group identical ballots in the returned profile and
            set the weight accordingly. Defaults to True.
        rng_seed (int, optional)): Seed for random number generator. An integer seed produces the
            same output given identical inputs; By default, seed is None which gives
            non-deterministic results.

    Returns:
        RankProfile: Merged ``RankProfile`` object generated by the model.
    """
    config.is_valid(raise_errors=True)

    candidate_ids_by_bloc = _inner_slate_plackett_luce(config, rng_seed=rng_seed)

    rankings = np.concatenate(list(candidate_ids_by_bloc.values()), axis=0)
    candidate_sets = np.array(
        [frozenset({"~"})] + [frozenset({c}) for c in config.candidates], dtype=object
    )
    df = pd.DataFrame(
        candidate_sets[rankings + 1],
        columns=[f"Ranking_{i + 1}" for i in range(len(config.candidates))],
        copy=False,
    )
    df.index.name = "Ballot Index"
    df["Weight"] = 1
    df["Voter Set"] = pd.Series([frozenset()] * len(df), dtype=object, index=df.index)
    pp = RankProfile(df=df, candidates=config.candidates, max_ranking_length=len(config.candidates))

    if group_ballots:
        pp = pp.group_ballots()

    return pp


def slate_pl_profiles_by_bloc_generator(
    config: BlocSlateConfig,
    *,
    group_ballots: bool = True,
    rng_seed: Optional[int] = None,
) -> dict[str, RankProfile]:
    """
    Generates a dictionary mapping bloc names to preference profiles using the slate-Plackett-Luce
    model.

    This model first samples a ballot type by flipping a cohesion parameter weighted coin.
    It then fills out the ballot type via sampling without replacement from the preference
    interval (i.e. according to the name-Plackett-Luce model).

    Slates with zero cohesion for the given bloc are randomly permuted at the end of the ballot.
    Candidates with zero support are randomly permuted at the end of the candidate ordering.

    Args:
        config (BlocSlateConfig): Configuration object containing all necessary parameters for
            working with a bloc-slate ballot generator.
        group_ballots (bool): If True, group identical ballots in the returned profile and
            set the weight accordingly. Defaults to True.
        rng_seed (int, optional)): Seed for random number generator. An integer seed produces the
            same output given identical inputs; By default, seed is None which gives
            non-deterministic results.

    Returns:
        dict[str, RankProfile]: Dictionary whose keys are bloc strings and values are
            ``RankProfile`` objects generated by the model for each bloc.
    """
    config.is_valid(raise_errors=True)

    candidate_ids_by_bloc = _inner_slate_plackett_luce(config, rng_seed=rng_seed)

    pp_by_bloc: dict[str, RankProfile] = {}
    for bloc in candidate_ids_by_bloc:
        rankings = candidate_ids_by_bloc[bloc]
        candidate_sets = np.array(
            [frozenset({"~"})] + [frozenset({c}) for c in config.candidates], dtype=object
        )
        df = pd.DataFrame(
            candidate_sets[rankings + 1],
            columns=[f"Ranking_{i + 1}" for i in range(len(config.candidates))],
            copy=False,
        )
        df.index.name = "Ballot Index"
        df["Weight"] = 1
        df["Voter Set"] = pd.Series([frozenset()] * len(df), dtype=object, index=df.index)
        pp = RankProfile(
            df=df, candidates=config.candidates, max_ranking_length=len(config.candidates)
        )

        if group_ballots:
            pp = pp.group_ballots()

        pp_by_bloc[bloc] = pp

    return pp_by_bloc
