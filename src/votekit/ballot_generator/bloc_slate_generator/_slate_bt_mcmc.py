"""Generate slate-BradleyTerry profiles using Markov Chain Monte Carlo."""

from __future__ import annotations

import random
from typing import cast

import apportionment.methods as apportion
import numpy as np

from votekit.ballot_generator.bloc_slate_generator.config import BlocSlateConfig
from votekit.ballot_generator.bloc_slate_generator.slate_utils import (
    _append_zero_slate_symbols,
    _convert_slate_ballots_to_profile,
)
from votekit.pref_profile import RankProfile


def _sample_bt_slate_ballots_mcmc(
    config: BlocSlateConfig,
    bloc_name: str,
    n_ballots: int,
    non_zero_slate_set: set[str],
    *,
    rng: random.Random | None = None,
) -> list[tuple[str, ...]]:
    """
    Generates ballot types (e.g. AABABB) for a given bloc using a Markov Chain Monte Carlo (MCMC)
    estimation of the slate Bradley-Terry model.
    Only generates ballots for slates that have non-zero cohesion for the given bloc.

    Args:
        config (BlocSlateConfig): Configuration object containing all necessary parameters for
            working with a bloc-slate ballot generator.
        bloc_name (str): The name of the voter bloc for which to generate ballot types.
        n_ballots (int): The number of ballots to generate.
        non_zero_slate_set (set[str]): Set of slates with non-zero cohesion for the given bloc.
        rng (random.Random, optional): Standard library random number generator. Pass a seeded
            instance for reproducible results. Defaults to None for non-deterministic results.

    Returns:
        list[tuple[str]]: A list of ballot types, where each ballot type is represented
            as a tuple of slate names.
    """

    # AABABB like
    seed_ballot_type = [
        slate for slate in sorted(non_zero_slate_set) for c in config.slate_to_candidates[slate]
    ]
    # randomly permute the seed ballot type
    rng = random.Random() if rng is None else rng
    seed_ballot_type = rng.sample(seed_ballot_type, len(seed_ballot_type))

    ballots: list[tuple[str, ...]] = [("~",) for _ in range(n_ballots)]
    current_ranking = seed_ballot_type
    # presample swap indices
    swap_indices = [
        (j1, j1 + 1) for j1 in rng.choices(range(len(seed_ballot_type) - 1), k=n_ballots)
    ]

    for i in range(n_ballots):
        j1, j2 = swap_indices[i]
        current_slate, new_slate = current_ranking[j1], current_ranking[j2]

        # swap probability is how much voters in given bloc tend to prefer current slate
        # over how much they tend to prefer new slate
        cohesion_j1 = config.cohesion_df[current_slate].loc[bloc_name]
        cohesion_j2 = config.cohesion_df[new_slate].loc[bloc_name]
        acceptance_prob = cohesion_j2 / cohesion_j1  # Doesn't matter if above 1

        if rng.random() < acceptance_prob:
            current_ranking[j1], current_ranking[j2] = (
                current_ranking[j2],
                current_ranking[j1],
            )

        ballots[i] = tuple(current_ranking.copy())

    return ballots


def _sample_slate_bt_mcmc_profiles(
    config: BlocSlateConfig,
    *,
    rng_seed: int | None = None,
) -> dict[str, RankProfile]:
    """
    Inner function to generate preference profiles using the slate-BradleyTerry model.

    Slates with zero cohesion for the given bloc are randomly permuted at the end of the ballot.
    Candidates with zero support are randomly permuted at the end of the candidate ordering.

    Args:
        config (BlocSlateConfig): Configuration object containing all necessary parameters for
            working with a bloc-slate ballot generator.
        rng_seed (int, optional)): Seed for random number generator. An integer seed produces the
            same output given identical inputs; By default, seed is None which gives
            non-deterministic results.

    Returns:
        dict[str, RankProfile]: A dictionary mapping bloc names to their corresponding
            generated preference profiles.
    """
    # Save on repeated calls to computed property
    bloc_lst = config.blocs

    bloc_counts = apportion.compute(
        "huntington", list(config.bloc_proportions.values()), config.n_voters
    )
    if not isinstance(bloc_counts, list):
        if not isinstance(bloc_counts, int):
            raise TypeError(f"Unexpected type from apportionment got {type(bloc_counts)}")

        bloc_counts = [bloc_counts]

    ballots_per_bloc = {bloc: bloc_counts[i] for i, bloc in enumerate(bloc_lst)}

    pref_profile_by_bloc = {b: RankProfile() for b in bloc_lst}
    numpy_rng = np.random.default_rng(seed=rng_seed)
    rng = random.Random(rng_seed)

    for bloc in bloc_lst:
        # number of voters in this bloc
        n_ballots = ballots_per_bloc[bloc]
        non_zero_slate_set = {
            slate for slate in config.slates if cast(float, config.cohesion_df.at[bloc, slate]) > 0
        }

        zero_slate_set = set(config.slates) - non_zero_slate_set

        slate_ballots = _sample_bt_slate_ballots_mcmc(
            config=config,
            bloc_name=bloc,
            n_ballots=n_ballots,
            non_zero_slate_set=non_zero_slate_set,
            rng=rng,
        )

        if len(zero_slate_set) != 0:
            slate_ballots = _append_zero_slate_symbols(
                slate_ballots,
                zero_slate_set,
                n_ballots,
                config,
                numpy_rng=numpy_rng,
            )

        pref_profile_by_bloc[bloc] = _convert_slate_ballots_to_profile(
            config,
            bloc,
            slate_ballots,
            numpy_rng=numpy_rng,
        )

    return pref_profile_by_bloc


def slate_bt_profile_generator_using_mcmc(
    config: BlocSlateConfig,
    *,
    group_ballots=True,
    rng_seed: int | None = None,
) -> RankProfile:
    """
    Generate a ranked preference profile using the slate-BradleyTerry model.

    This model is useful when the count-state table required by direct slate-BT sampling
    exceeds the memory available on the current machine.

    The MCMC version of this model uses a Markov Chain Monte Carlo method to sample
    ballot types according to the slate-BradleyTerry model. After selecting a seed ballot,
    the model proposes swaps of adjacent slates in the ballot type and accepts or rejects
    the swap according to the ratio of the cohesion parameters of the two slates being swapped
    within a given block.

    Once the ballot type is sampled, the candidate names for each of the positions is filled
    out by sampling without replacement within each slate according to the preference interval
    of that slate in the given bloc (i.e. according to the name-Plackett-Luce model).

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
        RankProfile: Generated preference profile.
    """
    config.is_valid(raise_errors=True)

    pp_by_bloc = _sample_slate_bt_mcmc_profiles(config, rng_seed=rng_seed)

    profile = RankProfile()
    for prof in pp_by_bloc.values():
        profile += prof

    if group_ballots:
        profile = profile.group_ballots()

    return profile


def slate_bt_profiles_by_bloc_generator_using_mcmc(
    config: BlocSlateConfig,
    *,
    group_ballots=True,
    rng_seed: int | None = None,
) -> dict[str, RankProfile]:
    """
    Generate a dictionary mapping bloc names to ranked preference profiles using the
    slate-BradleyTerry model.

    This model is useful when the count-state table required by direct slate-BT sampling
    exceeds the memory available on the current machine.

    The MCMC version of this model uses a Markov Chain Monte Carlo method to sample
    ballot types according to the slate-BradleyTerry model. After selecting a seed ballot,
    the model proposes swaps of adjacent slates in the ballot type and accepts or rejects
    the swap according to the ratio of the cohesion parameters of the two slates being swapped
    within a given block.

    Once the ballot type is sampled, the candidate names for each of the positions is filled
    out by sampling without replacement within each slate according to the preference interval
    of that slate in the given bloc (i.e. according to the name-Plackett-Luce model).

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
        dict[str, RankProfile]: Generated preference profiles by bloc.
    """
    config.is_valid(raise_errors=True)

    pp_by_bloc = _sample_slate_bt_mcmc_profiles(config, rng_seed=rng_seed)
    if group_ballots:
        for bloc in pp_by_bloc:
            pp_by_bloc[bloc] = pp_by_bloc[bloc].group_ballots()

    return pp_by_bloc
