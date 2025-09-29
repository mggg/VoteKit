"""
Generate ranked preference profiles using the slate-BradleyTerry model.

The main API functions in this module are:

- `slate_bt_profile_generator`: Generates a single preference profile using the slate-BradleyTerry
    model.
- `slate_bt_profiles_by_bloc_generator`: Generates preference profiles by bloc using the
    slate-BradleyTerry model.
- `slate_bt_profile_generator_using_mcmc`: Generates a single preference profile using MCMC
    sampling from the slate-BradleyTerry model.
- `slate_bt_profiles_by_bloc_generator_using_mcmc`: Generates preference profiles by bloc using
    MCMC sampling from the slate-BradleyTerry model.
"""

import itertools as it
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import math
import sys

import random

from typing import Sequence, cast
import apportionment.methods as apportion

from votekit.pref_profile import RankProfile
from votekit.ballot_generator.bloc_slate_generator.model import BlocSlateConfig
from votekit.ballot_generator.utils import system_memory
from votekit.ballot_generator.bloc_slate_generator.slate_utils import (
    _lexicographic_symbol_tuple_iterator,
    _make_cand_ordering_by_slate,
    _convert_ballot_type_to_ranking,
)

# ====================================================
# ================= Helper Functions =================
# ====================================================


def _count_pairs_a_before_b(x_array: NDArray, a: str, b: str) -> int:
    """
    Count the number of times a appears before b in x_array.

    Args:
        x_array (NDArray): 1D numpy array of strings.
        a (str): The string to count before b.
        b (str): The string to count after a.

    Returns:
        int: The number of times a appears before b in x_array.
    """
    cumulative_a = np.cumsum(x_array == a)
    ret = int(cumulative_a[x_array == b].sum())
    return ret


def _slate_bt_numerator_computation_single_bloc(
    ballot_type: Sequence[str], slate_cohesion_dict_for_single_bloc: dict[str, float]
):
    """
    Compute the numerator of the probability of a given ballot type for a single bloc.

    Args:
        ballot_type (Sequence[str]): A tuple of strings representing the ballot type.
        slate_cohesion_dict_for_single_bloc (dict[str, float]): A dictionary mapping slate
            names to their cohesion parameters for a single bloc.

    Returns:
        float: The numerator of the probability of the given ballot type.
    """
    # for each slate pair (i,j), count times slate i appears above slate j
    # component is then (cohesion[i] / (cohesion[i] + cohesion[j])) ^ count(i above j)
    all_slates = frozenset(slate_cohesion_dict_for_single_bloc.keys())
    ballot_arr = np.array(ballot_type)
    result = 1.0
    for slate, other_slate in it.permutations(all_slates, 2):
        result *= pow(
            slate_cohesion_dict_for_single_bloc[slate]
            / (
                slate_cohesion_dict_for_single_bloc[slate]
                + slate_cohesion_dict_for_single_bloc[other_slate]
            ),
            _count_pairs_a_before_b(ballot_arr, slate, other_slate),
        )

    return result


def _compute_ballot_type_dist(
    config: BlocSlateConfig, bloc: str, non_zero_candidate_set: set[str]
) -> dict[tuple[str, ...], float]:
    """
    Compute the probability distribution for ballot types for a given voter bloc.

    Args:
        config (BlocSlateConfig): The configuration for the bloc-slate type model.
        bloc (str): The voter bloc for which to compute the ballot type distribution.

    Returns:
        dict[tuple[str, ...], float]: A dictionary mapping ballot types (as tuples of
            slate names) to their probabilities.
    """

    # FIX: Do this for the non-zero support candidates
    slate_list = [
        s_name
        for s_name in config.slate_to_candidates.keys()
        for c_name in config.slate_to_candidates[s_name]
        if c_name in non_zero_candidate_set
    ]

    bloc_series = config.cohesion_df.loc[bloc]
    bloc_series.index = bloc_series.index.astype(str)
    bloc_series = bloc_series.astype(float)

    slate_cohesion_dict_for_bloc: dict[str, float] = cast(
        dict[str, float], bloc_series.to_dict()
    )

    pmf = {}
    for ballot_type in _lexicographic_symbol_tuple_iterator(slate_list):
        if ballot_type not in pmf:
            pmf[ballot_type] = _slate_bt_numerator_computation_single_bloc(
                ballot_type, slate_cohesion_dict_for_bloc
            )

    summ = sum(pmf.values())
    return {b: v / summ for b, v in pmf.items()}


def _sample_ballot_types_deterministic(
    config: BlocSlateConfig,
    bloc_name: str,
    n_ballots: int,
    non_zero_candidate_set: set[str],
) -> list[tuple[str, ...]]:
    """
    Generates ballot types (e.g. AABABB) for a given bloc using the slate Bradley-Terry model.

    Args:
        config (BlocSlateConfig): Configuration object containing all necessary parameters for
            working with a bloc-slate ballot generator.
        bloc_name (str): The name of the voter bloc for which to generate ballot types.
        n_ballots (int): The number of ballots to generate.
        non_zero_candidate_set (set[str]): Set of candidates that have non-zero preference
            intervals in the given bloc.

    Returns:
        list[tuple[str]]: A list of ballot types, where each ballot type is represented
            as a tuple of slate names.
    """
    pdf = _compute_ballot_type_dist(config, bloc_name, non_zero_candidate_set)
    b_types: list[tuple[str, ...]] = list(pdf.keys())
    probs = list(pdf.values())

    sampled_indices = np.random.choice(len(b_types), size=n_ballots, p=probs)

    return [b_types[i] for i in sampled_indices]


def _check_slate_bt_memory(config: BlocSlateConfig) -> None:
    """
    Check if there is enough memory to generate the profile using the slate-BradleyTerry model.

    Args:
        config (BlocSlateConfig): Configuration object containing all necessary parameters for
            working with a bloc-slate ballot generator.

    Raises:
        ValueError: If there are more than 12! possible ballot types.
        MemoryError: If there is not enough memory to generate the pmf.
    """
    n_cands = len(config.candidates)
    slate_counts = {
        slate: len(cands) for slate, cands in config.slate_to_candidates.items()
    }
    total_arrangements = math.factorial(n_cands) / math.prod(
        math.factorial(count) for count in slate_counts.values()
    )
    if total_arrangements > math.factorial(12):
        raise ValueError(
            "Given the number of candidates and slates you have entered, there appears to be "
            f"{total_arrangements:.2e} possible ballot types. This is beyond the standard limit "
            f"of 12! = {math.factorial(12)}. Please reduce the number of candidates or use the "
            "MCMC version of this generator instead."
        )

    mem = system_memory()
    pmf_size = total_arrangements
    candidate_with_longest_name = max(config.candidates, key=len)
    est_bytes_pmf = pmf_size * sys.getsizeof(candidate_with_longest_name) * n_cands
    est_bytes_profile = (
        config.n_voters
        * n_cands
        * sys.getsizeof(frozenset({candidate_with_longest_name}))
    )
    est_bytes = float(est_bytes_pmf + est_bytes_profile)

    # fudge factor for overhead. Just tuned to a couple of machines, but gives pretty close
    # upper bound on memory usage while leaving room for other processes
    est_bytes *= 1.5
    if est_bytes > mem["available_gib"] * 2**30:
        raise MemoryError(
            f"Not enough memory to generate the profile. Estimated memory usage is "
            f"{est_bytes / 2**30:.1f} GiB, but only {mem['available_gib']:.1f} GiB is available."
        )


def _sample_ballot_types_mcmc(
    config: BlocSlateConfig,
    bloc_name: str,
    n_ballots: int,
    non_zero_candidate_set: set[str],
) -> list[tuple[str, ...]]:
    """
    Generates ballot types (e.g. AABABB) for a given bloc using a Markov Chain Monte Carlo (MCMC)
    estimation of the slate Bradley-Terry model.

    Args:
        config (BlocSlateConfig): Configuration object containing all necessary parameters for
            working with a bloc-slate ballot generator.
        bloc_name (str): The name of the voter bloc for which to generate ballot types.
        n_ballots (int): The number of ballots to generate.
        non_zero_candidate_set (set[str]): Set of candidates that have non-zero preference
            intervals in the given bloc.

    Returns:
        list[tuple[str]]: A list of ballot types, where each ballot type is represented
            as a tuple of slate names.
    """

    # AABABB like
    seed_ballot_type = [
        slate
        for slate in config.slates
        for c in config.slate_to_candidates[slate]
        if c in non_zero_candidate_set
    ]
    # randomly permute the seed ballot type
    seed_ballot_type = random.sample(seed_ballot_type, k=len(seed_ballot_type))

    ballots: list[tuple[str, ...]] = [("~",)] * n_ballots
    current_ranking = seed_ballot_type

    # presample swap indices
    swap_indices = [
        (j1, j1 + 1)
        for j1 in np.random.choice(len(seed_ballot_type) - 1, size=n_ballots)
    ]

    for i in range(n_ballots):
        j1, j2 = swap_indices[i]
        current_slate, new_slate = current_ranking[j1], current_ranking[j2]

        # swap probability is how much voters in given bloc tend to prefer current slate
        # over how much they tend to prefer new slate
        cohesion_j1 = config.cohesion_df[current_slate].loc[bloc_name]
        cohesion_j2 = config.cohesion_df[new_slate].loc[bloc_name]
        acceptance_prob = cohesion_j2 / cohesion_j1  # Doesn't matter if above 1

        if random.random() < acceptance_prob:
            current_ranking[j1], current_ranking[j2] = (
                current_ranking[j2],
                current_ranking[j1],
            )

        ballots[i] = tuple(current_ranking.copy())

    return ballots


# ===========================================================
# ================= Interior Work Functions =================
# ===========================================================


def _inner_slate_bradley_terry(
    config: BlocSlateConfig,
    use_mcmc: bool = False,
) -> dict[str, RankProfile]:
    """
    Args:
        config (BlocSlateConfig): Configuration object containing all necessary parameters for
            working with a bloc-slate ballot generator.
        use_mcmc (bool): If True, use MCMC to sample ballot types. Defaults to False.

    Returns:
        dict[str, RankProfile]: A dictionary mapping bloc names to their corresponding
            generated preference profiles.
    """
    n_candidates = len(config.candidates)

    # Save on repeated calls to computed property
    bloc_lst = config.blocs

    bloc_counts = apportion.compute(
        "huntington", list(config.bloc_proportions.values()), config.n_voters
    )
    if not isinstance(bloc_counts, list):
        if not isinstance(bloc_counts, int):
            raise TypeError(
                f"Unexpected type from apportionment got {type(bloc_counts)}"
            )

        bloc_counts = [bloc_counts]

    ballots_per_bloc = {bloc: bloc_counts[i] for i, bloc in enumerate(bloc_lst)}

    pref_profile_by_bloc = {b: RankProfile() for b in bloc_lst}
    candidates = config.candidates

    for bloc in bloc_lst:
        # number of voters in this bloc
        n_ballots = ballots_per_bloc[bloc]
        ballot_pool = np.full((n_ballots, n_candidates), frozenset("~"))
        pref_intervals_by_slate_dict = config.get_preference_intervals_for_bloc(bloc)
        zero_cands = set(
            it.chain(*[pi.zero_cands for pi in pref_intervals_by_slate_dict.values()])
        )
        non_zero_cands_set = set(candidates) - zero_cands

        if use_mcmc:
            ballot_types = _sample_ballot_types_mcmc(
                config=config,
                bloc_name=bloc,
                n_ballots=n_ballots,
                non_zero_candidate_set=non_zero_cands_set,
            )
        else:
            ballot_types = _sample_ballot_types_deterministic(
                config=config,
                bloc_name=bloc,
                n_ballots=n_ballots,
                non_zero_candidate_set=non_zero_cands_set,
            )

        for j, bt in enumerate(ballot_types):
            cand_ordering_by_slate = _make_cand_ordering_by_slate(
                config, pref_intervals_by_slate_dict
            )
            ranking = _convert_ballot_type_to_ranking(
                ballot_type=bt, cand_ordering_by_slate=cand_ordering_by_slate
            )
            if ranking is None:
                raise RuntimeError(
                    "Unexpeceted None from internal function _convert_ballot_type_to_ranking "
                    "Likely caused by an empty ballot type."
                )

            if len(zero_cands) > 0:
                ranking.append(frozenset(zero_cands))
            ballot_pool[j] = np.array(ranking)

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
        pref_profile_by_bloc[bloc] = pp

    return pref_profile_by_bloc


# =================================================
# ================= API Functions =================
# =================================================


def slate_bt_profile_generator(
    config: BlocSlateConfig, *, group_ballots=True
) -> RankProfile:
    """
    Generate a preference profile using the name-BradleyTerry model.

    This model first samples a ballot type (e.g. AABABB) according the the Bradley-Terry model
    using the cohesion parameters for each slate and the candidate counts of those slates (so
    the utilities of each of the candidates in a slate are assumed to be uniform in this stage).

    Once the ballot type is sampled, the candidate names for each of the positions is filled
    out by sampling without replacement within each slate according to the preference interval
    of that slate in the given bloc (i.e. according to the name-Plackett-Luce model).

    Args:
        config (BlocSlateConfig): Configuration object containing all necessary parameters for
            working with a bloc-slate ballot generator.
        group_ballots (bool): If True, group identical ballots in the returned profile and
            set the weight accordingly. Defaults to True.

    Returns:
        RankProfile: Generated preference profile.
    """
    _check_slate_bt_memory(config)

    config.is_valid(raise_errors=True)
    pp_by_bloc = _inner_slate_bradley_terry(config)

    profile = RankProfile()
    for prof in pp_by_bloc.values():
        profile += prof

    if group_ballots:
        profile = profile.group_ballots()

    return profile


def slate_bt_profiles_by_bloc_generator(
    config: BlocSlateConfig, *, group_ballots=True
) -> dict[str, RankProfile]:
    """
    Generate a dictionary mapping bloc names to ranked preference profiles using the
    slate-BradleyTerry model.

    This model first samples a ballot type (e.g. AABABB) according the the Bradley-Terry model
    using the cohesion parameters for each slate and the candidate counts of those slates (so
    the utilities of each of the candidates in a slate are assumed to be uniform in this stage).

    Once the ballot type is sampled, the candidate names for each of the positions is filled
    out by sampling without replacement within each slate according to the preference interval
    of that slate in the given bloc (i.e. according to the name-Plackett-Luce model).

    Args:
        config (BlocSlateConfig): Configuration object containing all necessary parameters for
            working with a bloc-slate ballot generator.
        group_ballots (bool): If True, group identical ballots in the returned profile and
            set the weight accordingly. Defaults to True.

    Returns:
        dict[str, RankProfile]: Generated preference profiles by bloc.
    """
    _check_slate_bt_memory(config)
    config.is_valid(raise_errors=True)

    pp_by_bloc = _inner_slate_bradley_terry(config)
    if group_ballots:
        for bloc in pp_by_bloc:
            pp_by_bloc[bloc] = pp_by_bloc[bloc].group_ballots()

    return pp_by_bloc


def slate_bt_profile_generator_using_mcmc(
    config: BlocSlateConfig, *, group_ballots=True
) -> RankProfile:
    """
    Generate a ranked preference profile using the slate-BradleyTerry model.

    This model is mainly useful when then number of possible ballot types is too large
    to compute the full probability distribution on the present hardware (e.g. more than 12!
    possible ballot types).

    The MCMC version of this model uses a Markov Chain Monte Carlo method to sample
    ballot types according to the slate-BradleyTerry model. After selecting a seed ballot,
    the model proposes swaps of adjacent slates in the ballot type and accepts or rejects
    the swap according to the ratio of the cohesion parameters of the two slates being swapped
    within a given block.

    Once the ballot type is sampled, the candidate names for each of the positions is filled
    out by sampling without replacement within each slate according to the preference interval
    of that slate in the given bloc (i.e. according to the name-Plackett-Luce model).

    Args:
        config (BlocSlateConfig): Configuration object containing all necessary parameters for
            working with a bloc-slate ballot generator.
        group_ballots (bool): If True, group identical ballots in the returned profile and
            set the weight accordingly. Defaults to True.

    Returns:
        RankProfile: Generated preference profile.
    """
    config.is_valid(raise_errors=True)

    pp_by_bloc = _inner_slate_bradley_terry(config, use_mcmc=True)

    profile = RankProfile()
    for prof in pp_by_bloc.values():
        profile += prof

    if group_ballots:
        profile = profile.group_ballots()

    return profile


def slate_bt_profiles_by_bloc_generator_using_mcmc(
    config: BlocSlateConfig, *, group_ballots=True
) -> dict[str, RankProfile]:
    """
    Generate a dictionary mapping bloc names to ranked preference profiles using the
    slate-BradleyTerry model.

    This model is mainly useful when then number of possible ballot types is too large
    to compute the full probability distribution on the present hardware (e.g. more than 12!
    possible ballot types).

    The MCMC version of this model uses a Markov Chain Monte Carlo method to sample
    ballot types according to the slate-BradleyTerry model. After selecting a seed ballot,
    the model proposes swaps of adjacent slates in the ballot type and accepts or rejects
    the swap according to the ratio of the cohesion parameters of the two slates being swapped
    within a given block.

    Once the ballot type is sampled, the candidate names for each of the positions is filled
    out by sampling without replacement within each slate according to the preference interval
    of that slate in the given bloc (i.e. according to the name-Plackett-Luce model).

    Args:
        config (BlocSlateConfig): Configuration object containing all necessary parameters for
            working with a bloc-slate ballot generator.
        group_ballots (bool): If True, group identical ballots in the returned profile and
            set the weight accordingly. Defaults to True.

    Returns:
        dict[str, RankProfile]: Generated preference profiles by bloc.
    """
    config.is_valid(raise_errors=True)

    pp_by_bloc = _inner_slate_bradley_terry(config, use_mcmc=True)
    if group_ballots:
        for bloc in pp_by_bloc:
            pp_by_bloc[bloc] = pp_by_bloc[bloc].group_ballots()

    return pp_by_bloc
