"""
Generate ranked preference profiles using the name-BradleyTerry model.

The main API functions in this module are:

- `name_bt_profile_generator`: Generates a single preference profile using the name-BradleyTerry
    model.
- `name_bt_profiles_by_bloc_generator`: Generates preference profiles by bloc using the
    name-BradleyTerry model.
- `name_bt_profile_generator_using_mcmc`: Generates a single preference profile using MCMC
    sampling from the name-BradleyTerry model.
- `name_bt_profiles_by_bloc_generator_using_mcmc`: Generates preference profiles by bloc using
    MCMC sampling from the name-BradleyTerry model.
"""

import itertools as it
import math
import random
import sys
from collections.abc import Mapping

import apportionment.methods as apportion
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from votekit._df_schema import (
    BALLOT_INDEX_NAME,
    RANKING_COLUMN_PREFIX,
    VOTER_SET_COLUMN,
    WEIGHT_COLUMN,
)
from votekit.ballot import RankBallot
from votekit.ballot_generator.bloc_slate_generator.config import BlocSlateConfig
from votekit.ballot_generator.utils import system_memory
from votekit.pref_profile import RankProfile
from votekit.pref_profile.profile_matrix import (
    ProfileVoterWeightData,
    RankMatrix,
    WeightKind,
)
from votekit.pref_profile.profile_matrix.rank_matrix import EMPTY_VOTER_SET, END_OF_RANKING_SET
from votekit.types import Candidate

# ====================================================
# ================= Helper Functions =================
# ====================================================


def _calc_prob(permutations: list[tuple], cand_support_dict: dict) -> dict:
    """
    given a list of (possibly incomplete) rankings and the preference interval, \
    calculates the probability of observing each ranking

    Args:
        permutations (list[tuple]): a list of permuted rankings
        cand_support_dict (dict): a mapping from candidate to their \
        support (preference interval)

    Returns:
        dict: a mapping of the rankings to their probability
    """
    ranking_to_prob = {}
    for ranking in permutations:
        prob = 1
        for i in range(len(ranking)):
            cand_i = ranking[i]
            greater_cand_support = cand_support_dict[cand_i]
            for j in range(i + 1, len(ranking)):
                cand_j = ranking[j]
                cand_support = cand_support_dict[cand_j]
                prob *= greater_cand_support / (greater_cand_support + cand_support)
        ranking_to_prob[ranking] = prob
    return ranking_to_prob


def _make_bradley_terry_numerator(vals):
    """
    Given a list of values, returns the numerator of the Bradley-Terry probability
    for a given ranking.

    Args:
        vals (list): a list of values corresponding to the candidates in the ranking

    Returns:
        float: the numerator of the Bradley-Terry probability
    """
    ret = 1.0
    m = len(vals)

    # Faster than math.prod
    for i in range(m - 1):
        ret *= vals[i] ** (m - i - 1)
    return ret


def _bradley_terry_pdf(dct: Mapping[Candidate, float]) -> dict[tuple[Candidate, ...], float]:
    """
    Given a dictionary of candidates and their support, returns the probability density function
    over all possible rankings.

    Args:
        dct (Mapping[Candidate, float]): a mapping from candidate to their support
            Candidates can be strings, integers, or mix of both.

    Returns:
        dict: a mapping of the rankings to their probability
    """
    weights = {
        perm: _make_bradley_terry_numerator([dct[i] for i in perm])
        for perm in it.permutations(dct.keys(), len(dct))
    }
    total_weight = sum(weights.values())

    return {k: v / total_weight for k, v in weights.items()}


def _check_name_bt_memory(config: BlocSlateConfig) -> None:
    """
    Check if there is enough memory to generate the profile using the name-BradleyTerry model.

    Args:
        config (BlocSlateConfig): Configuration object containing all necessary parameters for
            working with a bloc-slate ballot generator.

    Raises:
        ValueError: If there are more than 12 candidates.
        MemoryError: If there is not enough memory to generate the pmf.
    """
    n_cands = len(config.candidates)
    if n_cands > 12:
        raise ValueError(
            "The name-BradleyTerry model is not recommended for more than 12 candidates due to "
            "combinatorial explosion of generating the pmf."
        )

    mem = system_memory()
    pmf_size = math.factorial(n_cands)
    candidate_with_longest_name = max(config.candidates, key=sys.getsizeof)
    est_bytes_pmf = pmf_size * sys.getsizeof(candidate_with_longest_name) * n_cands
    est_bytes_profile = (
        config.n_voters * n_cands * sys.getsizeof(frozenset({candidate_with_longest_name}))
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


# ===========================================================
# ================= Interior Work Functions =================
# ===========================================================


def _inner_name_bradley_terry(
    config: BlocSlateConfig, *, rng: random.Random | None = None
) -> dict[str, RankProfile]:
    """
    Sample from the BT distribution using direct sampling.

    This is an interior helper function that does the bulk of the work for generating
    name-BradleyTerry profiles. It is not recommended to use this function directly,
    as it does not perform any memory checks or grouping of ballots.

    Args:
        config (BlocSlateConfig): Configuration object containing all necessary parameters for
            working with a bloc-slate ballot generator.
        rng (random.Random, optional): Standard library random number generator. Pass a seeded
            instance for reproducible results. Defaults to None for non-deterministic results.

    Returns:
        dict[str, RankProfile]: Generated preference profiles by bloc.
    """
    n_candidates = len(config.candidates)

    bloc_lst = config.blocs

    bloc_counts = apportion.compute(
        "huntington", list(config.bloc_proportions.values()), config.n_voters
    )
    if not isinstance(bloc_counts, list):
        if not isinstance(bloc_counts, int):
            raise TypeError(f"Unexpected type from apportionment got {type(bloc_counts)}")

        bloc_counts = [bloc_counts]

    ballots_per_bloc = {bloc: bloc_counts[i] for i, bloc in enumerate(bloc_lst)}

    pp_by_bloc = {b: RankProfile() for b in bloc_lst}

    pref_interval_by_bloc_dict = config.get_combined_preference_intervals_by_bloc()
    rng = random.Random() if rng is None else rng

    for bloc in config.bloc_proportions.keys():
        n_ballots = ballots_per_bloc[bloc]

        # Directly initialize the list using good memory trick
        ballot_pool = np.full((n_ballots, n_candidates), frozenset("~"), dtype=object)
        single_bloc_pdf_dict = _bradley_terry_pdf(pref_interval_by_bloc_dict[bloc].interval)

        # Directly use the keys and values from the dictionary for sampling
        rankings, probs = zip(*single_bloc_pdf_dict.items())

        # The return of this will be a numpy array, so we don't need to make it into a list
        sampled_indices = np.array(
            rng.choices(
                range(len(rankings)),
                k=n_ballots,
                weights=probs,
            ),
            ndmin=1,
        )

        for j, index in enumerate(sampled_indices):
            ranking = [frozenset({cand}) for cand in rankings[index]]

            ballot_pool[j] = np.array(ranking)

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
        pp_by_bloc[bloc] = pp

    return pp_by_bloc


# TODO:
# - Improve the subsampling scheme here (memory + speed)
# - Construct ballots as tuples
# - Other speed improvements
def _bradley_terry_mcmc(
    n_ballots: int,
    pref_interval: Mapping[Candidate, float],
    seed_ballot: RankBallot,
    verbose: bool = False,
    burn_in_time: int = 0,
    chain_length: int | None = None,
    *,
    rng: random.Random | None = None,
):
    """
    Sample from BT distribution for a given preference interval using MCMC. Defaults
    to continuous sampling and no burn-in time.

    Args:
        n_ballots (int): the number of ballots to sample
        pref_interval (Mapping[Candidate, float]): the preference interval
            to determine BT distribution. Candidate can be a str or int.
        seed_ballot (RankBallot):  the seed ballot for the Markov chain
        verbose (bool): If True, print the acceptance ratio of the chain. Defaults to False.
        burn_in_time (int): the number of ballots discarded in the beginning of the chain
        chain_length (int | None): the length of the Markov Chain. Ballots are subsampled every
            chain_length//n_ballots steps from the chain until the desired number of ballots is
            reached. Defaults to None which sets the chain_length to the number of ballots in
            the config.
        rng (random.Random, optional): Standard library random number generator. Pass a seeded
            instance for reproducible results. Defaults to None for non-deterministic results.
    """

    if chain_length is None:
        chain_length = n_ballots

    assert seed_ballot.ranking is not None
    # check that seed ballot has no ties
    for s in seed_ballot.ranking:
        if len(s) > 1:
            raise ValueError("Seed ballot contains ties")

    ballots = [RankBallot()] * n_ballots
    accept = 0
    current_ranking = list(seed_ballot.ranking)
    n_candidates = len(current_ranking)

    # presample swap indices
    burn_in_time = burn_in_time  # int(10e5)
    if verbose:
        print(f"Burn in time: {burn_in_time}")
    rng = random.Random() if rng is None else rng
    swap_indices = [
        (j1, j1 + 1) for j1 in rng.choices(range(n_candidates - 1), k=n_ballots + burn_in_time)
    ]

    for i in range(burn_in_time):
        # choose adjacent pair to propose a swap
        j1, j2 = swap_indices[i]
        acceptance_prob = min(
            1,
            pref_interval[next(iter(current_ranking[j2]))]
            / pref_interval[next(iter(current_ranking[j1]))],
        )

        # if you accept, make the swap
        if rng.random() < acceptance_prob:
            current_ranking[j1], current_ranking[j2] = (
                current_ranking[j2],
                current_ranking[j1],
            )
            accept += 1

    # generate MCMC sample
    for i in range(n_ballots):
        # choose adjacent pair to propose a swap
        j1, j2 = swap_indices[i]
        acceptance_prob = min(
            1,
            pref_interval[next(iter(current_ranking[j2]))]
            / pref_interval[next(iter(current_ranking[j1]))],
        )

        # if you accept, make the swap
        if rng.random() < acceptance_prob:
            current_ranking[j1], current_ranking[j2] = (
                current_ranking[j2],
                current_ranking[j1],
            )
            accept += 1

        ballots[i] = RankBallot(ranking=current_ranking)

    if verbose:
        print(
            f"Acceptance ratio as number accepted / total steps: "
            f"{accept / (n_ballots + burn_in_time):.2}"
        )

    if -1 in ballots:
        raise ValueError("Some element of ballots list is not a ballot.")

    if n_ballots > chain_length:
        raise ValueError("The Markov Chain length cannot be less than the number of ballots.")

    if verbose:
        print(f"The number of ballots before is {len(ballots)}")

    # Subsample evenly ballots
    ballots = [
        ballots[i * chain_length // n_ballots + chain_length // (2 * n_ballots)]
        for i in range(n_ballots)
    ]

    if verbose:
        print(f"The number of ballots after is {len(ballots)}")

    pp = RankProfile(ballots=ballots)
    return pp


def _inner_name_bradley_terry_mcmc(
    config: BlocSlateConfig,
    *,
    verbose: bool = False,
    burn_in_time: int = 0,
    chain_length: int | None = None,
    rng: random.Random | None = None,
) -> dict[str, RankProfile]:
    """
    Sample from the BT distribution using Markov Chain Monte Carlo.

    Args:
        number_of_ballots (int): The number of ballots to generate.

    Kwargs:
        verbose (bool): If True, print the acceptance ratio of the chain. Defaults to False.
        burn_in_time (int): the number of ballots discarded in the beginning of the chain.
            Defaults to 0.
        chain_length (int | None): the length of the Markov Chain. Ballots are subsampled every
            chain_length//n_ballots steps from the chain until the desired number of ballots is
            reached. Defaults to None which sets the chain_length to the number of ballots in
            the config.
        rng (random.Random, optional): Standard library random number generator. Pass a seeded
            instance for reproducible results. Defaults to None for non-deterministic results.

    Returns:
        RankProfile | tuple
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

    pp_by_bloc = {b: RankProfile() for b in bloc_lst}
    pref_interval_by_bloc_dict = config.get_combined_preference_intervals_by_bloc()
    rng = random.Random() if rng is None else rng

    for bloc in bloc_lst:
        n_ballots = ballots_per_bloc[bloc]
        pref_interval = pref_interval_by_bloc_dict[bloc]
        pref_interval_dict = pref_interval.interval
        cands = list(pref_interval.interval.keys())

        seed_ballot = RankBallot(ranking=tuple([frozenset({c}) for c in cands]))
        pp = _bradley_terry_mcmc(
            n_ballots,
            pref_interval_dict,
            seed_ballot,
            verbose=verbose,
            burn_in_time=burn_in_time,
            chain_length=chain_length,
            rng=rng,
        )

        pp_by_bloc[bloc] = pp

    return pp_by_bloc


# =================================================
# ================= API Functions =================
# =================================================


def name_bt_profiles_by_bloc_generator(
    config: BlocSlateConfig,
    *,
    group_ballots=True,
    rng_seed: int | None = None,
) -> dict[str, RankProfile]:
    """
    Generate preference profiles by bloc using the name-BradleyTerry model.

    The probability of sampling the ranking :math:`X>Y>Z` is proportional to
    :math:`P(X>Y)*P(X>Z)*P(Y>Z)`. These individual probabilities are based on the preference
    interval: :math: `P(X>Y) = x/(x+y)`.

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
    _check_name_bt_memory(config)
    config.is_valid(raise_errors=True)
    rng = random.Random(rng_seed)
    pp_by_bloc = _inner_name_bradley_terry(config, rng=rng)
    if group_ballots:
        for bloc in pp_by_bloc:
            pp_by_bloc[bloc] = pp_by_bloc[bloc].group_ballots()

    return pp_by_bloc


def name_bt_profile_generator(
    config: BlocSlateConfig,
    *,
    group_ballots=True,
    rng_seed: int | None = None,
) -> RankProfile:
    """
    Generate a preference profile using the name-BradleyTerry model.

    The probability of sampling the ranking :math:`X>Y>Z` is proportional to
    :math:`P(X>Y)*P(X>Z)*P(Y>Z)`. These individual probabilities are based on the preference
    interval: :math: `P(X>Y) = x/(x+y)`.

    Args:
        config (BlocSlateConfig): Configuration object containing all necessary parameters for
            working with a bloc-slate ballot generator.
        group_ballots (bool): If True, group identical ballots in the returned profile and
            set the weight accordingly. Defaults to True.

    Returns:
        RankProfile: Generated preference profile.
    """
    _check_name_bt_memory(config)
    config.is_valid(raise_errors=True)
    rng = random.Random(rng_seed)
    pp_by_bloc = _inner_name_bradley_terry(config, rng=rng)

    # combine the profiles
    pp = RankProfile()
    for profile in pp_by_bloc.values():
        pp += profile

    if group_ballots:
        pp = pp.group_ballots()
    return pp


def name_bt_profile_generator_using_mcmc(
    config: BlocSlateConfig,
    *,
    group_ballots=True,
    verbose: bool = False,
    burn_in_time: int = 0,
    chain_length: int | None = None,
    rng_seed: int | None = None,
) -> RankProfile:
    """
    Generate a preference profile using MCMC sampling from the name-BradleyTerry model.

    Args:
        config (BlocSlateConfig): Configuration object containing all necessary parameters for
            working with a bloc-slate ballot generator.

    Kwargs:
        group_ballots (bool): If True, group identical ballots in the returned profile and
            set the weight accordingly. Defaults to True.
        verbose (bool): If True, print the acceptance ratio of the chain. Defaults to False.
        burn_in_time (int): the number of ballots discarded in the beginning of the chain.
            Defaults to 0.
        chain_length (int | None): the length of the Markov Chain. Ballots are subsampled every
            chain_length//n_ballots steps from the chain until the desired number of ballots is
            reached. Defaults to None which sets the chain_length to the number of ballots in
            the config.
        rng_seed (int | None): Seed for random number generator. An integer seed produces the
            same output given identical inputs; By default, seed is None which gives
            non-deterministic results.

    Returns:
        RankProfile: Generated preference profile.
    """
    config.is_valid(raise_errors=True)
    rng = random.Random(rng_seed)
    pp_by_bloc = _inner_name_bradley_terry_mcmc(
        config,
        verbose=verbose,
        burn_in_time=burn_in_time,
        chain_length=chain_length,
        rng=rng,
    )
    # combine the profiles
    pp = RankProfile()
    for profile in pp_by_bloc.values():
        pp += profile

    if group_ballots:
        pp = pp.group_ballots()
    return pp


def name_bt_profiles_by_bloc_generator_using_mcmc(
    config: BlocSlateConfig,
    *,
    group_ballots=True,
    verbose: bool = False,
    burn_in_time: int = 0,
    chain_length: int | None = None,
    rng_seed: int | None = None,
) -> dict[str, RankProfile]:
    """
    Generate a preference profile dictionary by bloc using MCMC sampling from the
    name-BradleyTerry model.

    Args:
        config (BlocSlateConfig): Configuration object containing all necessary parameters for
            working with a bloc-slate ballot generator.

    Kwargs:
        group_ballots (bool): If True, group identical ballots in the returned profile and
            set the weight accordingly. Defaults to True.
        verbose (bool): If True, print the acceptance ratio of the chain. Defaults to False.
        burn_in_time (int): the number of ballots discarded in the beginning of the chain.
            Defaults to 0.
        chain_length (int | None): the length of the Markov Chain. Ballots are subsampled every
            chain_length//n_ballots steps from the chain until the desired number of ballots is
            reached. Defaults to None which sets the chain_length to the number of ballots in
            the config.
        rng_seed (int | None): Seed for random number generator. An integer seed produces the
            same output given identical inputs; By default, seed is None which gives
            non-deterministic results.

    Returns:
        dict[str, RankProfile]: Generated preference profiles by bloc.
    """
    config.is_valid(raise_errors=True)
    rng = random.Random(rng_seed)
    pp_by_bloc = _inner_name_bradley_terry_mcmc(
        config,
        verbose=verbose,
        burn_in_time=burn_in_time,
        chain_length=chain_length,
        rng=rng,
    )

    if group_ballots:
        for bloc in pp_by_bloc:
            pp_by_bloc[bloc] = pp_by_bloc[bloc].group_ballots()

    return pp_by_bloc


# ===========================================================================
# ================= Additional Functions Not Yet Integrated =================
# ===========================================================================


def _bradley_terry_mcmc_shortcut(
    n_ballots,
    pref_interval,
    seed_ballot,
    verbose=False,
    burn_in_time=0,
    chain_length=None,
    BURN_IN_TIME=100000,
):
    """
    Sample from BT using MCMC on the shortcut ballot graph

    Args:
        n_ballots (int): the number of ballots to sample
        pref_interval (dict): the preference interval to determine BT distribution
        sub_sample_length (int): how many attempts at swaps to make before saving ballot
        seed_ballot: Ballot, the seed ballot for the Markov chain
        burn_in_time (int): the number of ballots discarded in the beginning of the chain
        chain_length (int | None): the length of the Markov Chain. Ballots are subsampled every
            chain_length//n_ballots steps from the chain until the desired number of ballots is
            reached. Defaults to None which sets the chain_length to the number of ballots in
            the config.
    """
    # NOTE: Most of this has been copied from `_bradley_terry_mcmc`
    # TODO: Abstract the overlapping steps into another helper
    # function, and just pass the indices / transition probability
    # function

    if chain_length is None:
        chain_length = n_ballots

    # check that seed ballot has no ties
    for s in seed_ballot.ranking:
        if len(s) > 1:
            raise ValueError("Seed ballot contains ties")

    ballots = [-1] * n_ballots
    accept = 0
    current_ranking = list(seed_ballot.ranking)
    n_candidates = len(current_ranking)

    if verbose:
        print("MCMC on shortcut")

    burn_in_time = burn_in_time
    if verbose:
        print(f"Burn in time: {burn_in_time}")

    # precompute all the swap indices
    swap_indices = [
        tuple(sorted(random.sample(range(n_candidates), 2)))
        for _ in range(n_ballots + burn_in_time)
    ]

    for i in range(burn_in_time):
        # choose adjacent pair to propose a swap
        j1, j2 = swap_indices[i]
        j1_rank = j1 + 1
        j2_rank = j2 + 1
        if j2_rank <= j1_rank:
            raise Exception("MCMC on Shortcut: invalid ranks found")

        acceptance_prob = min(
            1,
            (pref_interval[next(iter(current_ranking[j2]))] ** (j2_rank - j1_rank))
            / (pref_interval[next(iter(current_ranking[j1]))] ** (j2_rank - j1_rank)),
        )

        # if you accept, make the swap
        if random.random() < acceptance_prob:
            current_ranking[j1], current_ranking[j2] = (
                current_ranking[j2],
                current_ranking[j1],
            )
            accept += 1

    # generate MCMC sample
    for i in range(n_ballots):
        # choose adjacent pair to propose a swap
        j1, j2 = swap_indices[i]
        j1_rank = j1 + 1
        j2_rank = j2 + 1
        if j2_rank <= j1_rank:
            raise Exception("MCMC on Shortcut: invalid ranks found")

        acceptance_prob = min(
            1,
            (pref_interval[next(iter(current_ranking[j2]))] ** (j2_rank - j1_rank))
            / pref_interval[next(iter(current_ranking[j1]))] ** (j2_rank - j1_rank),
        )

        # if you accept, make the swap
        if random.random() < acceptance_prob:
            current_ranking[j1], current_ranking[j2] = (
                current_ranking[j2],
                current_ranking[j1],
            )
            accept += 1

        ballots[i] = RankBallot(ranking=current_ranking)

    if verbose:
        print(
            f"Acceptance ratio as number accepted / total steps: "
            f"{accept / (n_ballots + BURN_IN_TIME):.2}"
        )

    if -1 in ballots:
        raise ValueError("Some element of ballots list is not a ballot.")

    if n_ballots > chain_length:
        raise ValueError("The Markov Chain length cannot be less than the number of ballots.")

    if verbose:
        print(f"The number of ballots before is {len(ballots)}")

    # Subsample evenly ballots
    ballots = [
        ballots[i * chain_length // n_ballots + chain_length // (2 * n_ballots)]
        for i in range(n_ballots)
    ]

    if verbose:
        print(f"The number of ballots after is {len(ballots)}")

    pp = RankProfile(ballots=ballots)
    return pp


# ————————————————————————————————————————————————————————————————————————————————————————————————
# ————————————————————————————————————————————————————————————————————————————————————————————————
# ————————————————————————————————————————————————————————————————————————————————————————————————
# ————————————————————————————————————————————————————————————————————————————————————————————————


_DP_NAME_BT_BATCH_SIZE = 65_536


def _check_dp_name_bt_memory(n_candidates: int, n_voters: int, *, group_ballots: bool) -> None:
    """Check the estimated peak allocation against currently available RAM.

    Estimate CDF preparation, batched sampling, and profile construction separately, then
    allow 50% headroom above the largest peak for allocator and library overhead. Blocs share
    the output array and prepare their CDFs sequentially, so only one table is counted.
    Grouping assumes every ranking is distinct. This check cannot reserve memory against
    concurrent allocations by other processes.

    Args:
        n_candidates (int): Number of candidates in each complete ranking.
        n_voters (int): Total rows to generate across all blocs.
        group_ballots (bool): Whether to include grouping buffers and the grouped profile.

    Raises:
        ValueError: Subset masks or array dimensions exceed the supported integer widths.
        MemoryError: The estimated peak with headroom exceeds available RAM.
    """
    max_candidates = min(32, np.iinfo(np.intp).bits - 2)
    if n_candidates > max_candidates:
        raise ValueError(
            f"DP name-BradleyTerry supports at most {max_candidates} candidates with its "
            "uint32 subset masks and platform array-index width."
        )

    float_bytes = np.dtype(np.float64).itemsize
    index_bytes = np.dtype(int).itemsize
    n_subsets = 1 << n_candidates
    largest_subset_count = math.comb(n_candidates, n_candidates // 2)
    ranking_bytes = n_voters * n_candidates * index_bytes
    cdf_bytes = n_subsets * n_candidates * float_bytes

    # Log totals, uint32 masks, uint8 sizes, and the mask selecting a subset size.
    subset_vector_bytes = n_subsets * (float_bytes + 4 + 1 + 1)
    # Consecutive working matrices can coexist, along with indexed vectors and arithmetic results.
    preparation_work_bytes = largest_subset_count * (
        2 * n_candidates * float_bytes + 8 * index_bytes
    )
    preparation_peak = ranking_bytes + cdf_bytes + subset_vector_bytes + preparation_work_bytes

    buffer_rows = min(_DP_NAME_BT_BATCH_SIZE, n_voters)
    sampling_buffer_bytes = buffer_rows * (
        n_candidates * (float_bytes + 1) + 2 * index_bytes + 2 * float_bytes
    )
    sampling_peak = ranking_bytes + cdf_bytes + n_voters * index_bytes + sampling_buffer_bytes

    # Include owned rankings, candidate-discovery copies, weights, and validation temporaries.
    construction_peak = 4 * ranking_bytes + n_voters * (4 * index_bytes + 4 * float_bytes)
    if group_ballots:
        # Sorting and row selection can retain both profiles plus row keys and index arrays.
        construction_peak = 6 * ranking_bytes + n_voters * (12 * index_bytes + 4 * float_bytes)

    estimated_peak = max(preparation_peak, sampling_peak, construction_peak)
    required_bytes = (3 * estimated_peak + 1) // 2
    available_gib = system_memory()["available_gib"]
    if required_bytes > available_gib * 2**30:
        raise MemoryError(
            f"Not enough memory for DP name-BradleyTerry with {n_candidates} candidates and "
            f"{n_voters} voters. Estimated peak is {estimated_peak / 2**30:.3f} GiB; "
            f"{required_bytes / 2**30:.3f} GiB is required including 50% headroom, "
            f"but only {available_gib:.3f} GiB is available. "
            "Reduce the candidate or voter count, or use the MCMC generator."
        )


def _make_candidate_sets(candidates: list[Candidate]) -> tuple[frozenset[Candidate], ...]:
    """
    Given a list of candidates, returns a tuple of frozensets representing all possible
    subsets of the candidates, including the empty set and the full set.
    """

    return (EMPTY_VOTER_SET, END_OF_RANKING_SET, *[frozenset({s}) for s in candidates])


def _build_cdf_matrix(candidates: tuple[Candidate, ...], candidate_supports: NDArray) -> NDArray:
    n_candidates = len(candidates)

    # For a 3-candidate example, the subsets are:
    # [{}, {c1}, {c2}, {c1, c2}, {c3}, {c1, c3}, {c2, c3}, {c1, c2, c3}]
    n_dp_subsets = 2**n_candidates

    # Each candidate gets their own bit in the bitmask
    candidate_bit_assignments = [2**i for i in range(n_candidates)]

    log_supports = np.log(candidate_supports)

    log_total_suffix_weight = np.full(n_dp_subsets, -np.inf, dtype=np.float64)

    # If there are no candidates remaining to select, then the probability of selecting the empty
    # set is 1.
    log_total_suffix_weight[0] = 0

    cdf_matrix = np.zeros((n_dp_subsets, n_candidates), dtype=np.float64)
    states = np.arange(1 << n_candidates, dtype=np.uint32)
    subset_sizes = np.bitwise_count(states)

    for n_remaining_candidates in range(1, n_candidates + 1):
        states_with_n_remaining_candidates = states[subset_sizes == n_remaining_candidates]
        log_next_cand_weights = np.full(
            (len(states_with_n_remaining_candidates), n_candidates),
            -np.inf,
            dtype=np.float64,
        )

        for cand_id, cand_bit in enumerate(candidate_bit_assignments):
            cand_is_present = (states_with_n_remaining_candidates & cand_bit) != 0

            successor_remaining_cand_set_masks = (
                states_with_n_remaining_candidates[cand_is_present] - cand_bit
            )

            log_next_cand_weights[cand_is_present, cand_id] = (
                n_remaining_candidates - 1
            ) * log_supports[cand_id] + log_total_suffix_weight[successor_remaining_cand_set_masks]

        log_subset_totals = np.empty(len(states_with_n_remaining_candidates), dtype=np.float64)
        np.logaddexp.reduce(
            log_next_cand_weights,
            axis=1,
            out=log_subset_totals,
        )
        log_total_suffix_weight[states_with_n_remaining_candidates] = log_subset_totals

        # Reuse the working array for probabilities, then cumulative probabilities.
        subset_cdfs = log_next_cand_weights
        subset_cdfs -= log_subset_totals[:, None]

        np.exp(subset_cdfs, out=subset_cdfs)
        np.cumsum(subset_cdfs, axis=1, out=subset_cdfs)

        subset_cdfs /= subset_cdfs[:, -1:]

        cdf_matrix[states_with_n_remaining_candidates] = subset_cdfs

    return cdf_matrix


def _sample_rankings_from_cdf(
    cdf_matrix: NDArray[np.float64],
    rankings: NDArray[np.intp],
    rng: np.random.Generator,
) -> None:
    """Fill encoded rankings using bounded sampling buffers.

    Args:
        cdf_matrix (NDArray[np.float64]): Conditional CDFs indexed by remaining-candidate
            bitmasks, with one column per candidate. The empty-subset row is unused.
        rankings (NDArray[np.intp]): Writable output with one row per voter and one column
            per ranking position. Candidate IDs follow CDF column order.
        rng (np.random.Generator): Generator advanced in position order, then voter order.
    """
    n_voters, n_candidates = rankings.shape
    batch_size = _DP_NAME_BT_BATCH_SIZE
    buffer_rows = min(batch_size, n_voters)
    candidate_bits = np.left_shift(1, np.arange(n_candidates, dtype=int))
    remaining_subset_masks = np.full(n_voters, len(cdf_matrix) - 1, dtype=int)

    voter_cdf_rows = np.empty((buffer_rows, n_candidates), dtype=np.float64)
    passed_thresholds = np.empty((buffer_rows, n_candidates), dtype=bool)
    chosen_candidate_ids = np.empty(buffer_rows, dtype=int)

    # Keep positions outside batches so batching preserves the seeded rankings.
    for position in range(n_candidates):
        for start in range(0, n_voters, batch_size):
            stop = min(start + batch_size, n_voters)
            n_batch_voters = stop - start
            batch_subset_masks = remaining_subset_masks[start:stop]
            np.take(
                cdf_matrix,
                batch_subset_masks,
                axis=0,
                out=voter_cdf_rows[:n_batch_voters],
                mode="clip",
            )

            draws = rng.random(n_batch_voters)
            np.greater_equal(
                draws[:, None],
                voter_cdf_rows[:n_batch_voters],
                out=passed_thresholds[:n_batch_voters],
            )
            np.sum(
                passed_thresholds[:n_batch_voters],
                axis=1,
                dtype=int,
                out=chosen_candidate_ids[:n_batch_voters],
            )

            rankings[start:stop, position] = chosen_candidate_ids[:n_batch_voters]
            batch_subset_masks -= candidate_bits[chosen_candidate_ids[:n_batch_voters]]


def dp_name_bt_profile_generator(
    config: BlocSlateConfig,
    *,
    group_ballots: bool = True,
    rng_seed: int | None = None,
) -> RankProfile:
    """Generate name-BradleyTerry rankings using subset dynamic programming.

    Prepare one conditional CDF per populated bloc and sample voters in bounded batches.
    Candidate supports must be strictly positive. Before allocating the output or CDF, check
    that estimated peak memory plus 50% headroom fits in currently available RAM.

    Args:
        config (BlocSlateConfig): Bloc proportions, candidate supports, and voter count.
        group_ballots (bool, optional): Combine identical rankings and sum their weights.
            Defaults to True.
        rng_seed (int | None, optional): Seed for NumPy's random generator.
            Defaults to None.

    Returns:
        RankProfile: Complete rankings with the declared candidate order and total voter weight.

    Raises:
        ValueError: A populated bloc has nonfinite or nonpositive candidate support, or the
            candidate count exceeds the subset-mask or array-index width.
        MemoryError: Estimated memory requirements exceed currently available RAM.
    """
    config.is_valid(raise_errors=True)
    candidates = tuple(config.candidates)
    blocs = tuple(config.blocs)
    n_candidates = len(candidates)
    _check_dp_name_bt_memory(n_candidates, config.n_voters, group_ballots=group_ballots)
    rng = np.random.default_rng(rng_seed)

    bloc_counts = apportion.compute(
        "huntington",
        [config.bloc_proportions[bloc] for bloc in blocs],
        config.n_voters,
    )
    if isinstance(bloc_counts, int):
        bloc_counts = [bloc_counts]

    preference_intervals = config.get_combined_preference_intervals_by_bloc()

    ranking_matrix = np.empty((config.n_voters, n_candidates), dtype=int)
    row_start = 0

    for bloc, n_bloc_voters in zip(blocs, bloc_counts, strict=True):
        if n_bloc_voters == 0:
            continue

        supports = np.array(
            [preference_intervals[bloc].interval[candidate] for candidate in candidates],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(supports) & (supports > 0)):
            raise ValueError(f"Bloc {bloc!r} requires finite, positive candidate supports.")

        cdf_matrix = _build_cdf_matrix(candidates, supports)
        bloc_rankings = ranking_matrix[row_start : row_start + n_bloc_voters]

        _sample_rankings_from_cdf(cdf_matrix, bloc_rankings, rng)

        row_start += n_bloc_voters
        del cdf_matrix  # Release this bloc's table before preparing the next one.

    voter_weight_data = ProfileVoterWeightData(
        weights=np.ones(config.n_voters, dtype=np.float64),
        voter_sets=None,
        weight_kind=WeightKind.FLOAT,
    )
    matrix = RankMatrix(
        voter_weight_data=voter_weight_data,
        rankings=ranking_matrix,
        cand_sets=_make_candidate_sets(list(candidates)),
        ranking_columns=tuple(f"{RANKING_COLUMN_PREFIX}{i + 1}" for i in range(n_candidates)),
        candidates=candidates,
    )

    profile = RankProfile._from_matrix(
        matrix,
        max_ranking_length=n_candidates,
    )
    return profile.group_ballots() if group_ballots else profile
