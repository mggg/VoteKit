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
import numpy as np
import random
from typing import Mapping, Optional
import apportionment.methods as apportion
import math
import pandas as pd
import sys

from votekit.ballot import RankBallot
from votekit.pref_profile import RankProfile
from votekit.ballot_generator.bloc_slate_generator.model import BlocSlateConfig
from votekit.ballot_generator.utils import system_memory

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


def _bradley_terry_pdf(dct: Mapping[str, float]) -> dict[tuple[str, ...], float]:
    """
    Given a dictionary of candidates and their support, returns the probability density function
    over all possible rankings.

    Args:
        dct (Mapping[str, float]): a mapping from candidate to their support

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


# ===========================================================
# ================= Interior Work Functions =================
# ===========================================================


def _inner_name_bradley_terry(config: BlocSlateConfig) -> dict[str, RankProfile]:
    """
    Sample from the BT distribution using direct sampling.

    This is an interior helper function that does the bulk of the work for generating
    name-BradleyTerry profiles. It is not recommended to use this function directly,
    as it does not perform any memory checks or grouping of ballots.

    Args:
        config (BlocSlateConfig): Configuration object containing all necessary parameters for
            working with a bloc-slate ballot generator.

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
            raise TypeError(
                f"Unexpected type from apportionment got {type(bloc_counts)}"
            )

        bloc_counts = [bloc_counts]

    ballots_per_bloc = {bloc: bloc_counts[i] for i, bloc in enumerate(bloc_lst)}

    pp_by_bloc = {b: RankProfile() for b in bloc_lst}

    pref_interval_by_bloc_dict = config.get_combined_preference_intervals_by_bloc()

    for bloc in config.bloc_proportions.keys():
        n_ballots = ballots_per_bloc[bloc]

        # Directly initialize the list using good memory trick
        ballot_pool = np.full((n_ballots, n_candidates), frozenset("~"), dtype=object)
        zero_cands = pref_interval_by_bloc_dict[bloc].zero_cands
        single_bloc_pdf_dict = _bradley_terry_pdf(
            pref_interval_by_bloc_dict[bloc].interval
        )

        # Directly use the keys and values from the dictionary for sampling
        rankings, probs = zip(*single_bloc_pdf_dict.items())

        # The return of this will be a numpy array, so we don't need to make it into a list
        sampled_indices = np.array(
            np.random.choice(
                a=len(rankings),
                size=n_ballots,
                p=probs,
            ),
            ndmin=1,
        )

        for j, index in enumerate(sampled_indices):
            ranking = [frozenset({cand}) for cand in rankings[index]]

            # Add any zero candidates as ties only if they exist
            if zero_cands:
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
        pp_by_bloc[bloc] = pp

    return pp_by_bloc


# TODO:
# - Improve the subsampling scheme here (memory + speed)
# - Construct ballots as tuples
# - Other speed improvements
def _bradley_terry_mcmc(
    n_ballots: int,
    pref_interval: Mapping[str, float],
    seed_ballot: RankBallot,
    zero_cands: Optional[frozenset[str]] = None,
    verbose: bool = False,
    burn_in_time: int = 0,
    chain_length: Optional[int] = None,
):
    """
    Sample from BT distribution for a given preference interval using MCMC. Defaults
    to continuous sampling and no burn-in time.

    Args:
        n_ballots (int): the number of ballots to sample
        pref_interval (Mapping[str, float]): the preference interval to determine BT distribution
        seed_ballot (RankBallot):  the seed ballot for the Markov chain
        zero_cands (frozenset[str], optional): candidates with zero support to be added as ties
        verbose (bool): If True, print the acceptance ratio of the chain. Defaults to False.
        burn_in_time (int): the number of ballots discarded in the beginning of the chain
        chain_length (Optional[int]): the length of the Markov Chain. Ballots are subsampled every
            chain_length//n_ballots steps from the chain until the desired number of ballots is
            reached. Defaults to None which sets the chain_length to the number of ballots in
            the config.
    """
    if zero_cands is None:
        zero_cands = frozenset()

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
    swap_indices = [
        (j1, j1 + 1)
        for j1 in random.choices(range(n_candidates - 1), k=n_ballots + burn_in_time)
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
        acceptance_prob = min(
            1,
            pref_interval[next(iter(current_ranking[j2]))]
            / pref_interval[next(iter(current_ranking[j1]))],
        )

        # if you accept, make the swap
        if random.random() < acceptance_prob:
            current_ranking[j1], current_ranking[j2] = (
                current_ranking[j2],
                current_ranking[j1],
            )
            accept += 1

        if len(zero_cands) > 0:
            ballots[i] = RankBallot(ranking=current_ranking + [zero_cands])
        else:
            ballots[i] = RankBallot(ranking=current_ranking)

    if verbose:
        print(
            f"Acceptance ratio as number accepted / total steps: "
            f"{accept / (n_ballots + burn_in_time):.2}"
        )

    if -1 in ballots:
        raise ValueError("Some element of ballots list is not a ballot.")

    if n_ballots > chain_length:
        raise ValueError(
            "The Markov Chain length cannot be less than the number of ballots."
        )

    if verbose:
        print(f"The number of ballots before is {len(ballots)}")

    # Subsample evenly ballots
    ballots = [
        ballots[i * chain_length // n_ballots + chain_length // (2 * n_ballots)]
        for i in range(n_ballots)
    ]

    if verbose:
        print(f"The number of ballots after is {len(ballots)}")

    pp = RankProfile(ballots=ballots)  # type: ignore
    return pp


def _inner_name_bradley_terry_mcmc(
    config: BlocSlateConfig,
    *,
    verbose: bool = False,
    burn_in_time: int = 0,
    chain_length: Optional[int] = None,
) -> dict[str, RankProfile]:
    """
    Sample from the BT distribution using Markov Chain Monte Carlo.

    Args:
        number_of_ballots (int): The number of ballots to generate.

    Kwargs:
        verbose (bool): If True, print the acceptance ratio of the chain. Defaults to False.
        burn_in_time (int): the number of ballots discarded in the beginning of the chain.
            Defaults to 0.
        chain_length (Optional[int]): the length of the Markov Chain. Ballots are subsampled every
            chain_length//n_ballots steps from the chain until the desired number of ballots is
            reached. Defaults to None which sets the chain_length to the number of ballots in
            the config.

    Returns:
        Union[RankProfile, Tuple]
    """
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

    pp_by_bloc = {b: RankProfile() for b in bloc_lst}
    pref_interval_by_bloc_dict = config.get_combined_preference_intervals_by_bloc()

    for bloc in bloc_lst:
        n_ballots = ballots_per_bloc[bloc]
        pref_interval = pref_interval_by_bloc_dict[bloc]
        pref_interval_dict = pref_interval.interval
        non_zero_cands = pref_interval.non_zero_cands
        zero_cands = pref_interval.zero_cands

        seed_ballot = RankBallot(
            ranking=tuple([frozenset({c}) for c in non_zero_cands])
        )
        pp = _bradley_terry_mcmc(
            n_ballots,
            pref_interval_dict,
            seed_ballot,
            zero_cands=zero_cands,
            verbose=verbose,
            burn_in_time=burn_in_time,
            chain_length=chain_length,
        )

        pp_by_bloc[bloc] = pp

    return pp_by_bloc


# =================================================
# ================= API Functions =================
# =================================================


def name_bt_profiles_by_bloc_generator(
    config: BlocSlateConfig, *, group_ballots=True
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

    Returns:
        dict[str, RankProfile]: Generated preference profiles by bloc.
    """
    _check_name_bt_memory(config)
    config.is_valid(raise_errors=True)
    pp_by_bloc = _inner_name_bradley_terry(config)
    if group_ballots:
        for bloc in pp_by_bloc:
            pp_by_bloc[bloc] = pp_by_bloc[bloc].group_ballots()

    return pp_by_bloc


def name_bt_profile_generator(
    config: BlocSlateConfig, *, group_ballots=True
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
    pp_by_bloc = _inner_name_bradley_terry(config)

    # combine the profiles
    pp = RankProfile()
    for profile in pp_by_bloc.values():
        pp += profile

    if group_ballots:
        pp.group_ballots()
    return pp


def name_bt_profile_generator_using_mcmc(
    config: BlocSlateConfig,
    *,
    group_ballots=True,
    verbose: bool = False,
    burn_in_time: int = 0,
    chain_length: Optional[int] = None,
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
        chain_length (Optional[int]): the length of the Markov Chain. Ballots are subsampled every
            chain_length//n_ballots steps from the chain until the desired number of ballots is
            reached. Defaults to None which sets the chain_length to the number of ballots in
            the config.

    Returns:
        RankProfile: Generated preference profile.
    """
    config.is_valid(raise_errors=True)
    pp_by_bloc = _inner_name_bradley_terry_mcmc(
        config, verbose=verbose, burn_in_time=burn_in_time, chain_length=chain_length
    )
    # combine the profiles
    pp = RankProfile()
    for profile in pp_by_bloc.values():
        pp += profile

    if group_ballots:
        pp.group_ballots()
    return pp


def name_bt_profiles_by_bloc_generator_using_mcmc(
    config: BlocSlateConfig,
    *,
    group_ballots=True,
    verbose: bool = False,
    burn_in_time: int = 0,
    chain_length: Optional[int] = None,
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
        chain_length (Optional[int]): the length of the Markov Chain. Ballots are subsampled every
            chain_length//n_ballots steps from the chain until the desired number of ballots is
            reached. Defaults to None which sets the chain_length to the number of ballots in
            the config.

    Returns:
        dict[str, RankProfile]: Generated preference profiles by bloc.
    """
    config.is_valid(raise_errors=True)
    pp_by_bloc = _inner_name_bradley_terry_mcmc(
        config, verbose=verbose, burn_in_time=burn_in_time, chain_length=chain_length
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
    zero_cands={},
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
        chain_length (Optional[int]): the length of the Markov Chain. Ballots are subsampled every
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

        if len(zero_cands) > 0:
            ballots[i] = RankBallot(ranking=current_ranking + [zero_cands])
        else:
            ballots[i] = RankBallot(ranking=current_ranking)

    if verbose:
        print(
            f"Acceptance ratio as number accepted / total steps: "
            f"{accept / (n_ballots + BURN_IN_TIME):.2}"
        )

    if -1 in ballots:
        raise ValueError("Some element of ballots list is not a ballot.")

    if n_ballots > chain_length:
        raise ValueError(
            "The Markov Chain length cannot be less than the number of ballots."
        )

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
