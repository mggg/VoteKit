"""Generate name-BradleyTerry profiles using Markov Chain Monte Carlo."""

from __future__ import annotations

import random
from collections.abc import Mapping

import apportionment.methods as apportion

from votekit.ballot import RankBallot
from votekit.ballot_generator.bloc_slate_generator.config import BlocSlateConfig
from votekit.pref_profile import RankProfile
from votekit.types import Candidate

# ===========================================================
# ================= Interior Work Functions =================
# ===========================================================


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
