import itertools as it
import numpy as np
from numpy.typing import NDArray
import pandas as pd

# import random
# import warnings
from typing import Sequence, cast
import apportionment.methods as apportion

from votekit.pref_profile import RankProfile
from votekit.ballot_generator.bloc_slate_generator.model import BlocSlateConfig

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
    n_candidates = len(non_zero_candidate_set)
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

    pmf = {
        tuple(ballot_type): _slate_bt_numerator_computation_single_bloc(
            tuple(ballot_type), slate_cohesion_dict_for_bloc
        )
        for ballot_type in it.permutations(slate_list, n_candidates)
    }

    summ = sum(pmf.values())
    return {b: v / summ for b, v in pmf.items()}


def _sample_ballot_types_deterministic(
    config: BlocSlateConfig,
    bloc: str,
    n_ballots: int,
    non_zero_candidate_set: set[str],
) -> list[tuple[str]]:
    """
    Used to generate bloc orderings for deliberative.

    Returns a list of lists, where each sublist contains the bloc names in order they appear
    on the ballot.
    """
    pdf = _compute_ballot_type_dist(config, bloc, non_zero_candidate_set)
    b_types = list(pdf.keys())
    probs = list(pdf.values())

    sampled_indices = np.random.choice(len(b_types), size=n_ballots, p=probs)

    return [b_types[i] for i in sampled_indices]


# FIX: Still needs update for new slate-bloc disambiguation
# def _sample_ballot_types_MCMC(
#     config: BlocSlateConfig, bloc: str, n_ballots: int, verbose: bool = False
# ):
#     """
#     Generate ballot types using MCMC that has desired stationary distribution.
#     """
#
#     seed_ballot_type = [
#         b
#         for b in config.blocs
#         for _ in range(len(self.pref_intervals_by_bloc[bloc][b].non_zero_cands))
#     ]
#
#     ballots = [[-1]] * n_ballots
#     accept = 0
#     current_ranking = seed_ballot_type
#
#     cohesion = self.cohesion_parameters[bloc][bloc]
#
#     # presample swap indices
#     swap_indices = [
#         (j1, j1 + 1)
#         for j1 in np.random.choice(len(seed_ballot_type) - 1, size=n_ballots)
#     ]
#
#     odds = (1 - cohesion) / cohesion
#     # generate MCMC sample
#     for i in range(n_ballots):
#         # choose adjacent pair to propose a swap
#         j1, j2 = swap_indices[i]
#
#         # if swap reduces number of voters bloc above opposing bloc
#         if current_ranking[j1] != current_ranking[j2] and current_ranking[j1] == bloc:
#             acceptance_prob = odds
#
#         # if swap increases number of voters bloc above opposing or swaps two of same bloc
#         else:
#             acceptance_prob = 1
#
#         # if you accept, make the swap
#         if random.random() < acceptance_prob:
#             current_ranking[j1], current_ranking[j2] = (
#                 current_ranking[j2],
#                 current_ranking[j1],
#             )
#             accept += 1
#
#         ballots[i] = current_ranking.copy()
#
#     if verbose:
#         print(
#             f"Acceptance ratio as number accepted / total steps: {accept / n_ballots:.2}"
#         )
#
#     if -1 in ballots:
#         raise ValueError("Some element of ballots list is not a ballot.")
#
#     return ballots


# ===========================================================
# ================= Interior Work Functions =================
# ===========================================================


def _inner_slate_bradley_terry(
    config: BlocSlateConfig,
    deterministic: bool = True,
) -> dict[str, RankProfile]:
    """
    Args:
        number_of_ballots (int): The number of ballots to generate.
        by_bloc (bool): True if you want the generated profiles returned as a tuple
            ``(pp_by_bloc, pp)``, where ``pp_by_bloc`` is a dictionary with keys = bloc strings
            and values = ``RankProfile`` and ``pp`` is the aggregated profile. False if
            you only want the aggregated profile. Defaults to False.
        deterministic (bool): True if you want to use precise pdf, False to use MCMC sampling.
            Defaults to True.

    Returns:
        dict[str, RankProfile]
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

    pref_profile_by_bloc = {b: RankProfile() for b in bloc_lst}
    candidates = config.candidates

    for i, bloc in enumerate(config.blocs):
        # number of voters in this bloc
        n_ballots = ballots_per_bloc[bloc]
        ballot_pool = np.full((n_ballots, n_candidates), frozenset("~"))
        pref_intervals = config.get_preference_intervals_for_bloc(bloc)
        zero_cands = set(it.chain(*[pi.zero_cands for pi in pref_intervals.values()]))
        non_zero_cands_set = set(candidates) - zero_cands

        if deterministic and len(candidates) >= 12:
            raise UserWarning(
                "Deterministic sampling is only supported for 11 or fewer candidates.\n\
                Please set deterministic = False."
            )

        elif deterministic:
            ballot_types = _sample_ballot_types_deterministic(
                config=config,
                bloc=bloc,
                n_ballots=n_ballots,
                non_zero_candidate_set=non_zero_cands_set,
            )
        else:
            raise NotImplementedError(
                "MCMC sampling not yet implemented for slate-BradleyTerry."
            )

            # ballot_types = _sample_ballot_types_MCMC(
            #     config = config, bloc=bloc, n_ballots=n_ballots
            # )

        for j, bt in enumerate(ballot_types):
            cand_ordering_by_bloc = {}

            for b in config.blocs:
                # create a pref interval dict of only this blocs candidates
                bloc_cand_pref_interval = pref_intervals[b].interval
                cands = pref_intervals[b].non_zero_cands

                # if there are no non-zero candidates, skip this bloc
                if len(cands) == 0:
                    continue

                distribution = [bloc_cand_pref_interval[c] for c in cands]

                # sample by Plackett-Luce within the bloc
                cand_ordering = np.random.choice(
                    a=list(cands), size=len(cands), p=distribution, replace=False
                )

                cand_ordering_by_bloc[b] = list(cand_ordering)

            ranking = [frozenset({"~"})] * len(bt)
            for i, b in enumerate(bt):
                # append the current first candidate, then remove them from the ordering
                ranking[i] = frozenset({cand_ordering_by_bloc[b][0]})
                cand_ordering_by_bloc[b].pop(0)

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

    MORE INFO HERE

    Args:
        config (BlocSlateConfig): Configuration object containing all necessary parameters.
        group_ballots (bool): If True, group identical ballots in the returned profile and
            set the weight accordingly. Defaults to True.

    Returns:
        RankProfile: Generated preference profile.
    """
    # FIX: Check the memory
    # _check_slate_bt_memory(config)

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
    Generate a preference profile using the name-BradleyTerry model.

    MORE INFO HERE

    Args:
        config (BlocSlateConfig): Configuration object containing all necessary parameters.
        group_ballots (bool): If True, group identical ballots in the returned profile and
            set the weight accordingly. Defaults to True.

    Returns:
        dict[str, RankProfile]: Generated preference profiles by bloc.
    """
    # FIX: Check the memory
    # _check_slate_bt_memory(config)

    config.is_valid(raise_errors=True)
    pp_by_bloc = _inner_slate_bradley_terry(config)
    if group_ballots:
        for bloc in pp_by_bloc:
            pp_by_bloc[bloc] = pp_by_bloc[bloc].group_ballots()

    return pp_by_bloc
