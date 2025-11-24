"""
Generate ranked preference profiles using the slate-Plackett-Luce model.

The main API functions in this module are:

- `slate_pl_profile_generator`: Generates a single preference profile using the slate-Plackett-Luce
    model.
- `slate_pl_profiles_by_bloc_generator`: Generates preference profiles by bloc using the
    slate-Plackett-Luce model.
"""

import apportionment.methods as apportion

from votekit.pref_profile import RankProfile
from votekit.ballot_generator.bloc_slate_generator.model import BlocSlateConfig
from votekit.ballot_generator.bloc_slate_generator.slate_utils import (
    _convert_slate_ballots_to_profile,
    _append_zero_slate_symbols,
)
import numpy as np


# ====================================================
# ================= Helper Functions =================
# ====================================================


def _sample_pl_slate_ballots(
    config: BlocSlateConfig,
    num_ballots: int,
    bloc: str,
    non_zero_slate_set: set[str],
) -> list[tuple[str, ...]]:
    """
    Returns a list of slate ballots; each ballot is a list of slate names (strings)
    in the order they appear on that ballot. Only generates ballots for slates
    that have non-zero cohesion for the given bloc.


    Args:
        config (BlocSlateConfig): Configuration object containing all necessary parameters for
            working with a bloc-slate ballot generator.
        num_ballots (int): The number of ballots to generate.
        bloc (str): The name of the bloc for which to generate slate ballots.
        non_zero_slate_set (set[str]): Set of slates with non-zero cohesion for the given bloc.

    Returns:
        list[tuple[str, ...]]: A list of tuples, where each tuple contains the bloc names in the
            order they appear on the ballot.
    """
    num_candidates_per_slate = {
        s: len(config.slate_to_candidates[s]) for s in non_zero_slate_set
    }

    num_candidates = sum(num_candidates_per_slate.values())

    ballots: list[tuple[str, ...]] = [tuple() for _ in range(num_ballots)]

    rand_unif_seqs = np.random.uniform(size=(num_ballots, num_candidates))

    def which_bin(dist_bins: list[float], flip: float) -> int:
        for i, left in enumerate(dist_bins[:-1]):
            if left < flip <= dist_bins[i + 1]:
                return i
        return len(dist_bins) - 2

    slates = list(non_zero_slate_set)
    cohesion_values_og = [
        float(config.cohesion_df.loc[bloc][slate]) for slate in slates
    ]

    for i, rand_unif_seq in enumerate(rand_unif_seqs):
        cohesion_values = np.array(cohesion_values_og)
        distribution_bins: list[float] = [0.0] + np.cumsum(cohesion_values).tolist()
        ballot_type: list[str] = [""] * num_candidates
        slate_count = {s: 0 for s in slates}

        for j, rand_float in enumerate(rand_unif_seq):
            slate_index = which_bin(distribution_bins, float(rand_float))
            slate_type = slates[slate_index]
            ballot_type[j] = slate_type
            slate_count[slate_type] += 1

            if (
                j < num_candidates - 1
                and slate_count[slate_type] == num_candidates_per_slate[slate_type]
            ):
                cohesion_values[slate_index] = 0
                cohesion_values = cohesion_values / cohesion_values.sum()
                distribution_bins = [0.0] + np.cumsum(cohesion_values).tolist()

        ballots[i] = tuple(ballot_type)

    return ballots


# ===========================================================
# ================= Interior Work Functions =================
# ===========================================================


def _inner_slate_plackett_luce(
    config: BlocSlateConfig,
) -> dict[str, RankProfile]:
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

    Returns:
        dict[str, RankProfile]: Dictionary whose keys are bloc strings and values are
            ``RankProfile`` objects.
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

    pref_profile_by_bloc = {b: RankProfile() for b in bloc_lst}

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
        )

        if len(zero_slate_set) != 0:
            slate_ballots = _append_zero_slate_symbols(
                slate_ballots, zero_slate_set, n_ballots, config
            )
        pref_profile_by_bloc[bloc] = _convert_slate_ballots_to_profile(
            config, bloc, slate_ballots
        )

    return pref_profile_by_bloc


# =================================================
# ================= API Functions =================
# =================================================


def slate_pl_profile_generator(
    config: BlocSlateConfig,
    *,
    group_ballots: bool = True,
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

    Returns:
        RankProfile: Merged ``RankProfile`` object generated by the model.
    """
    config.is_valid(raise_errors=True)

    pp_by_bloc = _inner_slate_plackett_luce(config)

    pp = RankProfile(ballots=tuple())
    for profile in pp_by_bloc.values():
        pp += profile

    if group_ballots:
        pp = pp.group_ballots()

    return pp


def slate_pl_profiles_by_bloc_generator(
    config: BlocSlateConfig,
    *,
    group_ballots: bool = True,
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

    Returns:
        dict[str, RankProfile]: Dictionary whose keys are bloc strings and values are
            ``RankProfile`` objects generated by the model for each bloc.
    """
    config.is_valid(raise_errors=True)

    pp_by_bloc = _inner_slate_plackett_luce(config)

    if group_ballots:
        for bloc, profile in pp_by_bloc.items():
            pp_by_bloc[bloc] = profile.group_ballots()

    return pp_by_bloc
