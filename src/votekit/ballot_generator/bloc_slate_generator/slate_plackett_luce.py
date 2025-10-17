"""
Generate ranked preference profiles using the slate-Plackett-Luce model.

The main API functions in this module are:

- `slate_pl_profile_generator`: Generates a single preference profile using the slate-Plackett-Luce
    model.
- `slate_pl_profiles_by_bloc_generator`: Generates preference profiles by bloc using the
    slate-Plackett-Luce model.
"""

from re import L
import apportionment.methods as apportion

from votekit.pref_profile import RankProfile
from votekit.ballot_generator.bloc_slate_generator.model import BlocSlateConfig
from votekit.ballot_generator.bloc_slate_generator.slate_utils import (
    _convert_slate_ballots_to_profile,
)
import itertools as it
import random
import numpy as np
from typing import Union, Mapping


# ====================================================
# ================= Helper Functions =================
# ====================================================


# TODO: Fix this up to be more readable. Also make sure to mention keys of
# cohesion_parameters_for_bloc are slates now.
def _sample_pl_slate_ballots(
    config: BlocSlateConfig,
    num_ballots: int,
    cohesion_parameters_for_bloc: Mapping[str, Union[float, int]],
) -> list[list[str]]:
    """
    Returns a list of slate ballots; each ballot is a list of slate names (strings)
    in the order they appear on that ballot.
    Slates with 0 cohesion are randomly permuted and placed at the end of the ballot.


    Args:
        config (BlocSlateConfig):
        num_ballots (int):
        cohesion_parameters_for_bloc (Mapping[str, Union[float, int]]):

    Returns:
        list[list[str]]: A list of lists, where each list contains the bloc names in the order
            they appear on the ballot.
    """
    num_candidates = len(config.candidates)
    num_candidates_per_slate = {s: len(config.slate_to_candidates[s]) for s in config.slates}

    ballots: list[list[str]] = [[] for _ in range(num_ballots)]

    coin_flips = list(np.random.uniform(size=num_candidates * num_ballots))

    def which_bin(dist_bins: list[float], flip: float) -> int:
        for i, left in enumerate(dist_bins[:-1]):
            if left < flip <= dist_bins[i + 1]:
                return i
        return len(dist_bins) - 2

    blocs_og, values_og = [list(x) for x in zip(*cohesion_parameters_for_bloc.items())]

    for j in range(num_ballots):
        blocs = blocs_og.copy()
        values = values_og.copy()

        distribution_bins: list[float] = [0.0] + [
            sum(values[: i + 1]) for i in range(len(blocs))
        ]
        ballot_type: list[str] = [""] * num_candidates

        for i, flip in enumerate(
            coin_flips[j * num_candidates : (j + 1) * num_candidates]
        ):
            bloc_index = which_bin(distribution_bins, float(flip))
            bloc_type = blocs[bloc_index]
            ballot_type[i] = bloc_type

            if ballot_type.count(bloc_type) ==  num_candidates_per_slate[bloc_type]:
                del blocs[bloc_index]
                del values[bloc_index]
                total_value_sum = sum(values)

                if total_value_sum == 0 and len(values) > 0:
                    # remaining blocs have zero cohesion → fill with random permutation
                    remaining_blocs = [
                        b
                        for b in blocs
                        for _ in range(len(num_candidates_per_slate[b]))
                    ]
                    random.shuffle(remaining_blocs)
                    ballot_type[i + 1 :] = remaining_blocs
                    break

                # renormalize and recompute bins
                values = [v / total_value_sum for v in values]
                distribution_bins = [0.0] + [
                    sum(values[: k + 1]) for k in range(len(blocs))
                ]

        ballots[j] = ballot_type

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

        slate_ballots = _sample_pl_slate_ballots(
            config = config,
            num_ballots=n_ballots,
            cohesion_parameters_for_bloc=config.cohesion_df.loc[bloc].to_dict(),  # type: ignore
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
