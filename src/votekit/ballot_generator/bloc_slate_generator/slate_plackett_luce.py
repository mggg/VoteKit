"""
Generate ranked preference profiles using the slate-Plackett-Luce model.

The main API functions in this module are:

- `slate_pl_profile_generator`: Generates a single preference profile using the slate-Plackett-Luce
    model.
- `slate_pl_profiles_by_bloc_generator`: Generates preference profiles by bloc using the
    slate-Plackett-Luce model.
"""

import itertools as it
import numpy as np
import pandas as pd
from typing import Mapping
import apportionment.methods as apportion

from votekit.pref_profile import RankProfile
from votekit.ballot_generator import (
    sample_cohesion_ballot_types,
)
from votekit.ballot_generator.bloc_slate_generator.model import BlocSlateConfig
from votekit.ballot_generator.bloc_slate_generator.slate_utils import (
    _make_cand_ordering_by_slate,
    _convert_ballot_type_to_ranking,
)

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

    pref_profile_by_bloc: Mapping[str, RankProfile] = {
        b: RankProfile() for b in bloc_lst
    }

    pref_profile_by_bloc = {}

    for bloc in bloc_lst:
        n_ballots = ballots_per_bloc[bloc]
        ballot_pool = np.full((n_ballots, n_candidates), frozenset("~"))
        pref_intervals_by_slate_dict = config.get_preference_intervals_for_bloc(bloc)
        zero_cands = set(
            it.chain(*[pi.zero_cands for pi in pref_intervals_by_slate_dict.values()])
        )

        slate_to_non_zero_candidates = {
            s: [c for c in c_list if c not in zero_cands]
            for s, c_list in config.slate_to_candidates.items()
        }

        ballot_types = sample_cohesion_ballot_types(
            slate_to_non_zero_candidates=slate_to_non_zero_candidates,
            num_ballots=n_ballots,
            cohesion_parameters_for_bloc=config.cohesion_df.loc[bloc].to_dict(),  # type: ignore
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
