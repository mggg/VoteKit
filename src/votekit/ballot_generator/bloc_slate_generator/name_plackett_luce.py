"""
Generate ranked preference profiles using the name-Plackett-Luce model.

The main API functions in this module are:

- `name_pl_profile_generator`: Generates a single preference profile using the name-Plackett-Luce
    model.
- `name_pl_profiles_by_bloc_generator`: Generates preference profiles by bloc using the
    name-Plackett-Luce model.
"""

import numpy as np
from typing import Optional
import pandas as pd
import apportionment.methods as apportion

from votekit.ballot_generator.bloc_slate_generator.model import BlocSlateConfig
from votekit.pref_profile import RankProfile


# ===========================================================
# ================= Interior Work Functions =================
# ===========================================================


def _inner_name_plackett_luce(
    config: BlocSlateConfig,
    *,
    ballot_length: Optional[int] = None,
) -> dict[str, RankProfile]:
    """
    Inner function to generate preference profiles by bloc using the name-Plackett-Luce model.

    The Plackett-Luce model samples without replacement from each preference interval
    for each bloc of voters.

    Args:
        config (BlocSlateConfig): Configuration object containing all necessary parameters for
            working with a bloc-slate ballot generator.
        ballot_length (Optional[int]): Number of ranking positions allowed per ballot. If None,
            this is set to the total number of candidates in the configuration. Defaults to None.

    Returns:
        dict[str, RankProfile]: Dictionary whose keys are bloc strings and values are
            ``RankProfile`` objects.
    """
    n_candidates = len(config.candidates)
    if ballot_length is None:
        ballot_length = n_candidates

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

    for bloc in config.blocs:
        n_ballots = ballots_per_bloc[bloc]
        ballot_pool = np.full((n_ballots, ballot_length), frozenset("~"))
        non_zero_cands = list(pref_interval_by_bloc_dict[bloc].non_zero_cands)
        pref_interval_values = [
            pref_interval_by_bloc_dict[bloc].interval[c] for c in non_zero_cands
        ]
        zero_cands = list(pref_interval_by_bloc_dict[bloc].zero_cands)

        # if there aren't enough non-zero supported candidates,
        # include 0 support as ties
        number_to_sample = ballot_length
        number_tied = None

        if len(non_zero_cands) < number_to_sample:
            number_tied = number_to_sample - len(non_zero_cands)
            number_to_sample = len(non_zero_cands)

        for i in range(n_ballots):
            non_zero_ranking = list(
                np.random.choice(
                    non_zero_cands,
                    number_to_sample,
                    p=pref_interval_values,
                    replace=False,
                )
            )

            ranking = [frozenset({cand}) for cand in non_zero_ranking]

            if number_tied is not None:
                ranking.append(frozenset(zero_cands))

            ballot_pool[i] = np.array(ranking)

        df = pd.DataFrame(ballot_pool)
        df.index.name = "Ballot Index"
        df.columns = [f"Ranking_{i + 1}" for i in range(n_candidates)]
        df["Weight"] = 1
        df["Voter Set"] = [frozenset()] * len(df)
        pp = RankProfile(
            candidates=config.candidates,
            df=df,
            max_ranking_length=ballot_length,
        )
        pp_by_bloc[bloc] = pp

    return pp_by_bloc


# =================================================
# ================= API Functions =================
# =================================================

# NOTE: The 'ballot_length' parameter has been commented out
# since we don't have good analogues in any of the other Bloc-Slate
# generators. When we add this feature, we can just restore the functionality.


def name_pl_profile_generator(
    config: BlocSlateConfig,
    *,
    # ballot_length: Optional[int] = None,
    group_ballots: bool = True,
) -> RankProfile:
    """
    Generates a merged preference profile using the name-Plackett-Luce model.

    The Plackett-Luce model samples without replacement from each preference interval
    for each bloc of voters. These profiles are then merged into a single profile.

    Args:
        config (BlocSlateConfig): Configuration object containing all necessary parameters for
            working with a bloc-slate ballot generator.
        group_ballots (bool): If True, group identical ballots in the returned profile and
            set the weight accordingly. Defaults to True.

    Returns:
        RankProfile: Merged ``RankProfile`` object generated by the model.
    """
    config.is_valid(raise_errors=True)

    # pp_by_bloc = _inner_name_plackett_luce(config, ballot_length=ballot_length)
    pp_by_bloc = _inner_name_plackett_luce(config)

    pp = RankProfile(ballots=tuple())
    for profile in pp_by_bloc.values():
        pp += profile

    if group_ballots:
        pp = pp.group_ballots()

    return pp


def name_pl_profiles_by_bloc_generator(
    config: BlocSlateConfig,
    *,
    # ballot_length: Optional[int] = None,
    group_ballots: bool = True,
) -> dict[str, RankProfile]:
    """
    Generates a dictionary mapping bloc names to preference profiles by bloc using the
    name-Plackett-Luce model.

    The Plackett-Luce model samples without replacement from each preference interval
    for each bloc of voters.

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

    # pp_by_bloc = _inner_name_plackett_luce(config, ballot_length=ballot_length)
    pp_by_bloc = _inner_name_plackett_luce(config)

    if group_ballots:
        for bloc, profile in pp_by_bloc.items():
            pp_by_bloc[bloc] = profile.group_ballots()

    return pp_by_bloc
