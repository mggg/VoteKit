"""
Generate ranked preference profiles using the name-Plackett-Luce model.

The main API functions in this module are:

- `name_pl_profile_generator`: Generates a single preference profile using the name-Plackett-Luce
    model.
- `name_pl_profiles_by_bloc_generator`: Generates preference profiles by bloc using the
    name-Plackett-Luce model.
"""

from typing import Optional

import apportionment.methods as apportion
import numpy as np
import pandas as pd

from votekit.ballot_generator.bloc_slate_generator.config import BlocSlateConfig
from votekit.pref_profile import RankProfile

# ===========================================================
# ================= Interior Work Functions =================
# ===========================================================


def _inner_name_plackett_luce(
    config: BlocSlateConfig,
    *,
    ballot_length: Optional[int] = None,
    random_seed: Optional[int] = None,
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
        random_seed (int | None): seed for RNG, allows for reproducible results given the same
            inputs. Seed set to None by default, different results will be generated each time.

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
            raise TypeError(f"Unexpected type from apportionment got {type(bloc_counts)}")

        bloc_counts = [bloc_counts]

    ballots_per_bloc = {bloc: bloc_counts[i] for i, bloc in enumerate(bloc_lst)}

    pp_by_bloc = {b: RankProfile() for b in bloc_lst}
    pref_interval_by_bloc_dict = config.get_combined_preference_intervals_by_bloc()
    rng = np.random.default_rng(seed=random_seed)

    for bloc in config.blocs:
        n_ballots = ballots_per_bloc[bloc]
        ballot_pool = np.full((n_ballots, ballot_length), frozenset("~"))
        cands = list(pref_interval_by_bloc_dict[bloc].interval.keys())
        pref_interval_values = [pref_interval_by_bloc_dict[bloc].interval[c] for c in cands]
        for i in range(n_ballots):
            chosen_indices = list(
                rng.choice(
                    len(cands),
                    ballot_length,
                    p=pref_interval_values,
                    replace=False,
                )
            )
            non_zero_ranking = [cands[cand_idx] for cand_idx in chosen_indices]
            ranking = [frozenset({cand}) for cand in non_zero_ranking]
            ballot_pool[i] = np.array(ranking)

        df = pd.DataFrame(ballot_pool)
        df.index.name = "Ballot Index"
        df.columns = [f"Ranking_{i + 1}" for i in range(n_candidates)]
        df["Weight"] = 1
        df.insert(
            len(df.columns),
            "Voter Set",
            pd.Series([frozenset()] * len(df), dtype=object, index=df.index),
        )
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
    random_seed: Optional[int] = None,
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
        random_seed (int | None): seed for RNG, allows for reproducible results given the same
            inputs. Seed set to None by default, different results will be generated each time.

    Returns:
        RankProfile: Merged ``RankProfile`` object generated by the model.
    """
    config.is_valid(raise_errors=True)

    # pp_by_bloc = _inner_name_plackett_luce(config, ballot_length=ballot_length)
    pp_by_bloc = _inner_name_plackett_luce(config, random_seed=random_seed)

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
    random_seed: Optional[int] = None,
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
        random_seed (int | None): seed for RNG, allows for reproducible results given the same
            inputs. Seed set to None by default, different results will be generated each time.

    Returns:
        dict[str, RankProfile]: Dictionary whose keys are bloc strings and values are
            ``RankProfile`` objects generated by the model for each bloc.
    """
    config.is_valid(raise_errors=True)

    # pp_by_bloc = _inner_name_plackett_luce(config, ballot_length=ballot_length)
    pp_by_bloc = _inner_name_plackett_luce(config, random_seed=random_seed)

    if group_ballots:
        for bloc, profile in pp_by_bloc.items():
            pp_by_bloc[bloc] = profile.group_ballots()

    return pp_by_bloc
