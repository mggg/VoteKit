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


def _inner_slate_plackett_luce(
    config: BlocSlateConfig,
) -> dict[str, RankProfile]:
    """
    Inner function to generate preference profiles by bloc using the slate-Plackett-Luce model.

    This model first samples a ballot type by flipping a cohesion parameter weighted coin.
    It then fills out the ballot type via sampling with out replacement from the interval
    (i.e. according to the name-Plackett-Luce model).

    Args:
        config (BlocSlateConfig): Configuration object containing parameters for ballot generation.
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

    for i, bloc in enumerate(bloc_lst):
        # number of voters in this bloc
        n_ballots = ballots_per_bloc[bloc]
        ballot_pool = np.full((n_ballots, n_candidates), frozenset("~"))
        pref_intervals = config.get_preference_intervals_for_bloc(bloc)
        zero_cands = set(it.chain(*[pi.zero_cands for pi in pref_intervals.values()]))

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
            cand_ordering_by_bloc = {}

            for b in bloc_lst:
                # create a pref interval dict of only this blocs candidates
                bloc_cand_pref_interval = pref_intervals[b].interval
                cands = pref_intervals[b].non_zero_cands

                # if there are no non-zero candidates, skip this bloc
                if len(cands) == 0:
                    continue

                distribution = [bloc_cand_pref_interval[c] for c in cands]

                # sample
                cand_ordering = np.random.choice(
                    a=list(cands), size=len(cands), p=distribution, replace=False
                )
                cand_ordering_by_bloc[b] = list(cand_ordering)

            ranking = [frozenset({"-1"})] * len(bt)
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


def generate_slate_pl_profile(
    config: BlocSlateConfig,
    *,
    group_ballots: bool = True,
) -> RankProfile:
    """
    Generates a merged preference profiles using the slate-Plackett-Luce model.

    This model first samples a ballot type by flipping a cohesion parameter weighted coin.
    It then fills out the ballot type via sampling with out replacement from the interval
    (i.e. according to the name-Plackett-Luce model).

    Args:
        config (BlocSlateConfig): Configuration object containing parameters for ballot generation.
        group_ballots (bool): if True, group identical ballots in the returned profile and
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


def generate_slate_pl_profiles_by_bloc(
    config: BlocSlateConfig,
    *,
    group_ballots: bool = True,
) -> dict[str, RankProfile]:
    """
    Generates a merged preference profiles using the slate-Plackett-Luce model.

    This model first samples a ballot type by flipping a cohesion parameter weighted coin.
    It then fills out the ballot type via sampling with out replacement from the interval
    (i.e. according to the name-Plackett-Luce model).

    Args:
        config (BlocSlateConfig): Configuration object containing parameters for ballot generation.
        group_ballots (bool): if True, group identical ballots in the returned profile and
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


#
# class slate_PlackettLuce(BallotGenerator):
#     """
#     Class for generating ballots using a slate-PlackettLuce model.
#     This model first samples a ballot type by flipping a cohesion parameter weighted coin.
#     It then fills out the ballot type via sampling with out replacement from the interval.
#
#     Can be initialized with an interval or can be constructed with the Dirichlet distribution using
#     the `from_params` method of `BallotGenerator` class.
#
#     Args:
#         slate_to_candidates (dict): Dictionary whose keys are bloc names and whose
#             values are lists of candidate strings that make up the slate.
#         bloc_voter_prop (dict): Dictionary whose keys are bloc strings and values are floats
#                 denoting population share.
#         pref_intervals_by_bloc (dict): Dictionary whose keys are bloc strings and values are
#             dictionaries whose keys are bloc strings and values are ``PreferenceInterval`` objects.
#         cohesion_parameters (dict): Dictionary mapping of bloc string to dictionary whose
#             keys are bloc strings and values are cohesion parameters,
#             eg. ``{'bloc_1': {'bloc_1': .7, 'bloc_2': .2, 'bloc_3':.1}}``
#
#     Attributes:
#         candidates (list): List of candidate strings.
#         slate_to_candidates (dict): Dictionary whose keys are bloc names and whose
#             values are lists of candidate strings that make up the slate.
#         bloc_voter_prop (dict): Dictionary whose keys are bloc strings and values are floats
#                 denoting population share.
#         pref_intervals_by_bloc (dict): Dictionary whose keys are bloc strings and values are
#             dictionaries whose keys are bloc strings and values are ``PreferenceInterval`` objects.
#         cohesion_parameters (dict): Dictionary mapping of bloc string to dictionary whose
#             keys are bloc strings and values are cohesion parameters,
#             eg. ``{'bloc_1': {'bloc_1': .7, 'bloc_2': .2, 'bloc_3':.1}}``
#     """
#
#     def __init__(self, cohesion_parameters: dict, **data):
#         # Call the parent class's __init__ method to handle common parameters
#         super().__init__(cohesion_parameters=cohesion_parameters, **data)
#     def generate_profile(
#         self, number_of_ballots: int, by_bloc: bool = False
#     ) -> Union[PreferenceProfile, Tuple]:
#         """
#         Args:
#             number_of_ballots (int): The number of ballots to generate.
#             by_bloc (bool): True if you want the generated profiles returned as a tuple
#                 ``(pp_by_bloc, pp)``, where ``pp_by_bloc`` is a dictionary with keys = bloc strings
#                 and values = ``PreferenceProfile`` and ``pp`` is the aggregated profile. False if
#                 you only want the aggregated profile. Defaults to False.
#
#         Returns:
#             Union[PreferenceProfile, Tuple]
#         """
#         bloc_props = list(self.bloc_voter_prop.values())
#         ballots_per_block = dict(
#             zip(
#                 self.blocs,
#                 apportion.compute("huntington", bloc_props, number_of_ballots),
#             )
#         )
#
#         pref_profile_by_bloc = {}
#
#         for i, bloc in enumerate(self.blocs):
#             # number of voters in this bloc
#             n_ballots = ballots_per_block[bloc]
#             ballot_pool = [Ballot()] * n_ballots
#             pref_intervals = self.pref_intervals_by_bloc[bloc]
#             zero_cands = set(
#                 it.chain(*[pi.zero_cands for pi in pref_intervals.values()])
#             )
#
#             slate_to_non_zero_candidates = {
#                 s: [c for c in c_list if c not in zero_cands]
#                 for s, c_list in self.slate_to_candidates.items()
#             }
#
#             ballot_types = sample_cohesion_ballot_types(
#                 slate_to_non_zero_candidates=slate_to_non_zero_candidates,
#                 n_ballots=n_ballots,
#                 cohesion_parameters_for_bloc=self.cohesion_parameters[bloc],
#             )
#
#             for j, bt in enumerate(ballot_types):
#                 cand_ordering_by_bloc = {}
#
#                 for b in self.blocs:
#                     # create a pref interval dict of only this blocs candidates
#                     bloc_cand_pref_interval = pref_intervals[b].interval
#                     cands = pref_intervals[b].non_zero_cands
#
#                     # if there are no non-zero candidates, skip this bloc
#                     if len(cands) == 0:
#                         continue
#
#                     distribution = [bloc_cand_pref_interval[c] for c in cands]
#
#                     # sample
#                     cand_ordering = np.random.choice(
#                         a=list(cands), size=len(cands), p=distribution, replace=False
#                     )
#                     cand_ordering_by_bloc[b] = list(cand_ordering)
#
#                 ranking = [frozenset({"-1"})] * len(bt)
#                 for i, b in enumerate(bt):
#                     # append the current first candidate, then remove them from the ordering
#                     ranking[i] = frozenset({cand_ordering_by_bloc[b][0]})
#                     cand_ordering_by_bloc[b].pop(0)
#
#                 if len(zero_cands) > 0:
#                     ranking.append(frozenset(zero_cands))
#                 ballot_pool[j] = Ballot(ranking=tuple(ranking), weight=1)
#
#             pp = PreferenceProfile(ballots=tuple(ballot_pool))
#             pp = pp.group_ballots()
#             pref_profile_by_bloc[bloc] = pp
#
#         # combine the profiles
#         pp = PreferenceProfile()
#         for profile in pref_profile_by_bloc.values():
#             pp += profile
#
#         if by_bloc:
#             return (pref_profile_by_bloc, pp)
#
#         # else return the combined profiles
#         else:
#             return pp
