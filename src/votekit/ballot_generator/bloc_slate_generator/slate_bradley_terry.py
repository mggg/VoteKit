import itertools as it
import numpy as np
import random
import warnings
from typing import Union, Tuple
import apportionment.methods as apportion

from votekit.ballot import RankBallot
from votekit.pref_profile import RankProfile
from votekit.ballot_generator import BallotGenerator

# # ====================================================
# # ================= Helper Functions =================
# # ====================================================
#
#
# def _compute_ballot_type_dist(config, bloc_a, slate_a, bloc_b):
#     """
#     Return a dictionary with keys ballot types and values equal to probability of sampling.
#     """
#     slates_to_sample = [
#         slate
#         for slate in config.slates_to_candidates.keys()
#         for _ in range(
#             len(
#                 config.get_preference_interval_for_bloc_and_slate(
#                     bloc_name=bloc_a, slate_name=slate
#                 ).non_zero_cands
#             )
#         )
#     ]
#     total_comparisons = np.prod(
#         [
#             len(
#                 config.get_preference_interval_for_bloc_and_slate(
#                     bloc_name=bloc_a, slate_name=slate
#                 ).non_zero_cands
#             )
#             for slate in config.slates_to_candidates.keys()
#         ]
#     )
#
#     cohesion = config.cohesion_df[slate_a].loc[bloc_a]
#
#     def prob_of_type(ballot_type):
#         success = sum(
#             ballot_type[i + 1 :].count(bloc_b)  # noqa
#             for i, b in enumerate(ballot_type)
#             if b == bloc_a
#         )
#         return pow(cohesion, success) * pow(1 - cohesion, total_comparisons - success)
#
#     pdf = {
#         b: prob_of_type(b)
#         for b in set(it.permutations(slates_to_sample, len(slates_to_sample)))
#     }
#
#     summ = sum(pdf.values())
#     return {b: v / summ for b, v in pdf.items()}
#
#
# def _sample_ballot_types_deterministic(self, bloc: str, num_ballots: int):
#     """
#     Used to generate bloc orderings for deliberative.
#
#     Returns a list of lists, where each sublist contains the bloc names in order they appear
#     on the ballot.
#     """
#     # pdf = self._compute_ballot_type_dist(bloc=bloc, opp_bloc=opp_bloc)
#     pdf = self.ballot_type_pdf[bloc]
#     b_types = list(pdf.keys())
#     probs = list(pdf.values())
#
#     sampled_indices = np.random.choice(len(b_types), size=num_ballots, p=probs)
#
#     return [b_types[i] for i in sampled_indices]
#
#
# def _sample_ballot_types_MCMC(self, bloc: str, num_ballots: int, verbose: bool = False):
#     """
#     Generate ballot types using MCMC that has desired stationary distribution.
#     """
#
#     seed_ballot_type = [
#         b
#         for b in self.blocs
#         for _ in range(len(self.pref_intervals_by_bloc[bloc][b].non_zero_cands))
#     ]
#
#     ballots = [[-1]] * num_ballots
#     accept = 0
#     current_ranking = seed_ballot_type
#
#     cohesion = self.cohesion_parameters[bloc][bloc]
#
#     # presample swap indices
#     swap_indices = [
#         (j1, j1 + 1)
#         for j1 in np.random.choice(len(seed_ballot_type) - 1, size=num_ballots)
#     ]
#
#     odds = (1 - cohesion) / cohesion
#     # generate MCMC sample
#     for i in range(num_ballots):
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
#             f"Acceptance ratio as number accepted / total steps: {accept/num_ballots:.2}"
#         )
#
#     if -1 in ballots:
#         raise ValueError("Some element of ballots list is not a ballot.")
#
#     return ballots
#
#
# # ===========================================================
# # ================= Interior Work Functions =================
# # ===========================================================


class slate_BradleyTerry(BallotGenerator):
    """
    Class for generating ballots using a slate-BradleyTerry model. It
    presamples ballot types by checking all pairwise comparisons, then fills out candidate
    ordering by sampling without replacement from preference intervals.

    Only works with 2 blocs at the moment.

    Can be initialized with an interval or can be constructed with the Dirichlet distribution using
    the `from_params` method of `BallotGenerator`.

    Args:
        slate_to_candidates (dict): Dictionary whose keys are bloc names and whose
            values are lists of candidate strings that make up the slate.
        bloc_voter_prop (dict): Dictionary whose keys are bloc strings and values are floats
                denoting population share.
        pref_intervals_by_bloc (dict): Dictionary whose keys are bloc strings and values are
            dictionaries whose keys are bloc strings and values are ``PreferenceInterval`` objects.
        cohesion_parameters (dict): Dictionary mapping of bloc string to dictionary whose
            keys are bloc strings and values are cohesion parameters,
            eg. ``{'bloc_1': {'bloc_1': .7, 'bloc_2': .2, 'bloc_3':.1}}``

    Attributes:
        candidates (list): List of candidate strings.
        slate_to_candidates (dict): Dictionary whose keys are bloc names and whose
            values are lists of candidate strings that make up the slate.
        bloc_voter_prop (dict): Dictionary whose keys are bloc strings and values are floats
                denoting population share.
        pref_intervals_by_bloc (dict): Dictionary whose keys are bloc strings and values are
            dictionaries whose keys are bloc strings and values are ``PreferenceInterval`` objects.
        cohesion_parameters (dict): Dictionary mapping of bloc string to dictionary whose
            keys are bloc strings and values are cohesion parameters,
            eg. ``{'bloc_1': {'bloc_1': .7, 'bloc_2': .2, 'bloc_3':.1}}``
    """

    def __init__(self, cohesion_parameters: dict, **data):
        # Call the parent class's __init__ method to handle common parameters
        super().__init__(cohesion_parameters=cohesion_parameters, **data)

        if len(self.slate_to_candidates.keys()) > 2:
            raise UserWarning(
                f"This model currently only supports at most two blocs, but you \
                              passed {len(self.slate_to_candidates.keys())}"
            )

        if len(self.candidates) < 12 and len(self.blocs) == 2:
            # precompute pdfs for sampling
            self.ballot_type_pdf = {
                b: self._compute_ballot_type_dist(b, self.blocs[(i + 1) % 2])
                for i, b in enumerate(self.blocs)
            }

        elif len(self.blocs) == 1:
            # precompute pdf for sampling
            # uniform on ballot types
            bloc = self.blocs[0]
            bloc_to_sample = [
                bloc
                for _ in range(
                    len(self.pref_intervals_by_bloc[bloc][bloc].non_zero_cands)
                )
            ]
            pdf = {tuple(bloc_to_sample): 1}
            self.ballot_type_pdf = {bloc: pdf}

        else:
            warnings.warn(
                "For 12 or more candidates, exact sampling is computationally infeasible. \
                    Please set deterministic = False when calling generate_profile."
            )

    def _compute_ballot_type_dist(self, bloc, opp_bloc):
        """
        Return a dictionary with keys ballot types and values equal to probability of sampling.
        """
        blocs_to_sample = [
            b
            for b in self.blocs
            for _ in range(len(self.pref_intervals_by_bloc[bloc][b].non_zero_cands))
        ]
        total_comparisons = np.prod(
            [
                len(interval.non_zero_cands)
                for interval in self.pref_intervals_by_bloc[bloc].values()
            ]
        )

        cohesion = self.cohesion_parameters[bloc][bloc]

        def prob_of_type(b_type):
            success = sum(
                b_type[i + 1 :].count(opp_bloc)
                for i, b in enumerate(b_type)
                if b == bloc
            )
            return pow(cohesion, success) * pow(
                1 - cohesion, total_comparisons - success
            )

        pdf = {
            b: prob_of_type(b)
            for b in set(it.permutations(blocs_to_sample, len(blocs_to_sample)))
        }

        summ = sum(pdf.values())
        return {b: v / summ for b, v in pdf.items()}

    def _sample_ballot_types_deterministic(self, bloc: str, num_ballots: int):
        """
        Used to generate bloc orderings for deliberative.

        Returns a list of lists, where each sublist contains the bloc names in order they appear
        on the ballot.
        """
        # pdf = self._compute_ballot_type_dist(bloc=bloc, opp_bloc=opp_bloc)
        pdf = self.ballot_type_pdf[bloc]
        b_types = list(pdf.keys())
        probs = list(pdf.values())

        sampled_indices = np.random.choice(len(b_types), size=num_ballots, p=probs)

        return [b_types[i] for i in sampled_indices]

    def _sample_ballot_types_MCMC(
        self, bloc: str, num_ballots: int, verbose: bool = False
    ):
        """
        Generate ballot types using MCMC that has desired stationary distribution.
        """

        seed_ballot_type = [
            b
            for b in self.blocs
            for _ in range(len(self.pref_intervals_by_bloc[bloc][b].non_zero_cands))
        ]

        ballots = [[-1]] * num_ballots
        accept = 0
        current_ranking = seed_ballot_type

        cohesion = self.cohesion_parameters[bloc][bloc]

        # presample swap indices
        swap_indices = [
            (j1, j1 + 1)
            for j1 in np.random.choice(len(seed_ballot_type) - 1, size=num_ballots)
        ]

        odds = (1 - cohesion) / cohesion
        # generate MCMC sample
        for i in range(num_ballots):
            # choose adjacent pair to propose a swap
            j1, j2 = swap_indices[i]

            # if swap reduces number of voters bloc above opposing bloc
            if (
                current_ranking[j1] != current_ranking[j2]
                and current_ranking[j1] == bloc
            ):
                acceptance_prob = odds

            # if swap increases number of voters bloc above opposing or swaps two of same bloc
            else:
                acceptance_prob = 1

            # if you accept, make the swap
            if random.random() < acceptance_prob:
                current_ranking[j1], current_ranking[j2] = (
                    current_ranking[j2],
                    current_ranking[j1],
                )
                accept += 1

            ballots[i] = current_ranking.copy()

        if verbose:
            print(
                f"Acceptance ratio as number accepted / total steps: {accept/num_ballots:.2}"
            )

        if -1 in ballots:
            raise ValueError("Some element of ballots list is not a ballot.")

        return ballots

    def generate_profile(
        self, number_of_ballots: int, by_bloc: bool = False, deterministic: bool = True
    ) -> Union[RankProfile, Tuple]:
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
            Union[RankProfile, Tuple]
        """
        # the number of ballots per bloc is determined by Huntington-Hill apportionment
        bloc_props = list(self.bloc_voter_prop.values())
        ballots_per_block = dict(
            zip(
                self.blocs,
                apportion.compute("huntington", bloc_props, number_of_ballots),
            )
        )

        pref_profile_by_bloc = {}

        for i, bloc in enumerate(self.blocs):
            # number of voters in this bloc
            num_ballots = ballots_per_block[bloc]
            ballot_pool = [RankBallot()] * num_ballots
            pref_intervals = self.pref_intervals_by_bloc[bloc]
            zero_cands = set(
                it.chain(*[pi.zero_cands for pi in pref_intervals.values()])
            )

            if deterministic and len(self.candidates) >= 12:
                raise UserWarning(
                    "Deterministic sampling is only supported for 11 or fewer candidates.\n\
                    Please set deterministic = False."
                )

            elif deterministic:
                ballot_types = self._sample_ballot_types_deterministic(
                    bloc=bloc, num_ballots=num_ballots
                )
            else:
                ballot_types = self._sample_ballot_types_MCMC(
                    bloc=bloc, num_ballots=num_ballots
                )

            for j, bt in enumerate(ballot_types):
                cand_ordering_by_bloc = {}

                for b in self.blocs:
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
                ballot_pool[j] = RankBallot(ranking=tuple(ranking), weight=1)

            pp = RankProfile(ballots=tuple(ballot_pool))
            pp = pp.group_ballots()
            pref_profile_by_bloc[bloc] = pp

        # combine the profiles
        pp = RankProfile()
        for profile in pref_profile_by_bloc.values():
            pp += profile

        if by_bloc:
            return (pref_profile_by_bloc, pp)

        # else return the combined profiles
        else:
            return pp
