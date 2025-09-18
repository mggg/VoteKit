import itertools as it
import numpy as np
from typing import Union, Tuple
import apportionment.methods as apportion

from votekit.ballot import Ballot
from votekit.pref_profile import RankProfile
from votekit.ballot_generator import BallotGenerator, sample_cohesion_ballot_types


class slate_PlackettLuce(BallotGenerator):
    """
    Class for generating ballots using a slate-PlackettLuce model.
    This model first samples a ballot type by flipping a cohesion parameter weighted coin.
    It then fills out the ballot type via sampling with out replacement from the interval.

    Can be initialized with an interval or can be constructed with the Dirichlet distribution using
    the `from_params` method of `BallotGenerator` class.

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

    def generate_profile(
        self, number_of_ballots: int, by_bloc: bool = False
    ) -> Union[RankProfile, Tuple]:
        """
        Args:
            number_of_ballots (int): The number of ballots to generate.
            by_bloc (bool): True if you want the generated profiles returned as a tuple
                ``(pp_by_bloc, pp)``, where ``pp_by_bloc`` is a dictionary with keys = bloc strings
                and values = ``RankProfile`` and ``pp`` is the aggregated profile. False if
                you only want the aggregated profile. Defaults to False.

        Returns:
            Union[RankProfile, Tuple]
        """
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
            ballot_pool = [Ballot()] * num_ballots
            pref_intervals = self.pref_intervals_by_bloc[bloc]
            zero_cands = set(
                it.chain(*[pi.zero_cands for pi in pref_intervals.values()])
            )

            slate_to_non_zero_candidates = {
                s: [c for c in c_list if c not in zero_cands]
                for s, c_list in self.slate_to_candidates.items()
            }

            ballot_types = sample_cohesion_ballot_types(
                slate_to_non_zero_candidates=slate_to_non_zero_candidates,
                num_ballots=num_ballots,
                cohesion_parameters_for_bloc=self.cohesion_parameters[bloc],
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
                ballot_pool[j] = Ballot(ranking=tuple(ranking), weight=1)

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
