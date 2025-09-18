import numpy as np
from typing import Union, Tuple
import apportionment.methods as apportion

from votekit.ballot import Ballot
from votekit.pref_profile import ScoreProfile
from votekit.pref_interval import combine_preference_intervals, PreferenceInterval
from votekit.ballot_generator import BallotGenerator


class name_Cumulative(BallotGenerator):
    """
    Class for generating cumulative ballots, which have scores, not rankings. This model samples
    with replacement from a combined preference interval and counts candidates with multiplicity.
    Can be initialized with an interval or can be constructed with the Dirichlet distribution
    using the `from_params` method of `BallotGenerator`.

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
        num_votes (int): The number of votes allowed per ballot.

    Attributes:
        candidates (list): List of candidate strings.
        slate_to_candidates (dict): Dictionary whose keys are bloc names and whose
            values are lists of candidate strings that make up the slate.
        bloc_voter_prop (dict): Dictionary whose keys are bloc strings and values are floats
                denoting population share.
        pref_intervals_by_bloc (dict): Dictionary whose keys are bloc strings and values are
            dictionaries whose keys are bloc strings and values are ``PreferenceInterval`` objects.
        pref_interval_by_bloc (dict): Dictionary whose keys are bloc strings and values are
            ``PreferenceInterval`` objects. This is constructed by rescaling the intervals
            from ``pref_intervals_by_bloc`` via the ``cohesion_parameters`` and concatenating them.
        cohesion_parameters (dict): Dictionary mapping of bloc string to dictionary whose
            keys are bloc strings and values are cohesion parameters,
            eg. ``{'bloc_1': {'bloc_1': .7, 'bloc_2': .2, 'bloc_3':.1}}``
        num_votes (int): The number of votes allowed per ballot.
    """

    def __init__(self, cohesion_parameters: dict, num_votes: int, **data):
        # Call the parent class's __init__ method to handle common parameters
        super().__init__(cohesion_parameters=cohesion_parameters, **data)
        self.num_votes = num_votes

        # if dictionary of pref intervals is passed
        if isinstance(
            list(self.pref_intervals_by_bloc.values())[0], PreferenceInterval
        ):
            self.pref_interval_by_bloc = self.pref_intervals_by_bloc

        # if nested dictionary of pref intervals, combine by cohesion
        else:
            self.pref_interval_by_bloc = {
                bloc: combine_preference_intervals(
                    [self.pref_intervals_by_bloc[bloc][b] for b in self.blocs],
                    [self.cohesion_parameters[bloc][b] for b in self.blocs],
                )
                for bloc in self.blocs
            }

    def generate_profile(
        self, number_of_ballots: int, by_bloc: bool = False
    ) -> Union[ScoreProfile, Tuple]:
        """
        Args:
            number_of_ballots (int): The number of ballots to generate.
            by_bloc (bool): True if you want the generated profiles returned as a tuple
                ``(pp_by_bloc, pp)``, where ``pp_by_bloc`` is a dictionary with keys = bloc strings
                and values = ``ScoreProfile`` and ``pp`` is the aggregated profile. False if
                you only want the aggregated profile. Defaults to False.

        Returns:
            Union[ScoreProfile, Tuple]
        """
        # the number of ballots per bloc is determined by Huntington-Hill apportionment
        bloc_props = list(self.bloc_voter_prop.values())
        ballots_per_block = dict(
            zip(
                self.blocs,
                apportion.compute("huntington", bloc_props, number_of_ballots),
            )
        )

        pp_by_bloc = {b: ScoreProfile() for b in self.blocs}

        for bloc in self.bloc_voter_prop.keys():
            ballot_pool = []
            # number of voters in this bloc
            num_ballots = ballots_per_block[bloc]
            pref_interval = self.pref_interval_by_bloc[bloc]

            # finds candidates with non-zero preference
            non_zero_cands = list(pref_interval.non_zero_cands)
            # creates the interval of probabilities for candidates supported by this block
            cand_support_vec = [pref_interval.interval[cand] for cand in non_zero_cands]

            for _ in range(num_ballots):
                # generates ranking based on probability distribution of non zero candidate support
                list_ranking = list(
                    np.random.choice(
                        non_zero_cands,
                        self.num_votes,
                        p=cand_support_vec,
                        replace=True,
                    )
                )

                scores = {c: 0.0 for c in list_ranking}
                for c in list_ranking:
                    scores[c] += 1

                ballot_pool.append(Ballot(scores=scores, weight=1))

            pp = ScoreProfile(ballots=tuple(ballot_pool))
            pp = pp.group_ballots()
            pp_by_bloc[bloc] = pp

        # combine the profiles
        pp = ScoreProfile()
        for profile in pp_by_bloc.values():
            pp += profile

        if by_bloc:
            return (pp_by_bloc, pp)

        # else return the combined profiles
        else:
            return pp
