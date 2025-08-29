import numpy as np
from typing import Union, Tuple
import apportionment.methods as apportion

from votekit.ballot import Ballot
from votekit.pref_profile import PreferenceProfile
from votekit.pref_interval import combine_preference_intervals, PreferenceInterval
from votekit.ballot_generator import BallotGenerator


class short_name_PlackettLuce(BallotGenerator):
    """
    Class for generating short name-Plackett-Luce ballots. This model samples without
    replacement from a preference interval. Equivalent to name-PlackettLuce if
    ``ballot_length`` = number of candidates. Can be initialized with an interval or can be
    constructed with the Dirichlet distribution using the ``from_params`` method of
    ``BallotGenerator``.

    Args:
        slate_to_candidates (dict): Dictionary whose keys are bloc names and whose
            values are lists of candidate strings that make up the slate.
        bloc_voter_prop (dict): Dictionary whose keys are bloc strings and values are floats
                denoting population share.
        cohesion_parameters (dict): Dictionary mapping of bloc string to dictionary whose
            keys are bloc strings and values are cohesion parameters,
            eg. ``{'bloc_1': {'bloc_1': .7, 'bloc_2': .2, 'bloc_3':.1}}``
        ballot_length (int): Number of votes allowed per ballot.

    Attributes:
        candidates (list): List of candidate strings.
        slate_to_candidates (dict): Dictionary whose keys are bloc names and whose
            values are lists of candidate strings that make up the slate.
        bloc_voter_prop (dict): Dictionary whose keys are bloc strings and values are floats
                denoting population share.
        cohesion_parameters (dict): Dictionary mapping of bloc string to dictionary whose
            keys are bloc strings and values are cohesion parameters,
            eg. ``{'bloc_1': {'bloc_1': .7, 'bloc_2': .2, 'bloc_3':.1}}``
        ballot_length (int): Number of votes allowed per ballot.
    """

    def __init__(self, ballot_length: int, **data):
        # Call the parent class's __init__ method to handle common parameters
        super().__init__(**data)
        self.ballot_length = ballot_length

        # if dictionary of pref intervals
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
    ) -> Union[PreferenceProfile, Tuple]:
        """
        Args:
            number_of_ballots (int): The number of ballots to generate.
            by_bloc (bool): True if you want the generated profiles returned as a tuple
                ``(pp_by_bloc, pp)``, where ``pp_by_bloc`` is a dictionary with keys = bloc strings
                and values = ``PreferenceProfile`` and ``pp`` is the aggregated profile. False if
                you only want the aggregated profile. Defaults to False.

        Returns:
            Union[PreferenceProfile, Tuple]
        """
        # the number of ballots per bloc is determined by Huntington-Hill apportionment
        bloc_props = list(self.bloc_voter_prop.values())
        ballots_per_block = dict(
            zip(
                self.blocs,
                apportion.compute("huntington", bloc_props, number_of_ballots),
            )
        )

        # dictionary to store preference profiles by bloc
        pp_by_bloc = {b: PreferenceProfile() for b in self.blocs}

        for bloc in self.blocs:
            # number of voters in this bloc
            num_ballots = ballots_per_block[bloc]
            ballot_pool = [Ballot()] * num_ballots
            non_zero_cands = list(self.pref_interval_by_bloc[bloc].non_zero_cands)
            pref_interval_values = [
                self.pref_interval_by_bloc[bloc].interval[c] for c in non_zero_cands
            ]
            zero_cands = list(self.pref_interval_by_bloc[bloc].zero_cands)

            # if there aren't enough non-zero supported candidates,
            # include 0 support as ties
            number_to_sample = self.ballot_length
            number_tied = None

            if len(non_zero_cands) < number_to_sample:
                number_tied = number_to_sample - len(non_zero_cands)
                number_to_sample = len(non_zero_cands)

            for i in range(num_ballots):
                # generates ranking based on probability distribution of non candidate support
                # samples ballot_length candidates
                non_zero_ranking = list(
                    np.random.choice(
                        non_zero_cands,
                        number_to_sample,
                        p=pref_interval_values,
                        replace=False,
                    )
                )

                ranking = [frozenset({cand}) for cand in non_zero_ranking]

                if number_tied:
                    tied_candidates = list(
                        np.random.choice(
                            zero_cands,
                            number_tied,
                            replace=False,
                        )
                    )
                    ranking.append(frozenset(tied_candidates))

                ballot_pool[i] = Ballot(ranking=tuple(ranking), weight=1)

            # create PP for this bloc
            pp = PreferenceProfile(ballots=tuple(ballot_pool))
            pp = pp.group_ballots()
            pp_by_bloc[bloc] = pp

        # combine the profiles
        pp = PreferenceProfile(ballots=tuple())
        for profile in pp_by_bloc.values():
            pp += profile

        if by_bloc:
            return (pp_by_bloc, pp)

        # else return the combined profiles
        else:
            return pp


class name_PlackettLuce(short_name_PlackettLuce):
    """
    Class for generating name-Plackett-Luce ballots. This model samples without
    replacement from a preference interval. Can be initialized with an interval or can be
    constructed with the Dirichlet distribution using the ``from_params`` method of
    ``BallotGenerator``.

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
        pref_interval_by_bloc (dict): Dictionary whose keys are bloc strings and values are
            ``PreferenceInterval`` objects. This is constructed by rescaling the intervals
            from ``pref_intervals_by_bloc`` via the ``cohesion_parameters`` and concatenating them.
        cohesion_parameters (dict): Dictionary mapping of bloc string to dictionary whose
            keys are bloc strings and values are cohesion parameters,
            eg. ``{'bloc_1': {'bloc_1': .7, 'bloc_2': .2, 'bloc_3':.1}}``
        ballot_length (int): Number of votes allowed per ballot.
    """

    def __init__(self, cohesion_parameters: dict, **data):
        if "candidates" in data:
            ballot_length = len(data["candidates"])
        elif "slate_to_candidates" in data:
            ballot_length = sum(
                len(c_list) for c_list in data["slate_to_candidates"].values()
            )
        else:
            raise ValueError("One of candidates or slate_to_candidates must be passed.")

        # Call the parent class's __init__ method to handle common parameters
        super().__init__(
            ballot_length=ballot_length, cohesion_parameters=cohesion_parameters, **data
        )
