import numpy as np
from pathlib import Path
import pickle
import random
from typing import Optional, Union, Tuple
import apportionment.methods as apportion

from votekit.ballot import Ballot
from votekit.pref_profile import RankProfile
from votekit.pref_interval import combine_preference_intervals
from votekit.ballot_generator import BallotGenerator


class CambridgeSampler(BallotGenerator):
    """
    Class for generating ballots based on historical RCV elections occurring
    in Cambridge, MA. Alternative election data can be used if specified. Assumes that there are two
    blocs, a W and C bloc, which corresponds to the historical Cambridge data.
    By default, it assigns the W bloc to the majority bloc and C to the minority, but this
    can be changed.

    Based on cohesion parameters, decides if a voter casts their top choice within their bloc
    or in the opposing bloc. Then uses historical data; given their first choice, choose a
    ballot type from the historical distribution.

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
        W_bloc (str, optional): Name of the bloc corresponding to the W bloc. Defaults to
            whichever bloc has majority via ``bloc_voter_prop``.
        C_bloc (str, optional): Name of the bloc corresponding to the C bloc. Defaults to
            whichever bloc has minority via ``bloc_voter_prop``.
        historical_majority (str, optional): Name of majority bloc in historical data, defaults to W
            for Cambridge data.
        historical_minority (str, optional): Name of minority bloc in historical data, defaults to C
            for Cambridge data.
        path (str, optional): File path to an election data file to sample from. Defaults to
            Cambridge elections.

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
        W_bloc (str): The name of the W bloc.
        C_bloc (str): The name of the C bloc.
        historical_majority (str): Name of majority bloc in historical data.
        historical_minority (str): Name of minority bloc in historical data.
        path (str): File path to an election data file to sample from. Defaults to Cambridge
            elections.
        bloc_to_historical (dict): Dictionary which converts bloc names to historical bloc names.
    """

    def __init__(
        self,
        cohesion_parameters: dict,
        path: Optional[Path] = None,
        W_bloc: Optional[str] = None,
        C_bloc: Optional[str] = None,
        historical_majority: Optional[str] = "W",
        historical_minority: Optional[str] = "C",
        **data,
    ):
        # Call the parent class's __init__ method to handle common parameters
        super().__init__(cohesion_parameters=cohesion_parameters, **data)

        self.historical_majority = historical_majority
        self.historical_minority = historical_minority

        if len(self.slate_to_candidates.keys()) > 2:
            raise UserWarning(
                f"This model currently only supports at two blocs, but you \
                              passed {len(self.slate_to_candidates.keys())}"
            )

        if (W_bloc is None) != (C_bloc is None):
            raise ValueError(
                "Both W_bloc and C_bloc must be provided or not provided. \
                             You have provided only one."
            )

        elif W_bloc is not None and W_bloc == C_bloc:
            raise ValueError("W and C bloc must be distinct.")

        if W_bloc is None:
            self.W_bloc = [
                bloc for bloc, prop in self.bloc_voter_prop.items() if prop >= 0.5
            ][0]
        else:
            self.W_bloc = W_bloc

        if C_bloc is None:
            self.C_bloc = [
                bloc for bloc in self.bloc_voter_prop.keys() if bloc != self.W_bloc
            ][0]
        else:
            self.C_bloc = C_bloc

        self.bloc_to_historical = {
            self.W_bloc: self.historical_majority,
            self.C_bloc: self.historical_minority,
        }

        if path:
            self.path = path
        else:
            BASE_DIR = Path(__file__).resolve().parent
            DATA_DIR = BASE_DIR / "data/"
            self.path = Path(DATA_DIR, "Cambridge_09to17_ballot_types.p")

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
        with open(self.path, "rb") as pickle_file:
            ballot_frequencies = pickle.load(pickle_file)

        cohesion_parameters = {b: self.cohesion_parameters[b][b] for b in self.blocs}

        # compute the number of bloc and crossover voters in each bloc using Huntington Hill
        voter_types = [
            (b, t) for b in list(self.bloc_voter_prop.keys()) for t in ["bloc", "cross"]
        ]

        voter_props = [
            (
                cohesion_parameters[b] * self.bloc_voter_prop[b]
                if t == "bloc"
                else (1 - cohesion_parameters[b]) * self.bloc_voter_prop[b]
            )
            for b, t in voter_types
        ]

        ballots_per_type = dict(
            zip(
                voter_types,
                apportion.compute("huntington", voter_props, number_of_ballots),
            )
        )

        pp_by_bloc = {b: RankProfile() for b in self.blocs}

        for i, bloc in enumerate(self.blocs):
            bloc_voters = ballots_per_type[(bloc, "bloc")]
            cross_voters = ballots_per_type[(bloc, "cross")]
            ballot_pool = [Ballot()] * (bloc_voters + cross_voters)

            # store the opposition bloc
            opp_bloc = self.blocs[(i + 1) % 2]

            # find total number of ballots that start with bloc and opp_bloc
            bloc_first_count = sum(
                [
                    freq
                    for ballot, freq in ballot_frequencies.items()
                    if ballot[0] == self.bloc_to_historical[bloc]
                ]
            )

            opp_bloc_first_count = sum(
                [
                    freq
                    for ballot, freq in ballot_frequencies.items()
                    if ballot[0] == self.bloc_to_historical[opp_bloc]
                ]
            )

            # Compute the pref interval for this bloc
            pref_interval_dict = combine_preference_intervals(
                list(self.pref_intervals_by_bloc[bloc].values()),
                [cohesion_parameters[bloc], 1 - cohesion_parameters[bloc]],
            )

            # compute the relative probabilities of each ballot
            # sorted by ones where the ballot lists the bloc first
            # and those that list the opp first
            prob_ballot_given_bloc_first = {
                ballot: freq / bloc_first_count
                for ballot, freq in ballot_frequencies.items()
                if ballot[0] == self.bloc_to_historical[bloc]
            }

            prob_ballot_given_opp_first = {
                ballot: freq / opp_bloc_first_count
                for ballot, freq in ballot_frequencies.items()
                if ballot[0] == self.bloc_to_historical[opp_bloc]
            }

            bloc_voter_ordering = random.choices(
                list(prob_ballot_given_bloc_first.keys()),
                weights=list(prob_ballot_given_bloc_first.values()),
                k=bloc_voters,
            )
            cross_voter_ordering = random.choices(
                list(prob_ballot_given_opp_first.keys()),
                weights=list(prob_ballot_given_opp_first.values()),
                k=cross_voters,
            )

            # Generate ballots
            for i in range(bloc_voters + cross_voters):
                # Based on first choice, randomly choose
                # ballots weighted by Cambridge frequency
                if i < bloc_voters:
                    bloc_ordering = bloc_voter_ordering[i]
                else:
                    bloc_ordering = cross_voter_ordering[i - bloc_voters]

                # Now turn bloc ordering into candidate ordering
                pl_ordering = list(
                    np.random.choice(
                        list(pref_interval_dict.interval.keys()),
                        len(pref_interval_dict.interval),
                        p=list(pref_interval_dict.interval.values()),
                        replace=False,
                    )
                )
                ordered_bloc_slate = [
                    c for c in pl_ordering if c in self.slate_to_candidates[bloc]
                ]
                ordered_opp_slate = [
                    c for c in pl_ordering if c in self.slate_to_candidates[opp_bloc]
                ]

                # Fill in the bloc slots as determined
                # With the candidate ordering generated with PL
                full_ballot = []
                for b in bloc_ordering:
                    if b == self.bloc_to_historical[bloc]:
                        if ordered_bloc_slate:
                            full_ballot.append(ordered_bloc_slate.pop(0))
                    else:
                        if ordered_opp_slate:
                            full_ballot.append(ordered_opp_slate.pop(0))

                ranking = tuple([frozenset({cand}) for cand in full_ballot])
                ballot_pool[i] = Ballot(ranking=ranking, weight=1)

            pp = RankProfile(ballots=tuple(ballot_pool))
            pp = pp.group_ballots()
            pp_by_bloc[bloc] = pp

        # combine the profiles
        pp = RankProfile()
        for profile in pp_by_bloc.values():
            pp += profile

        if by_bloc:
            return (pp_by_bloc, pp)

        # else return the combined profiles
        else:
            return pp
