from abc import abstractmethod
import math
import numpy as np
from typing import Union, Tuple

from votekit.ballot import Ballot
from votekit.pref_profile import PreferenceProfile
from votekit.pref_interval import PreferenceInterval


# TODO: Change the functionality of this class to allow for sweeping over parameters
# or for generating multiple profiles using different generator models
# maybe break out into `BlocSlateBallotGenerator` and `StdBallotGenerator`
# to make the API clearer
class BallotGenerator:
    """
    Base class for ballot generation models that use the candidate simplex
    (e.g. Plackett-Luce, Bradley-Terry, etc.).

    Args:
        **kwargs: Arbitrary keyword arguments needed for different models.
    """

    def __init__(
        self,
        **kwargs,
    ):
        if "candidates" not in kwargs and "slate_to_candidates" not in kwargs:
            raise ValueError(
                "At least one of candidates or slate_to_candidates must be provided."
            )

        if "candidates" in kwargs:
            self.candidates = kwargs["candidates"]

        if "slate_to_candidates" in kwargs:
            self.slate_to_candidates = kwargs["slate_to_candidates"]
            self.candidates = [
                c for c_list in self.slate_to_candidates.values() for c in c_list
            ]

        nec_parameters = [
            "pref_intervals_by_bloc",
            "cohesion_parameters",
            "bloc_voter_prop",
        ]

        if any(x in kwargs for x in nec_parameters):
            if not all(x in kwargs for x in nec_parameters):
                raise ValueError(
                    f"If one of {nec_parameters} is provided, all must be provided."
                )

            bloc_voter_prop = kwargs["bloc_voter_prop"]
            pref_intervals_by_bloc = kwargs["pref_intervals_by_bloc"]
            cohesion_parameters = kwargs["cohesion_parameters"]

            if round(sum(bloc_voter_prop.values()), 8) != 1.0:
                raise ValueError("Voter proportion for blocs must sum to 1")

            if bloc_voter_prop.keys() != pref_intervals_by_bloc.keys():
                raise ValueError(
                    "Blocs are not the same between bloc_voter_prop and pref_intervals_by_bloc."
                )

            if bloc_voter_prop.keys() != cohesion_parameters.keys():
                raise ValueError(
                    "Blocs are not the same between bloc_voter_prop and cohesion_parameters."
                )

            if pref_intervals_by_bloc.keys() != cohesion_parameters.keys():
                raise ValueError(
                    "Blocs are not the same between pref_intervals_by_bloc and cohesion_parameters."
                )

            for bloc, cohesion_parameter_dict in cohesion_parameters.items():
                if round(sum(cohesion_parameter_dict.values()), 8) != 1.0:
                    raise ValueError(
                        f"Cohesion parameters for bloc {bloc} must sum to 1."
                    )

            self.pref_intervals_by_bloc = pref_intervals_by_bloc
            self.bloc_voter_prop = bloc_voter_prop
            self.blocs = list(self.bloc_voter_prop.keys())
            self.cohesion_parameters = cohesion_parameters

    @classmethod
    def from_params(
        cls,
        slate_to_candidates: dict,
        bloc_voter_prop: dict,
        cohesion_parameters: dict,
        alphas: dict,
        **data,
    ):
        """
        Initializes a ``BallotGenerator`` by constructing preference intervals
        from parameters.

        Args:
            slate_to_candidates (dict): Dictionary whose keys are bloc names and whose
                values are lists of candidate strings that make up the slate.
            bloc_voter_prop (dict): Dictionary whose keys are bloc strings and values are floats
                denoting population share.
            cohesion_parameters (dict): Dictionary mapping of bloc string to dictionary whose
                keys are bloc strings and values are cohesion parameters.
            alphas (dict): Dictionary mapping of bloc string to dictionary whose
                keys are bloc strings and values are alphas for Dirichlet distributions.
            **data: kwargs to be passed to the init method.

        Raises:
            ValueError: If the voter proportion for blocs don't sum to 1.
            ValueError: Blocs are not the same.

        Returns:
            BallotGenerator: Initialized ballot generator.
        """
        if round(sum(bloc_voter_prop.values()), 8) != 1.0:
            raise ValueError("Voter proportion for blocs must sum to 1")

        if slate_to_candidates.keys() != bloc_voter_prop.keys():
            raise ValueError("Blocs are not the same")

        pref_intervals_by_bloc = {}
        for current_bloc in bloc_voter_prop:
            intervals = {}
            for b in bloc_voter_prop:
                interval = PreferenceInterval.from_dirichlet(
                    candidates=slate_to_candidates[b], alpha=alphas[current_bloc][b]
                )
                intervals[b] = interval

            pref_intervals_by_bloc[current_bloc] = intervals

        if "candidates" not in data:
            cands = [cand for cands in slate_to_candidates.values() for cand in cands]
            data["candidates"] = cands

        data["pref_intervals_by_bloc"] = pref_intervals_by_bloc
        data["bloc_voter_prop"] = bloc_voter_prop
        data["cohesion_parameters"] = cohesion_parameters
        data["slate_to_candidates"] = slate_to_candidates

        return cls(**data)

    @abstractmethod
    def generate_profile(
        self, number_of_ballots: int, by_bloc: bool = False
    ) -> Union[PreferenceProfile, Tuple, dict]:
        """
        Generates a ``PreferenceProfile``.

        Args:
            number_of_ballots (int): Number of ballots to generate.
            by_bloc (bool): True if you want the generated profiles returned as a tuple
                ``(pp_by_bloc, pp)``, where ``pp_by_bloc`` is a dictionary with keys = bloc strings
                and values = ``PreferenceProfile`` and ``pp`` is the aggregated profile. False if
                you only want the aggregated profile. Defaults to False.

        Returns:
            Union[PreferenceProfile, Tuple]
        """
        pass

    @staticmethod
    def _round_num(num: float) -> int:
        """
        Rounds up or down a float randomly.

        Args:
            num (float): Number to round.

        Returns:
            int: A whole number.
        """
        rand = np.random.random()
        return math.ceil(num) if rand > 0.5 else math.floor(num)

    @staticmethod
    def ballot_pool_to_profile(ballot_pool, candidates) -> PreferenceProfile:
        """
        Given a list of ballots and candidates, convert them into a ``PreferenceProfile``.

        Args:
            ballot_pool (list): A list of ballots, where each ballot is a tuple
                    of candidates indicating their ranking from top to bottom.
            candidates (list): A list of candidate strings.

        Returns:
            PreferenceProfile: A ``PreferenceProfile`` representing the ballots in the election.
        """
        ranking_counts: dict[tuple, int] = {}
        ballot_list: list[Ballot] = []

        for ranking in ballot_pool:
            tuple_rank = tuple(ranking)
            ranking_counts[tuple_rank] = (
                ranking_counts[tuple_rank] + 1 if tuple_rank in ranking_counts else 1
            )

        for ranking, count in ranking_counts.items():
            rank = tuple([frozenset([cand]) for cand in ranking])
            b = Ballot(ranking=rank, weight=count)
            ballot_list.append(b)

        return PreferenceProfile(ballots=tuple(ballot_list), candidates=candidates)
