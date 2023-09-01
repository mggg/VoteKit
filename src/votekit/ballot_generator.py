from abc import abstractmethod
from functools import reduce
import itertools as it
from fractions import Fraction
import math
import numpy as np
from pathlib import Path
import pickle
import random
from typing import Optional

from .ballot import Ballot
from .pref_profile import PreferenceProfile


class BallotGenerator:
    """
    Base class for ballot generation models
    """

    def __init__(
        self,
        candidates: list,
        *,
        ballot_length: Optional[int] = None,
        pref_interval_by_bloc=None,
        bloc_voter_prop=None,
    ):

        """
         Initializes a Ballot Generator

        Args:
            candidates (list): list of candidates in the election
            ballot_length (Optional[int]): length of ballots to generate.
            Defaults to the length of candidates.
            pref_interval_by_bloc (dict[dict], optional): a mapping of slate to preference interval
            (ex. {race: {candidate : interval length}})
            bloc_voter_prop (dict): a mapping of slate to voter proportions
            (ex. {race: voter proportion}). Defaults to None.

        Raises:
            ValueError: if the voter proportion for blocs don't sum to 1
            ValueError: if preference interval for candidates must sum to 1
            ValueError: slates and blocs are not the same
        """

        self.ballot_length = (
            ballot_length if ballot_length is not None else len(candidates)
        )
        self.candidates = candidates

        if bloc_voter_prop and pref_interval_by_bloc:  # PL, BT, AC, CS
            if round(sum(bloc_voter_prop.values())) != 1:
                raise ValueError("Voter proportion for blocs must sum to 1")
            for interval in pref_interval_by_bloc.values():
                if round(sum(interval.values())) != 1:
                    raise ValueError("Preference interval for candidates must sum to 1")
            if bloc_voter_prop.keys() != pref_interval_by_bloc.keys():
                raise ValueError("slates and blocs are not the same")

            self.pref_interval_by_bloc = pref_interval_by_bloc
            self.bloc_voter_prop = bloc_voter_prop

    @classmethod
    def from_params(
        cls,
        slate_to_candidates: dict,
        bloc_voter_prop: dict,
        cohesion: dict,
        alphas: dict,
        **data,
    ):
        """
        Initializes a Ballot Generator by constructing a preference interval from parameters
        (the prior parameters will be overwrittern)

        Args:
            slate_to_candidate (dict): a mapping of slate to candidates
            (ex. {race: [candidate]})
            bloc_voter_prop (dict): a mapping of the percentage of total voters per bloc
            (ex. {race: 0.5})
            cohesion (dict): cohension factor for each bloc
            alphas (dict): alpha for the dirchlet distribution of each bloc

        Raises:
            ValueError: if the voter proportion for blocs don't sum to 1
            ValueError: slates and blocs are not the same

        Returns:
            BallotGenerator: initialized ballot generator
        """

        if sum(bloc_voter_prop.values()) != 1.0:
            raise ValueError(
                f"bloc proportions ({bloc_voter_prop.values()}) do not equal 1"
            )
        if slate_to_candidates.keys() != bloc_voter_prop.keys():
            raise ValueError("slates and blocs are not the same")

        def _construct_preference_interval(
            alphas: dict, cohesion: int, bloc: str, slate_to_cands: dict
        ) -> dict:
            intervals = {}

            for group, alpha in alphas.items():
                num_cands = len(slate_to_cands[group])
                alpha = [alpha] * num_cands
                probs = list(np.random.default_rng().dirichlet(alpha=alpha))
                for prob, cand in zip(probs, slate_to_cands[group]):
                    if group == bloc:  # e.g W for W cands
                        pi = cohesion
                    else:  # e.g W for POC cands
                        pi = 1 - cohesion
                    intervals[cand] = pi * prob

            return intervals

        interval_by_bloc = {}
        for bloc in bloc_voter_prop:
            interval = _construct_preference_interval(
                alphas[bloc], cohesion[bloc], bloc, slate_to_candidates
            )
            interval_by_bloc[bloc] = interval

        if "candidates" not in data:
            cands = list(
                {cand for cands in slate_to_candidates.values() for cand in cands}
            )
            data["candidates"] = cands

        if "pref_interval_by_bloc" not in data:
            data["pref_interval_by_bloc"] = interval_by_bloc

        if "bloc_voter_prop" not in data:
            data["bloc_voter_prop"] = bloc_voter_prop

        generator = cls(**data)

        if isinstance(generator, (AlternatingCrossover, CambridgeSampler)):
            generator.slate_to_candidates = slate_to_candidates

        return generator

    @abstractmethod
    def generate_profile(self, number_of_ballots: int) -> PreferenceProfile:
        """
        Generates a preference profile

        Args:
            number_of_ballots (int): number of ballots to generate
        Returns:
            PreferenceProfile: a generated preference profiles
        """
        pass

    @staticmethod
    def round_num(num: float) -> int:
        """
        rounds up or down a float randomly

        Args:
            num (float): number to round

        Returns:
            int: a whole number
        """
        rand = np.random.random()
        return math.ceil(num) if rand > 0.5 else math.floor(num)

    @staticmethod
    def ballot_pool_to_profile(ballot_pool, candidates) -> PreferenceProfile:
        """
        Given a list of ballots and candidates, convert them into a Preference Profile

        Args:
            ballot_pool (list of tuple): a list of ballots, with tuple as their ranking
            candidates (list): a list of candidates

        Returns:
            PreferenceProfile: a preference profile representing the ballots in the election
        """
        ranking_counts: dict[tuple, int] = {}
        ballot_list: list[Ballot] = []

        for ranking in ballot_pool:
            tuple_rank = tuple(ranking)
            ranking_counts[tuple_rank] = (
                ranking_counts[tuple_rank] + 1 if tuple_rank in ranking_counts else 1
            )

        for ranking, count in ranking_counts.items():
            rank = [set([cand]) for cand in ranking]
            b = Ballot(ranking=rank, weight=Fraction(count))
            ballot_list.append(b)

        return PreferenceProfile(ballots=ballot_list, candidates=candidates)


class BallotSimplex(BallotGenerator):
    """
    Base class for ballot generation models
    """

    def __init__(
        self, alpha: Optional[float] = None, point: Optional[dict] = None, **data
    ):
        """
        Initializes a Ballot Simplex model

        Args:
            alpha (float, optional): alpha parameter for ballot simplex. Defaults to None.
            point (dict, optional): a point in the ballot simplex,
            with candidate as keys and electoral support as values. Defaults to None.

        Raises:
            AttributeError: if point and alpha are not initialized
        """
        if alpha is None and point is None:
            raise AttributeError("point or alpha must be initialized")
        self.alpha = alpha
        if alpha == float("inf"):
            self.alpha = 1e20
        if alpha == 0:
            self.alpha = 1e-10
        self.point = point
        super().__init__(**data)

    @classmethod
    def from_point(cls, point: dict, **data):
        """
        Initializes a Ballot Simplex model from a point in the dirichlet distribution

        Args:
            point (dict): a mapping of candidate to candidate support

        Raises:
            ValueError: if the candidate support does not sum to 1

        Returns:
            BallotSimplex: initialized from point
        """
        if sum(point.values()) != 1.0:
            raise ValueError(
                f"probability distribution from point ({point.values()}) does not sum to 1"
            )
        return cls(point=point, **data)

    @classmethod
    def from_alpha(cls, alpha: float, **data):
        """
        Initializes a Ballot Simplex model from an alpha value for the dirichlet distribution

        Args:
            alpha (float): an alpha parameter for the dirichlet distribution

        Returns:
            BallotSimplex: initialized from alpha
        """

        return cls(alpha=alpha, **data)

    def generate_profile(self, number_of_ballots) -> PreferenceProfile:
        perm_set = it.permutations(self.candidates, self.ballot_length)

        perm_rankings = [list(value) for value in perm_set]

        if self.alpha is not None:
            draw_probabilities = list(
                np.random.default_rng().dirichlet([self.alpha] * len(perm_rankings))
            )
        elif self.point:
            # calculates probabilities for each ranking
            # using probability distribution for candidate support
            draw_probabilities = [
                reduce(
                    lambda prod, cand: prod * self.point[cand] if self.point else 0,
                    ranking,
                    1.0,
                )
                for ranking in perm_rankings
            ]
            draw_probabilities = [
                prob / sum(draw_probabilities) for prob in draw_probabilities
            ]

        ballot_pool = []

        for _ in range(number_of_ballots):
            index = np.random.choice(
                range(len(perm_rankings)), 1, p=draw_probabilities
            )[0]
            ballot_pool.append(perm_rankings[index])

        return self.ballot_pool_to_profile(ballot_pool, self.candidates)


class ImpartialCulture(BallotSimplex):
    """
    Impartial Culture model (child class of BallotSimplex)
    with an alpha value of 1e10 (should be infinity theoretically)
    """

    def __init__(self, **data):
        super().__init__(alpha=float("inf"), **data)


class ImpartialAnonymousCulture(BallotSimplex):
    """
    Impartial Anonymous Culture model (child class of BallotSimplex)
    with an alpha value of 1
    """

    def __init__(self, **data):
        super().__init__(alpha=1, **data)


class PlackettLuce(BallotGenerator):
    """
    Plackett Luce Ballot Generation Model (child class of BallotGenerator)
    """

    def __init__(self, **data):
        """
        Initializes Plackett Luce Ballot Generation Model

        Args:
            pref_interval_by_bloc (dict): a mapping of slate to preference interval
            (ex. {race: {candidate : interval length}})
            bloc_voter_prop (dict): a mapping of slate to voter proportions
            (ex. {race: voter proportion})
        """

        # Call the parent class's __init__ method to handle common parameters
        super().__init__(**data)

    def generate_profile(self, number_of_ballots) -> PreferenceProfile:
        ballot_pool = []

        for bloc in self.bloc_voter_prop.keys():
            # number of voters in this bloc
            num_ballots = self.round_num(number_of_ballots * self.bloc_voter_prop[bloc])
            pref_interval_dict = self.pref_interval_by_bloc[bloc]
            # creates the interval of probabilities for candidates supported by this block
            cand_support_vec = [pref_interval_dict[cand] for cand in self.candidates]

            for _ in range(num_ballots):
                # generates ranking based on probability distribution of candidate support
                ballot = list(
                    np.random.choice(
                        self.candidates,
                        self.ballot_length,
                        p=cand_support_vec,
                        replace=False,
                    )
                )

                ballot_pool.append(ballot)

        pp = self.ballot_pool_to_profile(
            ballot_pool=ballot_pool, candidates=self.candidates
        )
        return pp


class BradleyTerry(BallotGenerator):
    """
    Bradley Terry Ballot Generation Model (child class of BallotGenerator)
    """

    def __init__(self, **data):
        """
        Initializes a Bradley Terry Ballot Generation Model

        Args:
            pref_interval_by_bloc (dict): a mapping of slate to preference interval
            (ex. {race: {candidate : interval length}})
            bloc_voter_prop (dict): a mapping of slate to voter proportions
            (ex. {race: voter proportion})
        """

        # Call the parent class's __init__ method to handle common parameters
        super().__init__(**data)

    def _calc_prob(self, permutations: list[tuple], cand_support_dict: dict) -> dict:
        """
        given a list of rankings and the preference interval,
        calculates the probability of observing each ranking

        Args:
            permutations (list[tuple]): a list of permuted rankings
            cand_support_dict (dict): a mapping from candidate to their
            support (preference interval)

        Returns:
            dict: a mapping of the rankings to their probability
        """
        ranking_to_prob = {}
        for ranking in permutations:
            prob = 1
            for i in range(len(ranking)):
                cand_i = ranking[i]
                greater_cand_support = cand_support_dict[cand_i]
                for j in range(i + 1, len(ranking)):
                    cand_j = ranking[j]
                    cand_support = cand_support_dict[cand_j]
                    prob *= greater_cand_support / (greater_cand_support + cand_support)
            ranking_to_prob[ranking] = prob
        return ranking_to_prob

    def generate_profile(self, number_of_ballots) -> PreferenceProfile:

        permutations = list(it.permutations(self.candidates, self.ballot_length))
        ballot_pool: list[list] = []

        for bloc in self.bloc_voter_prop.keys():
            num_ballots = self.round_num(number_of_ballots * self.bloc_voter_prop[bloc])
            pref_interval_dict = self.pref_interval_by_bloc[bloc]

            ranking_to_prob = self._calc_prob(
                permutations=permutations, cand_support_dict=pref_interval_dict
            )

            indices = range(len(ranking_to_prob))
            prob_distrib = list(ranking_to_prob.values())
            prob_distrib = [float(p) / sum(prob_distrib) for p in prob_distrib]

            ballots_indices = np.random.choice(
                indices,
                num_ballots,
                p=prob_distrib,
                replace=True,
            )

            rankings = list(ranking_to_prob.keys())
            ballots = [rankings[i] for i in ballots_indices]

            ballot_pool = ballot_pool + ballots

        pp = self.ballot_pool_to_profile(
            ballot_pool=ballot_pool, candidates=self.candidates
        )
        return pp


class AlternatingCrossover(BallotGenerator):
    """
    Alternating Crossover Ballot Generation Model (child class of BallotGenerator)
    """

    def __init__(
        self,
        slate_to_candidates=None,
        bloc_crossover_rate=None,
        **data,
    ):
        """
        Initializes Alternating Crossover Ballot Generation Model

        Args:
            slate_to_candidate (dict): a mapping of slate to candidates
            (ex. {race: [candidate]})
            pref_interval_by_bloc (dict): a mapping of bloc to preference interval
            (ex. {race: {candidate : interval length}})
            bloc_voter_prop (dict): a mapping of the percentage of total voters per bloc
            (ex. {race: 0.5})
            bloc_crossover_rate (dict): a mapping of percentage of crossover voters per bloc
            (ex. {race: {other_race: 0.5}})
        """
        # Call the parent class's __init__ method to handle common parameters
        super().__init__(**data)

        self.slate_to_candidates = slate_to_candidates
        self.bloc_crossover_rate = bloc_crossover_rate

    def generate_profile(self, number_of_ballots) -> PreferenceProfile:

        ballot_pool = []

        for bloc in self.bloc_voter_prop.keys():

            num_ballots = self.round_num(number_of_ballots * self.bloc_voter_prop[bloc])
            crossover_dict = self.bloc_crossover_rate[bloc]
            pref_interval_dict = self.pref_interval_by_bloc[bloc]

            # generates crossover ballots from each bloc (allowing for more than two blocs)
            for opposing_slate in crossover_dict.keys():
                crossover_rate = crossover_dict[opposing_slate]
                num_crossover_ballots = self.round_num(crossover_rate * num_ballots)

                opposing_cands = self.slate_to_candidates[opposing_slate]
                bloc_cands = self.slate_to_candidates[bloc]

                for _ in range(num_crossover_ballots):
                    pref_for_opposing = [
                        pref_interval_dict[cand] for cand in opposing_cands
                    ]
                    # convert to probability distribution
                    pref_for_opposing = [
                        p / sum(pref_for_opposing) for p in pref_for_opposing
                    ]

                    pref_for_bloc = [pref_interval_dict[cand] for cand in bloc_cands]
                    # convert to probability distribution
                    pref_for_bloc = [p / sum(pref_for_bloc) for p in pref_for_bloc]

                    bloc_cands = list(
                        np.random.choice(
                            bloc_cands,
                            p=pref_for_bloc,
                            size=len(bloc_cands),
                            replace=False,
                        )
                    )
                    opposing_cands = list(
                        np.random.choice(
                            opposing_cands,
                            size=len(opposing_cands),
                            p=pref_for_opposing,
                            replace=False,
                        )
                    )

                    # alternate the bloc and opposing bloc candidates to create crossover ballots
                    if bloc != opposing_slate:  # alternate
                        ballot = [
                            item
                            for pair in zip(opposing_cands, bloc_cands)
                            for item in pair
                            if item is not None
                        ]

                    # check that ballot_length is shorter than total number of cands
                    ballot_pool.append(ballot)

                # Bloc ballots
                for _ in range(num_ballots - num_crossover_ballots):
                    ballot = bloc_cands + opposing_cands
                    ballot_pool.append(ballot)

        pp = self.ballot_pool_to_profile(
            ballot_pool=ballot_pool, candidates=self.candidates
        )
        return pp


class OneDimSpatial(BallotGenerator):
    def generate_profile(self, number_of_ballots) -> PreferenceProfile:
        candidate_position_dict = {c: np.random.normal(0, 1) for c in self.candidates}
        voter_positions = np.random.normal(0, 1, number_of_ballots)

        ballot_pool = []

        for vp in voter_positions:
            distance_dict = {
                c: abs(v - vp) for c, v, in candidate_position_dict.items()
            }
            candidate_order = sorted(distance_dict, key=distance_dict.__getitem__)
            ballot_pool.append(candidate_order)

        return self.ballot_pool_to_profile(ballot_pool, self.candidates)


class CambridgeSampler(BallotGenerator):
    """
    Cambridge Sampler Ballot Generation  model (child class of BallotGenerator)
    """

    def __init__(
        self,
        slate_to_candidates=None,
        bloc_crossover_rate=None,
        path: Optional[Path] = None,
        **data,
    ):
        """
        Initializes Cambridge Sampler Ballot Generation  model

        Args:
            slate_to_candidate (dict): a mapping of slate to candidates
            (ex. {race: [candidate]})
            pref_interval_by_bloc (dict): a mapping of bloc to preference interval
            (ex. {race: {candidate : interval length}})
            path (Optional[Path]): a path to an election data file to sample from.
            Defaults to Cambridge elections.
        """

        # Call the parent class's __init__ method to handle common parameters
        super().__init__(**data)

        self.slate_to_candidates = slate_to_candidates
        self.bloc_crossover_rate = bloc_crossover_rate

        if path:
            self.path = path
        else:
            BASE_DIR = Path(__file__).resolve().parent
            DATA_DIR = BASE_DIR / "data/"
            self.path = Path(DATA_DIR, "Cambridge_09to17_ballot_types.p")

    def generate_profile(self, number_of_ballots: int) -> PreferenceProfile:

        with open(self.path, "rb") as pickle_file:
            ballot_frequencies = pickle.load(pickle_file)

        ballot_pool = []

        blocs = self.slate_to_candidates.keys()
        for bloc in blocs:
            # compute the number of voters in this bloc
            bloc_voters = self.round_num(self.bloc_voter_prop[bloc] * number_of_ballots)

            # store the opposition bloc
            opp_bloc = next(iter(set(blocs).difference(set(bloc))))

            # compute how many ballots list a bloc candidate first
            bloc_first_count = sum(
                [
                    freq
                    for ballot, freq in ballot_frequencies.items()
                    if ballot[0] == bloc
                ]
            )

            # Compute the pref interval for this bloc
            pref_interval_dict = self.pref_interval_by_bloc[bloc]

            # compute the relative probabilities of each ballot
            # sorted by ones where the ballot lists the bloc first
            # and those that list the opp first
            prob_ballot_given_bloc_first = {
                ballot: freq / bloc_first_count
                for ballot, freq in ballot_frequencies.items()
                if ballot[0] == bloc
            }
            prob_ballot_given_opp_first = {
                ballot: freq / bloc_first_count
                for ballot, freq in ballot_frequencies.items()
                if ballot[0] == opp_bloc
            }

            # Generate ballots
            for _ in range(bloc_voters):
                # Randomly choose first choice based off
                # bloc crossover rate
                first_choice = np.random.choice(
                    [bloc, opp_bloc],
                    p=[
                        1 - self.bloc_crossover_rate[bloc][opp_bloc],
                        self.bloc_crossover_rate[bloc][opp_bloc],
                    ],
                )
                # Based on first choice, randomly choose
                # ballots weighted by Cambridge frequency
                if first_choice == bloc:
                    bloc_ordering = random.choices(
                        list(prob_ballot_given_bloc_first.keys()),
                        weights=list(prob_ballot_given_bloc_first.values()),
                        k=1,
                    )[0]
                else:
                    bloc_ordering = random.choices(
                        list(prob_ballot_given_opp_first.keys()),
                        weights=list(prob_ballot_given_opp_first.values()),
                        k=1,
                    )[0]

                # Now turn bloc ordering into candidate ordering
                pl_ordering = list(
                    np.random.choice(
                        list(pref_interval_dict.keys()),
                        self.ballot_length,
                        p=list(pref_interval_dict.values()),
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
                    if b == bloc:
                        if ordered_bloc_slate:
                            full_ballot.append(ordered_bloc_slate.pop(0))
                    else:
                        if ordered_opp_slate:
                            full_ballot.append(ordered_opp_slate.pop(0))

                ballot_pool.append(tuple(full_ballot))

        pp = self.ballot_pool_to_profile(
            ballot_pool=ballot_pool, candidates=self.candidates
        )
        return pp
