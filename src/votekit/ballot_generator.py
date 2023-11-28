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
import apportionment.methods as apportion  # type: ignore

from .ballot import Ballot
from .pref_profile import PreferenceProfile


class BallotGenerator:
    """
    Base class for ballot generation models that use the candidate simplex
    (e.g. Plackett-Luce, Bradley-Terry, etc.).

    **Attributes**

    `candidates`
    :   list of candidates in the election.

    `ballot_length`
    :   (optional) length of ballots to generate. Defaults to the length of
        `candidates`.

    `pref_interval_by_bloc`
    :   dictionary mapping of bloc to preference interval.
        (ex. {bloc: {candidate : interval length}}). Defaults to None.

    `bloc_voter_prop`
    :   dictionary mapping of bloc to voter proportions (ex. {bloc: voter proportion}).
        Defaults to None.

    ???+ note
        * Voter proportion for blocs must sum to 1.
        * Preference interval for candidates must sum to 1.
        * Must have same blocs in `pref_interval_by_bloc` and `bloc_voter_prop`.

    **Methods**
    """

    def __init__(
        self,
        candidates: list,
        *,
        ballot_length: Optional[int] = None,
        pref_interval_by_bloc=None,
        bloc_voter_prop=None,
    ):
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
        Initializes a BallotGenerator by constructing a preference interval
        from parameters; the prior parameters (if inputted) will be overwritten.

        Args:
            slate_to_candidates (dict): A mapping of blocs to candidates
                (ex. {bloc: [candidate]})
            bloc_voter_prop (dict): A mapping of the percentage of total voters
                 per bloc (ex. {bloc: 0.7})
            cohesion (dict): Cohension factor for each bloc (ex. {bloc: .9})
            alphas (dict): Alpha for the Dirichlet distribution of each bloc
                            (ex. {bloc: {bloc: 1, opposing_bloc: 1/2}}).

        Raises:
            ValueError: If the voter proportion for blocs don't sum to 1.
            ValueError: Blocs are not the same.

        Returns:
            (BallotGenerator): Initialized ballot generator.

        ???+ note
            * Voter proportion for blocs must sum to 1.
            * Each cohesion parameter must be in the interval [0,1].
            * Dirichlet parameters are in the interval $(0,\infty)$.
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
            generator.cohesion_parameters = cohesion

            # rename blocs to match historical data
            if isinstance(generator, CambridgeSampler):
                generator._rename_blocs()

        return generator

    @abstractmethod
    def generate_profile(self, number_of_ballots: int) -> PreferenceProfile:
        """
        Generates a `PreferenceProfile`.

        Args:
            number_of_ballots (int): Number of ballots to generate.

        Returns:
            (PreferenceProfile): A generated `PreferenceProfile`.
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
        Given a list of ballots and candidates, convert them into a `PreferenceProfile.`

        Args:
            ballot_pool (list of tuple): A list of ballots, where each ballot is a tuple
                    of candidates indicating their ranking from top to bottom.
            candidates (list): A list of candidates.

        Returns:
            (PreferenceProfile): A PreferenceProfile representing the ballots in the election.
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
    Base class for ballot generation models that use the ballot simplex
    (e.g. ImpartialCulture, ImpartialAnonymousCulture).

    **Attributes**

    `alpha`
    :   (float) alpha parameter for ballot simplex. Defaults to None.

    `point`
    :   dictionary representing a point in the ballot simplex with candidate as
        keys and electoral support as values. Defaults to None.

    ???+ note

        Point or alpha arguments must be included to initialize.

    **Methods**
    """

    def __init__(
        self, alpha: Optional[float] = None, point: Optional[dict] = None, **data
    ):
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
        Initializes a Ballot Simplex model from a point in the Dirichlet distribution.

        Args:
            point (dict): A mapping of candidate to candidate support.

        Raises:
            ValueError: If the candidate support does not sum to 1.

        Returns:
            (BallotSimplex): Initialized from point.
        """
        if sum(point.values()) != 1.0:
            raise ValueError(
                f"probability distribution from point ({point.values()}) does not sum to 1"
            )
        return cls(point=point, **data)

    @classmethod
    def from_alpha(cls, alpha: float, **data):
        """
        Initializes a Ballot Simplex model from an alpha value for the Dirichlet
        distribution.

        Args:
            alpha (float): An alpha parameter for the Dirichlet distribution.

        Returns:
            (BallotSimplex): Initialized from alpha.
        """

        return cls(alpha=alpha, **data)

    def generate_profile(self, number_of_ballots) -> PreferenceProfile:
        """
        Generates a PreferenceProfile from the ballot simplex.
        """

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
    Impartial Culture model with an alpha value of 1e10 (should be infinity theoretically).
    This model is uniform on all linear rankings.


    **Attributes**

    `candidates`
    : (list) a list of candidates

    `alpha`
    :   (float) alpha parameter for ballot simplex. Defaults to None.

    `point`
    :   dictionary representing a point in the ballot simplex with candidate as
        keys and electoral support as values. Defaults to None.



    **Methods**

    See `BallotSimplex` object.

    ???+ note

        Point or alpha arguments must be included to initialize. For details see
        `BallotSimplex` and `BallotGenerator` object.
    """

    def __init__(self, **data):
        super().__init__(alpha=float("inf"), **data)


class ImpartialAnonymousCulture(BallotSimplex):
    """
    Impartial Anonymous Culture model with an alpha value of 1. This model choose uniformly
        from among all distributions on full linear rankings, and then draws according to the
        chosen distribution.

    **Attributes**

    `candidates`
    : (list) a list of candidates

    `alpha`
    :   (float) alpha parameter for ballot simplex. Defaults to None.

    `point`
    :   dictionary representing a point in the ballot simplex with candidate as
        keys and electoral support as values. Defaults to None.

    **Methods**

    See `BallotSimplex` base class.

    ???+ note

        Point or alpha arguments must be included to initialize. For details see
        `BallotSimplex` and `BallotGenerator` object.
    """

    def __init__(self, **data):
        super().__init__(alpha=1, **data)


class PlackettLuce(BallotGenerator):
    """
    Class for generating ballots using a Plackett-Luce model. This model samples without
    replacement from a preference interval. Can be initialized with an interval or can be
    constructed with the Dirichlet distribution using the `from_params` method in the
    `BallotGenerator` class.

    **Attributes**

    `candidates`
    : a list of candidates.

    `pref_interval_by_bloc`
    :   dictionary mapping of bloc to preference interval.
        (ex. {bloc: {candidate : interval length}}).

    `bloc_voter_prop`
    :   dictionary mapping of bloc to voter proportions (ex. {bloc: proportion}).

    **Methods**

    See `BallotGenerator` base class
    """

    def __init__(self, **data):
        # Call the parent class's __init__ method to handle common parameters
        super().__init__(**data)

    def generate_profile(self, number_of_ballots) -> PreferenceProfile:
        ballot_pool = []

        # the number of ballots per bloc is determined by Huntington-Hill apportionment
        blocs = list(self.bloc_voter_prop.keys())
        bloc_props = list(self.bloc_voter_prop.values())
        ballots_per_block = dict(
            zip(blocs, apportion.compute("huntington", bloc_props, number_of_ballots))
        )

        for bloc in self.bloc_voter_prop.keys():
            # number of voters in this bloc
            num_ballots = ballots_per_block[bloc]

            pref_interval_dict = self.pref_interval_by_bloc[bloc]

            # finds candidates with non-zero preference
            non_zero_cands = [
                cand for cand, pref in pref_interval_dict.items() if pref > 0
            ]
            # creates the interval of probabilities for candidates supported by this block
            cand_support_vec = [pref_interval_dict[cand] for cand in non_zero_cands]

            for _ in range(num_ballots):
                # generates ranking based on probability distribution of non candidate support
                non_zero_ranking = list(
                    np.random.choice(
                        non_zero_cands,
                        len(non_zero_cands),
                        p=cand_support_vec,
                        replace=False,
                    )
                )

                ranking = [{cand} for cand in non_zero_ranking]

                # add zero support candidates to end as tie
                zero_cands = set(self.candidates).difference(non_zero_cands)
                if len(zero_cands) > 0:
                    ranking.append(zero_cands)

                ballot_pool.append(Ballot(ranking=ranking, weight=Fraction(1, 1)))

        pp = PreferenceProfile(ballots=ballot_pool)
        pp.condense_ballots()
        return pp


class BradleyTerry(BallotGenerator):
    """
    Class for generating ballots using a Bradley-Terry model. The probability of sampling
    the ranking $X>Y>Z$ is $P(X>Y)*P(X>Z)*P(Y>Z)$. These individual probabilities are based
    on the preference interval: $P(X>Y) = x/(x+y)$. Can be initialized with an interval
    or can be constructed with the Dirichlet distribution using the `from_params` method in the
    `BallotGenerator` class.

    **Attributes**

    `candidates`
    : a list of candidates.

    `pref_interval_by_bloc`
    :   dictionary mapping of slate to preference interval
        (ex. {race: {candidate : interval length}}).

    `bloc_voter_prop`
    :   dictionary mapping of slate to voter proportions (ex. {race: voter proportion}).

    **Methods**

    See `BallotGenerator` base class.
    """

    def __init__(self, **data):
        # Call the parent class's __init__ method to handle common parameters
        super().__init__(**data)

    def _calc_prob(self, permutations: list[tuple], cand_support_dict: dict) -> dict:
        """
        given a list of rankings and the preference interval, \
        calculates the probability of observing each ranking

        Args:
            permutations (list[tuple]): a list of permuted rankings
            cand_support_dict (dict): a mapping from candidate to their \
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
        ballot_pool: list[Ballot] = []

        # the number of ballots per bloc is determined by Huntington-Hill apportionment
        blocs = list(self.bloc_voter_prop.keys())
        bloc_props = list(self.bloc_voter_prop.values())
        ballots_per_block = dict(
            zip(blocs, apportion.compute("huntington", bloc_props, number_of_ballots))
        )

        for bloc in self.bloc_voter_prop.keys():
            num_ballots = ballots_per_block[bloc]

            pref_interval_dict = self.pref_interval_by_bloc[bloc]
            # compute non-zero pref candidates
            non_zero_pref_dict = {
                cand: prop for cand, prop in pref_interval_dict.items() if prop > 0
            }
            non_zero_cands = non_zero_pref_dict.keys()
            zero_cands = set(self.candidates).difference(non_zero_cands)

            # all possible rankings of non zero candidates
            permutations = list(it.permutations(non_zero_cands, len(non_zero_cands)))

            # compute the prob of each ranking given bloc support
            ranking_to_prob = self._calc_prob(
                permutations=permutations, cand_support_dict=non_zero_pref_dict
            )

            # numpy can only sample from 1D arrays, so we sample the indices instead of rankings
            rankings = list(ranking_to_prob.keys())
            indices = range(len(rankings))
            probs = list(ranking_to_prob.values())

            # create distribution to sample ballots from
            normalizing_constant = sum(probs)
            prob_distrib = [float(p) / normalizing_constant for p in probs]

            # sample ballots
            for _ in range(num_ballots):
                index = list(
                    np.random.choice(
                        indices,
                        1,
                        p=prob_distrib,
                    )
                )[0]

                # convert index to ranking
                ranking = [{cand} for cand in rankings[index]]

                # add any zero candidates as ties
                if len(zero_cands) > 0:
                    ranking.append(zero_cands)

                ballot = Ballot(ranking=ranking, weight=Fraction(1, 1))
                ballot_pool.append(ballot)

        # pp = self.ballot_pool_to_profile(
        #     ballot_pool=ballot_pool, candidates=self.candidates
        # )
        pp = PreferenceProfile(ballots=ballot_pool)
        pp.condense_ballots()
        return pp


class AlternatingCrossover(BallotGenerator):
    """
    Class for Alternating Crossover style of generating ballots.
    AC assumes that voters either rank all of their own blocs candidates above the other bloc,
    or the voters "crossover" and rank a candidate from the other bloc first, then alternate
    between candidates from their own bloc and the opposing.
    Should only be used when there are two blocs.

    Can be initialized with an interval or can be
    constructed with the Dirichlet distribution using the `from_params` method in the
    `BallotGenerator` class.

    **Attributes**

    `pref_interval_by_bloc`
    :   dictionary mapping of slate to preference interval. Preference interval should
        include all candidates regardless of which bloc they are from.
        (ex. {bloc: {candidate : interval length}}).

    `bloc_voter_prop`
    :   dictionary mapping of slate to voter proportions (ex. {bloc: voter proportion}).

    `slate_to_candidates`
    :   dictionary mapping of slate to candidates (ex. {bloc: [candidate1, candidate2]}).

    `cohesion_parameters`
    :   dictionary mapping of bloc to cohesion parameter. A parameter of .6 means voters vote
        in bloc 60% of time (ex. {bloc: .6}).

    **Methods**

    See `BallotGenerator` base class.
    """

    def __init__(
        self,
        slate_to_candidates: dict = {},
        cohesion_parameters: dict = {},
        **data,
    ):
        # Call the parent class's __init__ method to handle common parameters
        super().__init__(**data)

        self.slate_to_candidates = slate_to_candidates
        self.cohesion_parameters = cohesion_parameters

        for bloc, pref_interval in self.pref_interval_by_bloc.items():
            if 0 in pref_interval.values():
                raise ValueError(
                    "In AC model, all candidates must have non-zero preference."
                )

    def generate_profile(self, number_of_ballots) -> PreferenceProfile:
        ballot_pool = []

        # compute the number of bloc and crossover voters in each bloc using Huntington Hill
        voter_types = [
            (b, type) for b in self.bloc_voter_prop.keys() for type in ["bloc", "cross"]
        ]

        voter_props = [
            self.cohesion_parameters[b] * self.bloc_voter_prop[b]
            if t == "bloc"
            else (1 - self.cohesion_parameters[b]) * self.bloc_voter_prop[b]
            for b, t in voter_types
        ]

        ballots_per_type = dict(
            zip(
                voter_types,
                apportion.compute("huntington", voter_props, number_of_ballots),
            )
        )

        for bloc in self.bloc_voter_prop.keys():
            num_bloc_ballots = ballots_per_type[(bloc, "bloc")]
            num_cross_ballots = ballots_per_type[(bloc, "cross")]

            pref_interval_dict = self.pref_interval_by_bloc[bloc]

            opposing_slate = list(set(self.bloc_voter_prop.keys()).difference([bloc]))[
                0
            ]
            opposing_cands = self.slate_to_candidates[opposing_slate]
            bloc_cands = self.slate_to_candidates[bloc]

            pref_for_opposing = [pref_interval_dict[cand] for cand in opposing_cands]
            # convert to probability distribution
            pref_for_opposing = [p / sum(pref_for_opposing) for p in pref_for_opposing]

            pref_for_bloc = [pref_interval_dict[cand] for cand in bloc_cands]
            # convert to probability distribution
            pref_for_bloc = [p / sum(pref_for_bloc) for p in pref_for_bloc]

            for i in range(num_cross_ballots + num_bloc_ballots):
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
                        p=pref_for_opposing,
                        size=len(opposing_cands),
                        replace=False,
                    )
                )

                if i < num_cross_ballots:
                    # alternate the bloc and opposing bloc candidates to create crossover ballots
                    ranking = [
                        {cand}
                        for pair in zip(opposing_cands, bloc_cands)
                        for cand in pair
                    ]
                else:
                    ranking = [{c} for c in bloc_cands] + [{c} for c in opposing_cands]

                ballot = Ballot(ranking=ranking, weight=Fraction(1, 1))
                ballot_pool.append(ballot)

        pp = PreferenceProfile(ballots=ballot_pool, candidates=self.candidates)
        pp.condense_ballots()
        return pp


class OneDimSpatial(BallotGenerator):
    """
    1-D spatial model for ballot generation. Assumes the candidates are normally distributed on
    the real line. Then voters are also normally distributed, and vote based on Euclidean distance
    to the candidates.

    **Attributes**
    `candidates`
        : a list of candidates.

    See `BallotGenerator` base class.

    **Methods**

    See `BallotGenerator` base class.
    """

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
    Class for generating ballots based on historical RCV elections occurring
    in Cambridge. Alternative election data can be used if specified. Assumes that there are two
    blocs, a majority and a minority bloc, and determines this based on the bloc_voter_prop attr.

    Based on cohesion parameters, decides if a voter casts their top choice within their bloc
    or in the opposing bloc. Then uses historical data; given their first choice, choose a
    ballot type from the historical distribution.


    **Attributes**

    `slate_to_candidates`
    :   dictionary mapping of slate to candidates (ex. {bloc: [candidate]}).

    `bloc_voter_prop`
    :   dictionary mapping of bloc to voter proportions (ex. {bloc: voter proportion}).
        Defaults to None.

    `cohesion_parameters`
    :   dictionary mapping of slate to cohesion level (ex. {bloc: .7}).

    `pref_interval_by_bloc`
    :   dictionary mapping of bloc to preference interval
        (ex. {bloc: {candidate : interval length}}).

    `historical_majority`
    : name of majority bloc in historical data, defaults to W for Cambridge.

    `historical_minority`
    : name of minority bloc in historical data, defaults to C for Cambridge.

    `path`
    :   file path to an election data file to sample from. Defaults to Cambridge elections.

    **Methods**

    See `BallotGenerator` base class.
    """

    def __init__(
        self,
        slate_to_candidates: dict = {},
        cohesion_parameters: dict = {},
        path: Optional[Path] = None,
        historical_majority: Optional[str] = "W",
        historical_minority: Optional[str] = "C",
        **data,
    ):
        # Call the parent class's __init__ method to handle common parameters
        super().__init__(**data)

        self.slate_to_candidates = slate_to_candidates
        self.cohesion_parameters = cohesion_parameters
        self.historical_majority = historical_majority
        self.historical_minority = historical_minority

        # changing names to match historical data, if statement handles generating from_params
        # only want to run this now if generating from init
        if len(self.cohesion_parameters) > 0:
            self._rename_blocs()

        if path:
            self.path = path
        else:
            BASE_DIR = Path(__file__).resolve().parent
            DATA_DIR = BASE_DIR / "data/"
            self.path = Path(DATA_DIR, "Cambridge_09to17_ballot_types.p")

    def _rename_blocs(self):
        """
        Changes relevant data to match historical majority/minority names.
        """
        # changing names to match historical data
        majority_bloc = [
            bloc for bloc, prop in self.bloc_voter_prop.items() if prop >= 0.5
        ][0]
        minority_bloc = [
            bloc for bloc in self.bloc_voter_prop.keys() if bloc != majority_bloc
        ][0]

        cambridge_names = {
            majority_bloc: self.historical_majority,
            minority_bloc: self.historical_minority,
        }

        self.slate_to_candidates = {
            cambridge_names[b]: self.slate_to_candidates[b]
            for b in self.slate_to_candidates.keys()
        }

        self.bloc_voter_prop = {
            cambridge_names[b]: self.bloc_voter_prop[b]
            for b in self.bloc_voter_prop.keys()
        }

        self.pref_interval_by_bloc = {
            cambridge_names[b]: self.pref_interval_by_bloc[b]
            for b in self.pref_interval_by_bloc.keys()
        }

        self.cohesion_parameters = {
            cambridge_names[b]: self.cohesion_parameters[b]
            for b in self.cohesion_parameters.keys()
        }

    def generate_profile(self, number_of_ballots: int) -> PreferenceProfile:
        with open(self.path, "rb") as pickle_file:
            ballot_frequencies = pickle.load(pickle_file)

        ballot_pool = []

        # compute the number of bloc and crossover voters in each bloc using Huntington Hill
        voter_types = [
            (b, t) for b in list(self.bloc_voter_prop.keys()) for t in ["bloc", "cross"]
        ]

        voter_props = [
            self.cohesion_parameters[b] * self.bloc_voter_prop[b]
            if t == "bloc"
            else (1 - self.cohesion_parameters[b]) * self.bloc_voter_prop[b]
            for b, t in voter_types
        ]

        ballots_per_type = dict(
            zip(
                voter_types,
                apportion.compute("huntington", voter_props, number_of_ballots),
            )
        )

        blocs = self.slate_to_candidates.keys()

        for bloc in blocs:
            # store the opposition bloc
            opp_bloc = next(iter(set(blocs).difference(set(bloc))))

            # find total number of ballots that start with bloc and opp_bloc
            bloc_first_count = sum(
                [
                    freq
                    for ballot, freq in ballot_frequencies.items()
                    if ballot[0] == bloc
                ]
            )

            opp_bloc_first_count = sum(
                [
                    freq
                    for ballot, freq in ballot_frequencies.items()
                    if ballot[0] == opp_bloc
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
                ballot: freq / opp_bloc_first_count
                for ballot, freq in ballot_frequencies.items()
                if ballot[0] == opp_bloc
            }

            bloc_voters = ballots_per_type[(bloc, "bloc")]
            cross_voters = ballots_per_type[(bloc, "cross")]

            # Generate ballots
            for i in range(bloc_voters + cross_voters):
                # Based on first choice, randomly choose
                # ballots weighted by Cambridge frequency
                if i < bloc_voters:
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
