from abc import abstractmethod
from fractions import Fraction
from typing import Optional, Union
from .profile import PreferenceProfile
from .ballot import Ballot
from numpy.random import choice
import itertools as it
import random
import numpy as np
import pickle
from pathlib import Path
from itertools import zip_longest

"""
IC
IAC
1d spatial
2d spatial
PL
BT
AC
Cambridge
"""


class BallotGenerator:

    # cand is a set
    # number_of_ballots: int
    # candidates: list
    # ballot_length: Optional[int]
    # slate_to_candidate: dict  # race: [candidate]
    # pref_interval_by_slate: dict = None  # race: {candidate : interval length}
    # hyperameters: dict cohesion, Direchlet alphas
    # demo_breakdown: dict = None  # race: percentage

    def __init__(
        self,
        number_of_ballots: int,
        candidates: Union[list, dict],
        ballot_length: Optional[int] = None,
        hyperparameters: Optional[dict] = {},
    ):
        self.number_of_ballots = number_of_ballots
        if isinstance(candidates, dict):
            self.slate_to_candidate = candidates
            self.candidates = list(
                {cand for cands in candidates.values() for cand in cands}
            )
        else:
            self.candidates = candidates
        self.ballot_length = (
            ballot_length if ballot_length is not None else len(self.candidates)
        )

        if hyperparameters:
            if isinstance(candidates, dict):  # add type error
                self.set_params(candidates, **hyperparameters)
                self.parameterized = True
            else:
                raise TypeError(
                    "'candidates' must be dictionary when hyperparameters are set"
                )

    @abstractmethod
    def generate_profile(self) -> PreferenceProfile:
        pass

    # @staticmethod
    # def list_to_set(candidates):
    #     return [set(cand) for cand in candidates]

    @staticmethod
    def ballot_pool_to_profile(ballot_pool, candidates):
        ranking_counts = {}
        ballot_list = []

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

    def set_params(
        self,
        candidates: dict,  # add type error for list here
        blocs: dict,
        cohesion: dict,
        alphas: dict,
        crossover: dict = {},
    ) -> None:
        """
        Generates perference intervals for slates based on pararmeters and specified models.
        """
        if sum(blocs.values()) != 1.0:
            raise ValueError(f"bloc proportions ({blocs.values()}) do not equal 1")

        def _preference_interval(
            alphas: dict, cohesion: int, bloc: str, candidates: dict
        ) -> dict:
            """
            Creates a preference interval for bloc of votes (e.g. W or POC)

            Inputs:
                alphas: dict of alpha parameters for Dirichlet distribution mapped to a
                    group (e.g. support for POC, or support for W)
                cohesion: support for candidates
                candidates dict: list of candidates mapped to their group

            Returns: perference interval, a dictionary with candidates mapped to
                their intervals
            """
            intervals = {}

            for group, alpha in alphas.items():
                num_cands = len(candidates[group])
                probs = list(np.random.dirichlet([alpha] * num_cands))
                for prob, cand in zip(probs, candidates[group]):
                    if group == bloc:  # e.g W for W cands
                        pi = cohesion
                    elif group != bloc:  # e.g W for POC cands
                        pi = 1 - cohesion
                    intervals[cand] = pi * prob

            return intervals

        interval_by_slate = {}
        for bloc in blocs:
            interval = _preference_interval(
                alphas[bloc], cohesion[bloc], bloc, candidates
            )
            interval_by_slate[bloc] = interval

        self.pref_interval_by_slate = interval_by_slate
        self.slate_voter_prop = blocs
        self.slate_crossover_rate = crossover


class IC(BallotGenerator):
    def generate_profile(self) -> PreferenceProfile:
        perm_set = it.permutations(self.candidates, self.ballot_length)
        # Create a list of every perm [['A', 'B', 'C'], ['A', 'C', 'B'], ...]
        perm_rankings = [list(value) for value in perm_set]

        ballot_pool = []

        for _ in range(self.number_of_ballots):
            index = random.randint(0, len(perm_rankings) - 1)
            ballot_pool.append(perm_rankings[index])

        return self.ballot_pool_to_profile(ballot_pool, self.candidates)


class IAC(BallotGenerator):
    def generate_profile(self) -> PreferenceProfile:
        perm_set = it.permutations(self.candidates, self.ballot_length)
        # Create a list of every perm [['A', 'B', 'C'], ['A', 'C', 'B'], ...]
        perm_rankings = [list(value) for value in perm_set]

        # IAC Process is equivalent to drawing from dirichlet dist with uniform parameters
        draw_probabilities = np.random.dirichlet([1] * len(perm_rankings))

        ballot_pool = []

        for _ in range(self.number_of_ballots):
            index = np.random.choice(
                range(len(perm_rankings)), 1, p=draw_probabilities
            )[0]
            ballot_pool.append(perm_rankings[index])

        return self.ballot_pool_to_profile(ballot_pool, self.candidates)


class PlackettLuce(BallotGenerator):
    def __init__(
        self, pref_interval_by_slate: dict = {}, slate_voter_prop: dict = {}, **data
    ):
        # Call the parent class's __init__ method to handle common parameters
        super().__init__(**data)

        # Assign additional parameters specific to PlackettLuce
        if not self.parameterized:
            self.pref_interval_by_slate = pref_interval_by_slate
            self.slate_voter_prop = slate_voter_prop

    def generate_profile(self) -> PreferenceProfile:
        ballot_pool = []

        for race in self.slate_voter_prop.keys():
            # number of voters in this race/block
            num_ballots_race = int(self.number_of_ballots * self.slate_voter_prop[race])
            pref_interval_dict = self.pref_interval_by_slate[race]
            # creates the interval of probabilities for candidates supported by this block
            cand_support_vec = [pref_interval_dict[cand] for cand in self.candidates]

            for _ in range(num_ballots_race):
                ballot = list(
                    choice(
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
    def __init__(
        self, pref_interval_by_slate: dict = {}, slate_voter_prop: dict = {}, **data
    ):
        # Call the parent class's __init__ method to handle common parameters
        super().__init__(**data)

        # Assign additional parameters specific to Bradley Terrys
        if not self.parameterized:
            self.pref_interval_by_slate = pref_interval_by_slate
            self.slate_voter_prop = slate_voter_prop

    def _calc_prob(self, permutations: list, cand_support_dict: dict) -> dict:
        ranking_to_prob = {}
        for ranking in permutations:
            prob = 1
            for i in range(len(ranking)):
                cand_i = ranking[i]
                greater_cand_support = cand_support_dict[cand_i]
                for j in range(i, len(ranking)):
                    cand_j = ranking[j]
                    cand_support = cand_support_dict[cand_j]
                    prob *= greater_cand_support / (greater_cand_support + cand_support)
            ranking_to_prob[ranking] = prob
        return ranking_to_prob

    def generate_profile(self) -> PreferenceProfile:

        permutations = list(it.permutations(self.candidates, self.ballot_length))
        ballot_pool: list = []

        for slate in self.slate_voter_prop.keys():
            num_ballots_slate = int(
                self.number_of_ballots * self.slate_voter_prop[slate]
            )
            pref_interval_dict = self.pref_interval_by_slate[slate]
            # cand_support_vec = [pref_interval_dict[cand] for cand in self.candidates]

            ranking_to_prob = self._calc_prob(
                permutations=permutations, cand_support_dict=pref_interval_dict
            )

            indices = range(len(ranking_to_prob))
            prob_distrib = list(ranking_to_prob.values())
            prob_distrib = [float(p) / sum(prob_distrib) for p in prob_distrib]

            ballots_indices = choice(
                indices,
                num_ballots_slate,
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
    def __init__(
        self,
        slate_to_candidate: dict = {},
        pref_interval_by_slate: dict = {},
        slate_voter_prop: dict = {},
        slate_crossover_rate: dict = {},
        **data,
    ):
        # Call the parent class's __init__ method to handle common parameters
        super().__init__(**data)

        # Assign additional parameters specific to AC
        if not self.parameterized:
            self.slate_to_candidate = slate_to_candidate
            self.pref_interval_by_slate = pref_interval_by_slate
            self.slate_voter_prop = slate_voter_prop
            self.slate_crossover_rate = slate_crossover_rate

    def generate_profile(self) -> PreferenceProfile:

        # TODO: do some assertion checking,
        # like slate in slate_crossover == slate_voter_prop, and slate_to_cand,
        # basically all the places where we use one key for every dict

        ballot_pool = []

        for slate in self.slate_voter_prop.keys():

            # TODO: need to address a case that num ballots is not even number
            num_ballots_race = int(
                self.number_of_ballots * self.slate_voter_prop[slate]
            )
            # print(num_ballots_race)
            crossover_dict = self.slate_crossover_rate[slate]
            pref_interval_dict = self.pref_interval_by_slate[slate]

            for opposing_slate in crossover_dict.keys():
                crossover_rate = crossover_dict[opposing_slate]
                # print('xrate ', crossover_rate)
                crossover_ballots = int(crossover_rate * num_ballots_race)

                # print('num_ballots ', crossover_ballots)

                opposing_cands = self.slate_to_candidate[opposing_slate]
                bloc_cands = self.slate_to_candidate[slate]

                # print('og op cands: ', opposing_cands)
                # print('og bloc cands: ', bloc_cands)

                for _ in range(crossover_ballots):
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

                    # print('p op', pref_for_opposing)
                    # print('p bloc', pref_for_bloc)

                    bloc_cands = list(
                        choice(
                            bloc_cands,
                            p=pref_for_bloc,
                            size=len(bloc_cands),
                            replace=False,
                        )
                    )
                    opposing_cands = list(
                        choice(
                            opposing_cands,
                            size=len(opposing_cands),
                            p=pref_for_opposing,
                            replace=False,
                        )
                    )

                    # print('op cands: ', opposing_cands)
                    # print('bloc cands: ', bloc_cands)

                    # TODO: if there aren't enough candidates for a bloc,
                    # do we rank only the bloc candidates, or the bloc candidates and the others
                    # to fill the slots

                    if slate != opposing_slate:  # alternate
                        # ballot = [None]*(len(opposing_cands)+len(bloc_cands))
                        ballot = [
                            item
                            for pair in zip_longest(opposing_cands, bloc_cands)
                            for item in pair
                            if item is not None
                        ]
                        # print('opposing ', ballot)

                    # print(ballot)

                    # check that ballot_length is shorter than total number of cands
                    # ballot = ballot[: self.ballot_length+1]
                    ballot_pool.append(ballot)

                # Bloc ballots
                for _ in range(num_ballots_race - crossover_ballots):
                    ballot = bloc_cands + opposing_cands
                    # ballot = ballot[: self.ballot_length+1]
                    ballot_pool.append(ballot)

        pp = self.ballot_pool_to_profile(
            ballot_pool=ballot_pool, candidates=self.candidates
        )
        return pp


class CambridgeSampler(BallotGenerator):
    def __init__(
        self,
        path: Path,
        slate_to_candidate: dict = {},
        pref_interval_by_slate: dict = {},
        slate_voter_prop: dict = {},
        slate_crossover_rate: dict = {},
        **data,
    ):
        # Call the parent class's __init__ method to handle common parameters
        super().__init__(**data)

        # Assign additional parameters specific to
        if not self.parameterized:
            self.slate_to_candidate = slate_to_candidate
            self.pref_interval_by_slate = pref_interval_by_slate
            self.slate_voter_prop = slate_voter_prop
            self.slate_crossover_rate = slate_crossover_rate
            self.path = path

    def generate_profile(self) -> PreferenceProfile:

        # Load the ballot_type frequencies used in the Cambridge model
        # dict of form
        # {(cand_race1, cand_race1, ...): num_ballots, }
        # e.g. {('W', 'C', 'C'): 15, ('C', 'W', 'C', 'W'): 9, ...}
        with open(self.path, "rb") as pickle_file:
            ballot_frequencies = pickle.load(pickle_file)

        ballot_pool = []

        blocs = self.slate_to_candidate.keys()
        for bloc in blocs:
            opp_bloc = next(iter(set(blocs).difference(set(bloc))))

            # bloc or opp-first probabilities for each ballot variant
            bloc_first_count = sum(
                [
                    freq
                    for ballot, freq in ballot_frequencies.items()
                    if ballot[0] == bloc
                ]
            )
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

            bloc_voters = int(self.slate_voter_prop[bloc] * self.number_of_ballots)
            pref_interval_dict = self.pref_interval_by_slate[bloc]
            for _ in range(bloc_voters):
                first_choice = np.random.choice(
                    [bloc, opp_bloc],
                    p=[
                        1 - self.slate_crossover_rate[bloc][opp_bloc],
                        self.slate_crossover_rate[bloc][opp_bloc],
                    ],
                )
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
                    choice(
                        list(pref_interval_dict.keys()),
                        self.ballot_length,
                        p=list(pref_interval_dict.values()),
                        replace=False,
                    )
                )
                ordered_bloc_slate = [
                    c for c in pl_ordering if c in self.slate_to_candidate[bloc]
                ]
                ordered_opp_slate = [
                    c for c in pl_ordering if c in self.slate_to_candidate[opp_bloc]
                ]

                full_ballot = []
                for b in bloc_ordering:
                    if b == bloc:
                        if ordered_bloc_slate:
                            full_ballot.append(ordered_bloc_slate.pop())
                    else:
                        if ordered_opp_slate:
                            full_ballot.append(ordered_opp_slate.pop())

                ballot_pool.append(full_ballot)

        pp = self.ballot_pool_to_profile(
            ballot_pool=ballot_pool, candidates=self.candidates
        )
        return pp


class OneDimSpatial(BallotGenerator):
    def generate_profile(self) -> PreferenceProfile:
        candidate_position_dict = {c: np.random.normal(0, 1) for c in self.candidates}
        voter_positions = np.random.normal(0, 1, self.number_of_ballots)

        ballot_pool = []

        for vp in voter_positions:
            distance_dict = {
                c: abs(v - vp) for c, v, in candidate_position_dict.items()
            }
            candidate_order = sorted(distance_dict, key=distance_dict.__getitem__)
            ballot_pool.append(candidate_order)

        return self.ballot_pool_to_profile(ballot_pool, self.candidates)


# class TwoDimSpatial(BallotGenerator):
#     @override
#     def generate_profile() -> PreferenceProfile:
#         pass
