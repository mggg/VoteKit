from abc import abstractmethod
from pydantic import BaseModel
from typing import Optional
from .profile import PreferenceProfile
from .ballot import Ballot
from numpy.random import choice
import itertools as it
import random
import numpy as np

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


class Ballot_Generator(BaseModel):

    # cand is a set
    number_of_ballots: int
    candidates: list
    ballot_length: Optional[int]
    # slate_to_candidate: Optional[dict]  # race: [candidate]
    # pref_interval_by_slate: Optional[dict] = None  # race: {candidate : interval length}
    # demo_breakdown: Optional[dict] = None  # race: percentage

    def __init__(self, **data):
        super().__init__(**data)
        self.ballot_length = len(self.candidate_list)
        self.candidate_list = Ballot_Generator.cand_list_set(self.candiates)

    @abstractmethod
    def generate_ballots(self) -> PreferenceProfile:
        pass

    @staticmethod
    def cand_list_to_set(candidate_list):
        return [set(cand) for cand in candidate_list]

    @staticmethod
    def ballot_pool_to_profile(ballot_pool, candidate_list):
        ballot_weights = {}
        for ballot in ballot_pool:
            ballot_weights[tuple(ballot)] = ballot_weights.get(tuple(ballot), 0) + 1
        ballot_list = [
            Ballot(ranking=list(ranking), weight=weight)
            for ranking, weight in ballot_weights.items()
        ]
        return PreferenceProfile(ballots=ballot_list, candidates=candidate_list)


class IC(Ballot_Generator):
    def generate_ballots(self) -> PreferenceProfile:
        perm_set = it.permutations(self.candidate_list)
        #
        # Create a list of every perm [['A', 'B', 'C'], ['A', 'C', 'B'], ...]
        perm_rankings = [list(value) for value in perm_set]

        ballot_pool = []
        # TODO: why not just generate ballots from a uniform distribution like this
        #  ballot = list(choice(cand_list, ballot_length, p=cand_support_vec, replace=False))
        # instead of permutating all the candidates and then sampling

        for _ in range(self.number_of_ballots):
            index = random.randint(0, len(perm_rankings) - 1)
            ballot_pool.append(perm_rankings[index])

        return self.ballot_pool_to_profile(ballot_pool, self.candidate_list)


class IAC(Ballot_Generator):
    def generate_ballots(self) -> PreferenceProfile:
        perm_set = it.permutations(self.candidate_list)
        # Create a list of every perm [['A', 'B', 'C'], ['A', 'C', 'B'], ...]
        perm_rankings = [list(value) for value in perm_set]

        # IAC Process is equivalent to drawing from dirichlet dist with uniform parameters
        draw_probabilites = np.random.dirichlet([1] * len(perm_rankings))

        ballot_pool = []

        for _ in range(self.number_of_ballots):
            index = np.random.choice(range(6), 1, p=draw_probabilites)[0]
            ballot_pool.append(perm_rankings[index])

        return self.ballot_pool_to_profile(ballot_pool, self.candidate_list)


class PlackettLuce(Ballot_Generator):
    def __init__(
        self,
        number_of_ballots: int,
        candidate_list: list[set],
        pref_interval_by_slate: dict,
        slate_voter_prop: dict,
    ):
        # Call the parent class's __init__ method to handle common parameters
        super().__init__(
            number_of_ballots=number_of_ballots, candidate_list=candidate_list
        )

        # Assign additional parameters specific to PlackettLuce
        self.pref_interval_by_slate = pref_interval_by_slate
        self.slate_voter_prop = slate_voter_prop

    def generate_ballots(self) -> PreferenceProfile:

        # TODO: what to do with candidate_to_slate? add dirchlet sample option?s

        ballots_list = []

        for race in self.slate_voter_prop.keys():
            # number of voters in this race/block
            num_ballots_race = int(self.number_of_ballots * self.slate_voter_prop[race])
            pref_interval_dict = self.pref_interval_by_slate[race]
            # creates the interval of probabilities for candidates supported by this block
            cand_support_vec = [
                pref_interval_dict[cand] for cand in self.candidate_list
            ]

            for _ in range(num_ballots_race):
                ballot = list(
                    choice(
                        self.candidate_list,
                        self.ballot_length,
                        p=cand_support_vec,
                        replace=False,
                    )
                )
                ballots_list.append(ballot)

        pp = self.ballot_pool_to_profile(
            ballot_pool=ballots_list, candidate_list=self.candidate_list
        )
        return pp


class BradleyTerry(Ballot_Generator):
    def __init__(
        self,
        number_of_ballots: int,
        candidate_list: list[set],
        # slate_to_candidate: dict,
        pref_interval_by_slate: dict,
        slate_voter_prop: dict,
    ):
        # Call the parent class's __init__ method to handle common parameters
        super().__init__(
            number_of_ballots=number_of_ballots, candidate_list=candidate_list
        )

        # Assign additional parameters specific to Bradley Terry
        # self.slate_to_candidate = slate_to_candidate
        self.pref_interval_by_slate = pref_interval_by_slate
        self.slate_voter_prop = slate_voter_prop

    def calc_prob(self, ranking: list[set], cand_support: dict(set, float)) -> float:
        prob = 1
        for i in range(len(ranking)):
            cand_i = ranking[i]
            greater_cand_support = cand_support[cand_i]
            for j in range(i, len(ranking)):
                cand_j = ranking[j]
                cand_support = cand_support[cand_j]
                prob *= greater_cand_support / (greater_cand_support + cand_support)
        return prob

    def generate_ballots(self) -> PreferenceProfile:

        permutations = list(it.permutations(self.candidate_list, self.ballot_length))
        ballots_list = []

        for slate in self.slate_voter_prop.keys():
            num_ballots_slate = int(
                self.number_of_ballots * self.slate_voter_prop[slate]
            )
            pref_interval_dict = self.pref_interval_by_slate[slate]
            cand_support_vec = [
                pref_interval_dict[cand] for cand in self.candidate_list
            ]

            ranking_to_prob = {}
            for ranking in permutations:
                prob = self.calc_prob(ranking=ranking, cand_support=cand_support_vec)
                ranking_to_prob[ranking] = prob

            ballots = list(
                choice(
                    ranking_to_prob.keys(),
                    num_ballots_slate,
                    p=ranking_to_prob.values(),
                    replace=True,
                )
            )

            ballots_list = ballots_list + ballots

        pp = self.ballot_pool_to_profile(
            ballot_pool=ballots_list, candidate_list=self.candidate_list
        )
        return pp


class AlternatingCrossover(Ballot_Generator):
    def __init__(
        self,
        number_of_ballots: int,
        candidate_list: list[set],
        slate_to_candidate: dict,
        pref_interval_by_slate: dict,
        slate_voter_prop: dict,
        slate_crossover_rate: dict,
    ):
        # Call the parent class's __init__ method to handle common parameters
        super().__init__(
            number_of_ballots=number_of_ballots, candidate_list=candidate_list
        )

        # Assign additional parameters specific to
        self.slate_to_candidate = slate_to_candidate
        self.pref_interval_by_slate = pref_interval_by_slate
        self.slate_voter_prop = slate_voter_prop
        self.slate_crossover_rate = slate_crossover_rate

    def generate_ballots(self) -> PreferenceProfile:
        # assumes only two slates?
        ballots_list = []

        for slate in self.slate_voter_prop.keys():
            num_ballots_race = int(
                self.number_of_ballots * self.slate_voter_prop[slate]
            )
            crossover_dict = self.slate_crossover_rate[slate]
            pref_interval_dict = self.pref_interval_by_slate[slate]

            for opposing_slate in crossover_dict.keys():
                crossover_rate = crossover_dict[opposing_slate]
                crossover_ballots = crossover_rate * num_ballots_race

                opposing_cands = self.slate_to_candidate[opposing_slate]
                bloc_cands = self.slate_to_candidate[slate]

                for _ in range(crossover_ballots):
                    pref_for_opposing = [
                        pref_interval_dict[cand] for cand in opposing_cands
                    ]
                    pref_for_bloc = [pref_interval_dict[cand] for cand in bloc_cands]

                    bloc_cands = list(
                        choice(self.candidate_list, p=pref_for_bloc, replace=False)
                    )
                    opposing_cands = list(
                        choice(self.candidate_list, p=pref_for_opposing, replace=False)
                    )

                    ballot = bloc_cands
                    if slate != opposing_slate:  # alternate
                        ballot = [
                            item
                            for sublist in zip(opposing_cands, bloc_cands)
                            for item in sublist
                        ]

                    # check that ballot_length is shorter than total number of cands
                    ballot = ballot[: self.ballot_length]
                    ballots_list.append(ballot)

        pp = self.ballot_pool_to_profile(
            ballot_pool=ballots_list, candidate_list=self.candidate_list
        )
        return pp


class CambridgeSampler(Ballot_Generator):
    def generate_ballots(self) -> PreferenceProfile:
        pass


class OneDimSpatial(Ballot_Generator):
    def generate_ballots(self) -> PreferenceProfile:
        candidate_position_dict = {
            c: np.random.normal(1, 0, len(self.candidate_list))
            for c in self.candidate_list
        }
        voter_positions = np.random.normal(1, 0, self.number_of_ballots)

        ballot_pool = []

        for vp in voter_positions:
            distance_dict = {
                c: abs(v - vp) for c, v, in candidate_position_dict.items()
            }
            candidate_order = sorted(distance_dict, key=distance_dict.get)
            ballot_pool.append(candidate_order)

        return self.ballot_pool_to_profile(ballot_pool)


# class TwoDimSpatial(Ballot_Generator):
#     @override
#     def generate_ballots() -> PreferenceProfile:
#         pass


if __name__ == "__main__":
    ...
