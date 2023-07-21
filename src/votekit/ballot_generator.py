from abc import abstractmethod
from fractions import Fraction
from typing import Optional
from profile import PreferenceProfile
from votekit.ballot import Ballot
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


class Ballot_Generator:

    # cand is a set
    # number_of_ballots: int
    # candidates: list
    # ballot_length: Optional[int]
    # slate_to_candidate: Optional[dict]  # race: [candidate]
    # pref_interval_by_slate: Optional[dict] = None  # race: {candidate : interval length}
    # demo_breakdown: Optional[dict] = None  # race: percentage

    def __init__(
        self, number_of_ballots: int, candidates: list, ballot_length: Optional[int]
    ):
        self.number_of_ballots = number_of_ballots
        self.ballot_length = (
            ballot_length if ballot_length is not None else len(candidates)
        )
        # self.candidate_list = Ballot_Generator.list_to_set(candidates)
        self.candidates = candidates

    @abstractmethod
    def generate_ballots(self) -> PreferenceProfile:
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
            rank = [set(cand) for cand in ranking]
            b = Ballot(ranking=rank, weight=Fraction(count))
            ballot_list.append(b)
        return PreferenceProfile(ballots=ballot_list, candidates=candidates)


class IC(Ballot_Generator):
    def generate_ballots(self) -> PreferenceProfile:
        perm_set = it.permutations(self.candidates, self.ballot_length)
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

        return self.ballot_pool_to_profile(ballot_pool, self.candidates)


class IAC(Ballot_Generator):
    def generate_ballots(self) -> PreferenceProfile:
        perm_set = it.permutations(self.candidates, self.ballot_length)
        # Create a list of every perm [['A', 'B', 'C'], ['A', 'C', 'B'], ...]
        perm_rankings = [list(value) for value in perm_set]

        # IAC Process is equivalent to drawing from dirichlet dist with uniform parameters
        draw_probabilites = np.random.dirichlet([1] * len(perm_rankings))

        ballot_pool = []

        for _ in range(self.number_of_ballots):
            index = np.random.choice(range(6), 1, p=draw_probabilites)[0]
            ballot_pool.append(perm_rankings[index])

        return self.ballot_pool_to_profile(ballot_pool, self.candidates)


class PlackettLuce(Ballot_Generator):
    def __init__(self, pref_interval_by_slate: dict, slate_voter_prop: dict, **data):
        # Call the parent class's __init__ method to handle common parameters
        super().__init__(**data)

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
                # ballot = Ballot_Generator.list_to_set(ballot)
                ballots_list.append(ballot)

        pp = self.ballot_pool_to_profile(
            ballot_pool=ballots_list, candidates=self.candidates
        )
        return pp


class BradleyTerry(Ballot_Generator):
    def __init__(self, pref_interval_by_slate: dict, slate_voter_prop: dict, **data):
        # Call the parent class's __init__ method to handle common parameters
        super().__init__(**data)

        # Assign additional parameters specific to Bradley Terry
        # self.slate_to_candidate = slate_to_candidate
        self.pref_interval_by_slate = pref_interval_by_slate
        self.slate_voter_prop = slate_voter_prop

    def _calc_prob(self, ranking: list[set], cand_support: dict) -> float:
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

        permutations = list(it.permutations(self.candidates, self.ballot_length))
        ballots_list = []

        for slate in self.slate_voter_prop.keys():
            num_ballots_slate = int(
                self.number_of_ballots * self.slate_voter_prop[slate]
            )
            pref_interval_dict = self.pref_interval_by_slate[slate]
            cand_support_vec = [pref_interval_dict[cand] for cand in self.candidates]

            ranking_to_prob = {}
            for ranking in permutations:
                prob = self._calc_prob(ranking=ranking, cand_support=cand_support_vec)
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
            ballot_pool=ballots_list, candidates=self.candidates
        )
        return pp


class AlternatingCrossover(Ballot_Generator):
    def __init__(
        self,
        slate_to_candidate: dict,
        pref_interval_by_slate: dict,
        slate_voter_prop: dict,
        slate_crossover_rate: dict,
        **data
    ):
        # Call the parent class's __init__ method to handle common parameters
        super().__init__(**data)

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
                        choice(self.candidates, p=pref_for_bloc, replace=False)
                    )
                    opposing_cands = list(
                        choice(self.candidates, p=pref_for_opposing, replace=False)
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
            ballot_pool=ballots_list, candidates=self.candidates
        )
        return pp


class CambridgeSampler(Ballot_Generator):
    def generate_ballots(self) -> PreferenceProfile:
        pass


class OneDimSpatial(Ballot_Generator):
    def generate_ballots(self) -> PreferenceProfile:
        candidate_position_dict = {
            c: np.random.normal(1, 0, len(self.candidates)) for c in self.candidates
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
    candidates = ["a", "b", "c"]
    number_of_ballots = 5
    ballot_length = 2
    pref_interval_by_slate = {
        "white": {"a": 0.1, "b": 0.5, "c": 0.4},
        "black": {"a": 0.2, "b": 0.5, "c": 0.3},
    }
    slate_voter_prop = {"white": 0.8, "black": 0.2}

    # gen = IC(number_of_ballots=number_of_ballots,
    #  candidates=candidates, ballot_length=ballot_length)

    # gen = IAC(
    #     number_of_ballots=number_of_ballots,
    #     candidates=candidates,
    #     ballot_length=ballot_length,
    # )

    gen = PlackettLuce(
        number_of_ballots=number_of_ballots,
        candidates=candidates,
        ballot_length=2,
        pref_interval_by_slate=pref_interval_by_slate,
        slate_voter_prop=slate_voter_prop,
    )
    print(gen.generate_ballots())