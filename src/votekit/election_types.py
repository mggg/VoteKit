from .profile import PreferenceProfile
from .ballot import Ballot
from .models import Outcome
from typing import Callable
import random
from fractions import Fraction
from copy import deepcopy


class STV:
    def __init__(self, profile: PreferenceProfile, transfer: Callable, seats: int):
        self.profile = profile
        self.transfer = transfer
        self.elected: set = set()
        self.eliminated: set = set()
        self.seats = seats
        self.threshold = self.get_threshold()

    # can cache since it will not change throughout rounds
    def get_threshold(self) -> int:
        """
        Droop qouta
        """
        return int(self.profile.num_ballots() / (self.seats + 1) + 1)

    def next_round(self) -> bool:
        """
        Determines if the number of seats has been met to call election
        """
        return len(self.elected) != self.seats

    def run_step(self, profile: PreferenceProfile) -> tuple[PreferenceProfile, Outcome]:
        """
        Simulates one round an STV election
        """
        candidates: list = profile.get_candidates()
        ballots: list = profile.get_ballots()
        fp_votes: dict = compute_votes(candidates, ballots)

        # if number of remaining candidates equals number of remaining seats
        if len(candidates) == self.seats - len(self.elected):
            # TODO: sort remaing candidates by vote share
            self.elected.update(set(candidates))
            return profile, Outcome(
                elected=self.elected,
                eliminated=self.eliminated,
                remaining=set(candidates),
                votes=fp_votes,
            )

        for candidate in candidates:
            if fp_votes[candidate] >= self.threshold:
                self.elected.add(candidate)
                candidates.remove(candidate)
                ballots = self.transfer(candidate, ballots, fp_votes, self.threshold)

        if self.next_round():
            lp_votes = min(fp_votes.values())
            lp_candidates = [
                candidate for candidate, votes in fp_votes.items() if votes == lp_votes
            ]
            # is this how to break ties, can be different based on locality
            lp_cand = random.choice(lp_candidates)
            ballots = remove_cand(lp_cand, ballots)
            candidates.remove(lp_cand)
            self.eliminated.add(lp_cand)

        return PreferenceProfile(ballots=ballots), Outcome(
            elected=self.elected,
            eliminated=self.eliminated,
            remaining=set(candidates),
            votes=fp_votes,
        )

    def run_election(self) -> Outcome:
        """
        Runs complete STV election
        """
        profile = deepcopy(self.profile)

        if not self.next_round():
            raise ValueError(
                f"Length of elected set equal to number of seats ({self.seats})"
            )

        while self.next_round():
            profile, outcome = self.run_step(profile)

        return outcome


## Election Helper Functions


class Borda:
    def __init__(self, profile: PreferenceProfile, seats: int, borda_weights: list):

        self.profile = profile
        self.borda_weights = borda_weights
        self.seats = seats

    def run_borda_step(self):
        """
        Simulates a complete Borda election
        """

        borda_scores = {}  # {candidate : int borda_score}
        candidate_rank_freq = (
            {}
        )  # {candidate : [1st rank total, 2nd rank total,..., n rank total]}
        candidates_ballots = {}  # {candidate : [ballots mentioning candidate]}

        for ballot in self.profile.get_ballots():
            frequency = ballot.weight
            index = 0
            for candidate in ballot.ranking:
                candidate = str(candidate)

                if candidate not in candidate_rank_freq:
                    candidate_rank_freq[candidate] = [
                        0 for _ in range(len(ballot.ranking))
                    ]
                    candidate_rank_freq[candidate][index] = frequency
                else:
                    candidate_rank_freq[candidate][index] += frequency
                if candidate not in candidates_ballots:
                    candidates_ballots[candidate] = []
                    candidates_ballots[candidate].append(ballot)
                else:
                    candidates_ballots[candidate].append(ballot)
                index += 1

        for key in candidate_rank_freq:
            borda_scores[key] = sum(
                [x * y for x, y in zip(candidate_rank_freq[key], self.borda_weights)]
            )

        sorted_borda = sorted(borda_scores, key=borda_scores.get, reverse=True)

        winners = sorted_borda[: self.seats]

        # get winner_votes
        # TO-DO: Adjust Outcome class to new args
        winner_votes = {}
        for winner in winners:
            winner_votes[winner] = candidates_ballots[winner]

        return PreferenceProfile(ballots=self.profile.get_ballots()), Outcome(
            remaining=set(),
            elected=set(winners),
            eliminated=set(sorted_borda[self.seats :]),
        )

        # return PreferenceProfile(ballots=profile.get_ballots(), Outcome(
        #     curr_round=1,
        #     elected=winners,
        #     eliminated=sorted_borda[seats:],
        #     remaining=[],
        #     profile=profile,
        #     winner_votes=winner_votes,
        #     previous=None
        # )

    def run_borda_election(self):
        return self.run_borda_step()[1]


def compute_votes(candidates: list, ballots: list[Ballot]) -> dict:
    """
    Computes first place votes for all candidates in a preference profile
    """
    votes = {}

    for candidate in candidates:
        weight = Fraction(0)
        for ballot in ballots:
            if ballot.ranking and ballot.ranking[0] == {candidate}:
                weight += ballot.weight
        votes[candidate] = weight

    return votes


def fractional_transfer(
    winner: str, ballots: list[Ballot], votes: dict, threshold: int
) -> list[Ballot]:
    # find the transfer value, add tranfer value to weights of vballots
    # that listed the elected in first place, remove that cand and shift
    # everything up, recomputing first-place votes
    transfer_value = (votes[winner] - threshold) / votes[winner]

    for ballot in ballots:
        if ballot.ranking and ballot.ranking[0] == {winner}:
            ballot.weight = ballot.weight * transfer_value

    transfered = remove_cand(winner, ballots)

    return transfered


def remove_cand(removed_cand: str, ballots: list[Ballot]) -> list[Ballot]:
    """
    Removes candidate from ranking of the ballots
    """
    update = deepcopy(ballots)

    for n, ballot in enumerate(update):
        new_ranking = []
        for candidate in ballot.ranking:
            if candidate != {removed_cand}:
                new_ranking.append(candidate)
        update[n].ranking = new_ranking

    return update
