from .profile import PreferenceProfile
from .ballot import Ballot
from .models import Outcome
from typing import Callable
import random
from fractions import Fraction


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

    # change name of this function and reverse bool
    def is_complete(self) -> bool:
        """
        Determines if the number of seats has been met to call election
        """
        return len(self.elected) == self.seats

    def run_step(self, profile: PreferenceProfile) -> tuple[PreferenceProfile, Outcome]:
        """
        Simulates one round an STV election
        """
        candidates: list = profile.get_candidates()
        ballots: list = profile.get_ballots()
        fp_votes: dict = compute_votes(candidates, ballots)

        # print('ballots', type(ballots))
        # print('candidates', type(candidates))

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
                print("winner", candidate)
                ballots = self.transfer(candidate, ballots, fp_votes, self.threshold)

        if not self.is_complete():
            lp_votes = min(fp_votes.values())
            lp_candidates = [
                candidate for candidate, votes in fp_votes.items() if votes == lp_votes
            ]
            # is this how to break ties, can be different based on locality
            lp_cand = random.choice(lp_candidates)
            ballots = remove_cand(lp_cand, ballots)
            candidates.remove(lp_cand)
            print("loser", lp_cand)
            self.eliminated.add(lp_cand)

        return PreferenceProfile(ballots=ballots), Outcome(
            elected=self.elected,
            eliminated=self.eliminated,
            remaining=set(candidates),
            votes=fp_votes,
        )


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
    for n, ballot in enumerate(ballots):
        new_ranking = []
        for candidate in ballot.ranking:
            if candidate != {removed_cand}:
                new_ranking.append(candidate)
        ballots[n].ranking = new_ranking

    return ballots
