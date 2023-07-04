from unnamed_rcv_thing.election import Election
from unnamed_rcv_thing.profile import PreferenceProfile
from unnamed_rcv_thing.ballot import Ballot
from unnamed_rcv_thing.models import Outcome
from typing import Callable
from fractions import Fraction
import random


class STV(Election):
    def __init__(self, profile: PreferenceProfile, transfer: Callable, seats: int):

        self.profile = profile
        self.transfer = transfer
        self.elected = []
        self.eliminated = []
        self.seats = seats
        self.threshold = self.get_threshold()

    # can cache since it will not change throughout rounds
    def get_threshold(self):
        """
        Droop qouta
        """
        return int(self.profile.num_ballots() / (self.seats + 1) + 1)

    def is_complete(self):
        """
        Determines if the number of seats has been met to call election
        """
        if len(self.elected) == self.seats:
            return True

    def run_step(self, profile: PreferenceProfile):
        """
        Simulates one round an STV election
        """
        candidates = profile.get_candidates()
        ballots = profile.get_ballots()
        fp_votes = compute_votes(candidates, ballots)

        # if number of remaining candidates equals number of remaining seats
        if len(candidates) == self.seats - len(self.elected):
            # sort remaing candidates by vote share
            self.elected = self.elected + candidates
            return profile, Outcome(
                elected=self.elected,
                eliminated=self.eliminated,
                remaining=candidates,
                votes=fp_votes,
            )

        for candidate in candidates:
            if fp_votes[candidate] >= self.threshold:
                self.elected.append(candidate)
                candidates.remove(candidate)
                print(candidate)
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
            self.eliminated.append(lp_cand)

        return PreferenceProfile(ballots=ballots), Outcome(
            elected=self.elected,
            eliminated=self.eliminated,
            remaining=candidates,
            votes=fp_votes,
        )


def compute_votes(candidates: list, ballots: list[Ballot]):
    """
    Computes first place votes for all candidates in a preference profile
    """
    votes = {}

    for candidate in candidates:
        new_weight = 0
        for ballot in ballots:
            if ballot.ranking and ballot.ranking[0] == candidate:
                new_weight += ballot.weight
        votes[candidate] = new_weight

    return votes


def fractional_transfer(winner, ballots, votes, threshold):
    # find the transfer value, add tranfer value to weights of ballots
    # that listed the elected in first place, remove that cand and shift
    # everything up, recomputing first-place votes
    transfer_value = (votes[winner] - threshold) / votes[threshold]

    for ballot in ballots:
        if ballot.ranking[0] == winner:
            ballot.weight += ballot.weight * transfer_value

    transfered = remove_cand(winner, ballots)

    return transfered


def remove_cand(cand, ballots):
    """
    Removes candidate from ranking of the ballots
    """
    for n, ballot in enumerate(ballots):
        new_ranking = []
        for c in ballot.ranking:
            if c != cand:
                new_ranking.append(c)
        ballots[n].ranking = new_ranking

    return ballots


def gen_fake_ballots(ranks, cands, voters, weights=None):
    ballots = {}

    for voter in range(voters):
        ballot = tuple(random.choices(cands, weights, k=ranks))
        if ballot not in ballots:
            ballots[ballot] = 0
        ballots[ballot] += 1

    return ballots


# run time use case
# e.run_step(pp)  -> raise ElectionIsDone


# while not outcome.is_complete():
#     outcome = e.run_step()

fakes = gen_fake_ballots(3, ["a", "b", "c", "d", "e"], 45)

ballot_lst = []
for ranking, weight in fakes.items():
    ballot_lst.append(Ballot(ranking=ranking, weight=Fraction(weight)))

pp = PreferenceProfile(ballots=ballot_lst)
