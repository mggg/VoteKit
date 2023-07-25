from .profile import PreferenceProfile
from .ballot import Ballot
from .election_state import ElectionState
from typing import Callable
import random
from fractions import Fraction
from copy import deepcopy

##We're not passing winner_votes to outcome right now
# If we want to track which candidate each ballot is ultimately going to, we need to do more.
# TODO:
# 1. winner_votes to election_state is supposed to have all the ballots in the
# initial profile that ended up going to any elected candidate.
# Bit tricky since all of our methods work with current state of profile not the initial
# 2. integrate Cincinatti transfer
class STV:
    def __init__(self, profile: PreferenceProfile, transfer: Callable, seats: int):
        self.__profile = profile
        self.transfer: Callable = transfer
        self.seats: int = seats
        self.election_state: ElectionState = ElectionState(
            curr_round=0,
            elected=[],
            eliminated=[],
            remaining=compute_votes(profile.get_candidates(), profile.get_ballots())[1],
            profile=profile,
        )
        self.threshold: float = self.get_threshold()

    # can cache since it will not change throughout rounds
    def get_threshold(self) -> int:
        """
        Droop quota
        """
        return int(self.election_state.profile.num_ballots() / (self.seats + 1) + 1)

    def next_round(self) -> bool:
        """
        Determines if the number of seats has been met to call election
        """
        return len(self.election_state.get_all_winners()) != self.seats

    def run_step(self) -> ElectionState:
        """
        Simulates one round an STV election
        """
        ##TODO:must change the way we pass winner_votes
        remaining: list[str] = self.election_state.remaining
        ballots: list[Ballot] = self.election_state.profile.get_ballots()
        fp_votes: dict
        fp_order: list[str]
        fp_votes, fp_order = compute_votes(remaining, ballots)
        elected = []
        eliminated = []

        # if number of remaining candidates equals number of remaining seats, everyone is elected
        if len(remaining) == self.seats - len(self.election_state.get_all_winners()):
            elected = fp_order
            remaining = []
            ballots = []
            # TODO: sort remaining candidates by vote share

        # elect all candidates who crossed threshold
        elif fp_votes[fp_order[0]] >= self.threshold:
            for candidate in fp_order:
                if fp_votes[candidate] >= self.threshold:
                    elected.append(candidate)
                    remaining.remove(candidate)
                    ballots = self.transfer(
                        candidate, ballots, fp_votes, self.threshold
                    )
        # since no one has crossed threshold, eliminate one of the people
        # with least first place votes
        elif self.next_round():
            lp_votes = min(fp_votes.values())
            lp_candidates = [
                candidate for candidate in fp_order if fp_votes[candidate] == lp_votes
            ]
            # is this how to break ties, can be different based on locality
            lp_cand = random.choice(lp_candidates)
            eliminated.append(lp_cand)
            ballots = remove_cand(lp_cand, ballots)
            remaining.remove(lp_cand)

        self.election_state = ElectionState(
            curr_round=self.election_state.curr_round + 1,
            elected=elected,
            eliminated=eliminated,
            remaining=remaining,
            profile=PreferenceProfile(ballots=ballots),
            previous=self.election_state,
        )
        return self.election_state

    def run_election(self) -> ElectionState:
        """
        Runs complete STV election
        """
        if not self.next_round():
            raise ValueError(
                f"Length of elected set equal to number of seats ({self.seats})"
            )

        while self.next_round():
            self.run_step()

        return self.election_state

    def get_init_profile(self):
        "returns the initial profile of the election"
        return self.__profile


## Election Helper Functions


def compute_votes(candidates: list, ballots: list[Ballot]) -> tuple[dict, list]:
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

    order = [y[0] for y in sorted(votes.items(), key=lambda x: x[1], reverse=True)]

    return votes, order


def fractional_transfer(
    winner: str, ballots: list[Ballot], votes: dict, threshold: int
) -> list[Ballot]:
    # find the transfer value, add transfer value to weights of ballots
    # that listed the elected in first place, remove that cand and shift
    # everything up, recomputing first-place votes
    transfer_value = (votes[winner] - threshold) / votes[winner]

    for ballot in ballots:
        if ballot.ranking and ballot.ranking[0] == {winner}:
            ballot.weight = ballot.weight * transfer_value

    return remove_cand(winner, ballots)


def remove_cand(removed_cand: str, ballots: list[Ballot]) -> list[Ballot]:
    """
    Removes candidate from ranking of the ballots
    """
    update = deepcopy(ballots)

    for n, ballot in enumerate(update):
        new_ranking = [
            candidate for candidate in ballot.ranking if candidate != {removed_cand}
        ]
        update[n].ranking = new_ranking

    return update
