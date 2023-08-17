from .profile import PreferenceProfile
from .ballot import Ballot
from .election_state import ElectionState
from .graphs import PairwiseComparisonGraph
from .metrics import borda_scores
from typing import Callable, Iterable, Union
import random
from fractions import Fraction
from copy import deepcopy
import itertools as it
import numpy as np


class STV:
    def __init__(self, profile: PreferenceProfile, transfer: Callable, seats: int):
        self.state = ElectionState(curr_round=0, profile=profile)
        self.transfer = transfer
        self.seats = seats
        self.threshold = self.get_threshold()

    # can cache since it will not change throughout rounds
    def get_threshold(self) -> int:
        """
        Droop qouta
        """
        return int(self.state.profile.num_ballots() / (self.seats + 1) + 1)

    def next_round(self) -> bool:
        """
        Determines if the number of seats has been met to call election
        """
        return len(self.state.get_all_winners()) != self.seats

    def run_step(self) -> ElectionState:
        """
        Simulates one round an STV election
        """
        candidates: list = self.state.profile.get_candidates()
        ballots: list = self.state.profile.get_ballots()
        fp_votes: dict = compute_votes(candidates, ballots)

        print(fp_votes)
        round_elected = list()
        round_eliminated = list()

        # if number of remaining candidates equals number of remaining seats
        if len(candidates) == self.seats - len(self.state.get_all_winners()):
            round_elected = sorted(fp_votes, key=lambda x: (-fp_votes[x], x))
        else:
            for candidate in sorted(fp_votes, key=lambda x: (-fp_votes[x], x)):
                if fp_votes[candidate] >= self.threshold:
                    round_elected.append(candidate)
                    ballots = self.transfer(
                        candidate, ballots, fp_votes, self.threshold
                    )

            if self.next_round():
                lp_votes = min(fp_votes.values())
                lp_candidates = [
                    candidate
                    for candidate, votes in fp_votes.items()
                    if votes == lp_votes
                ]
                # is this how to break ties, can be different based on locality
                lp_cand = random.choice(lp_candidates)
                ballots = remove_cand(lp_cand, ballots)
                round_eliminated.append(lp_cand)

        cands_removed = set(round_elected).union(set(round_eliminated))
        round_remaining = set(candidates).difference(cands_removed)
        new_state = ElectionState(
            curr_round=self.state.curr_round + 1,
            elected=round_elected,
            eliminated=round_eliminated,
            remaining=round_remaining,
            profile=PreferenceProfile(ballots=ballots),
            votes=fp_votes,
            previous=self.state,
        )
        self.state = new_state
        return new_state

    def run_election(self) -> ElectionState:
        """
        Runs complete STV election
        """

        if not self.next_round():
            raise ValueError(
                f"Length of elected set equal to number of seats ({self.seats})"
            )

        outcome = None
        while self.next_round():
            outcome = self.run_step()
        return outcome


class Limited:
    def __init__(self, profile: PreferenceProfile, seats: int, k: int):
        self.state = ElectionState(curr_round=0, profile=profile)
        self.seats = seats
        self.k = k

    """Limited: This rule returns the m (seats) candidates with the highest k-approval scores. 
    The k-approval score of a candidate is equal to the number of voters who rank this 
    candidate among their k top ranked candidates. """

    def run_step(self) -> ElectionState:
        profile = self.state.profile
        candidates = profile.get_candidates()
        candidate_approvals = {c: Fraction(0) for c in candidates}

        for ballot in profile.get_ballots():
            # First we have to determine which candidates are approved
            # i.e. in first k ranks on a ballot
            approvals = []
            for i, cand_set in enumerate(ballot.ranking):
                # If list of total candidates before and including current set
                # are less than seat count, all candidates are approved
                if len(list(it.chain(*ballot.ranking[: i + 1]))) < self.k:
                    approvals.extend(list(cand_set))
                # If list of total candidates before current set
                # are greater than seat count, no candidates are approved
                elif len(list(it.chain(*ballot.ranking[:i]))) > self.k:
                    approvals.extend([])
                # Else we know the cutoff is in the set, we compute and randomly
                # select the number of candidates we can select
                else:
                    accepted = len(list(it.chain(*ballot.ranking[:i])))
                    num_to_allow = self.k - accepted
                    approvals.extend(
                        np.random.choice(list(cand_set), num_to_allow, replace=False)
                    )

            # Add approval votes equal to ballot weight (i.e. number of voters with this ballot)
            for cand in approvals:
                candidate_approvals[cand] += ballot.weight

        # Order candidates by number of approval votes received
        ordered_results = sorted(
            candidate_approvals, key=lambda x: (-candidate_approvals[x], x)
        )

        # Construct ElectionState information
        elected = ordered_results[: self.seats]
        eliminated = ordered_results[self.seats :][::-1]
        cands_removed = set(elected).union(set(eliminated))
        remaining = set(profile.get_candidates()).difference(cands_removed)
        profile_remaining = PreferenceProfile(
            ballots=remove_cand(cands_removed, profile.get_ballots())
        )
        new_state = ElectionState(
            curr_round=self.state.curr_round + 1,
            elected=elected,
            eliminated=eliminated,
            remaining=remaining,
            profile=profile_remaining,
            previous=self.state,
        )
        self.state = new_state
        return self.state

    def run_election(self) -> ElectionState:
        outcome = self.run_step()
        return outcome


class Bloc:
    def __init__(self, profile: PreferenceProfile, seats: int):
        self.state = ElectionState(curr_round=0, profile=profile)
        self.seats = seats

    """Bloc: This rule returns the m candidates with the highest m-approval scores. 
    The m-approval score of a candidate is equal to the number of voters who rank this 
    candidate among their m top ranked candidates. """

    def run_step(self) -> ElectionState:
        limited_equivalent = Limited(
            profile=self.state.profile, seats=self.seats, k=self.seats
        )
        outcome = limited_equivalent.run_election()
        return outcome

    def run_election(self) -> ElectionState:
        outcome = self.run_step()
        return outcome


class SNTV:
    def __init__(self, profile: PreferenceProfile, seats: int):
        self.state = ElectionState(curr_round=0, profile=profile)
        self.seats = seats

    """Single nontransferable vote (SNTV): SNTV returns k candidates with the highest
    Plurality scores """

    def run_step(self) -> ElectionState:
        limited_equivalent = Limited(profile=self.state.profile, seats=self.seats, k=1)
        outcome = limited_equivalent.run_election()
        return outcome

    def run_election(self) -> ElectionState:
        outcome = self.run_step()
        return outcome


class SNTV_STV_Hybrid:
    def __init__(
        self, profile: PreferenceProfile, transfer: Callable, r1_cutoff: int, seats: int
    ):
        self.state = ElectionState(curr_round=0, profile=profile)
        self.transfer = transfer
        self.r1_cutoff = r1_cutoff
        self.seats = seats
        self.stage = "SNTV"  # SNTV, switches to STV, then Complete

    """SNTV-IRV Hybrid: This method first runs SNTV to a cutoff r1_cutoff, then
    runs STV to pick a committee with a given number of seats."""

    def run_step(self, stage: str) -> ElectionState:
        profile = self.state.profile

        new_state = None
        if stage == "SNTV":
            round_state = SNTV(profile=profile, seats=self.r1_cutoff).run_election()

            # The STV election will be run on the new election state
            # Therefore we should not add any winners, but rather
            # set the SNTV winners as remaining candidates and update pref profiles
            new_profile = PreferenceProfile(
                ballots=remove_cand(round_state.eliminated, profile.get_ballots())
            )
            new_state = ElectionState(
                curr_round=self.state.curr_round + 1,
                elected=list(),
                eliminated=round_state.eliminated,
                remaining=new_profile.get_candidates(),
                profile=new_profile,
                previous=self.state,
            )
        elif stage == "STV":
            round_state = STV(
                profile=profile, transfer=self.transfer, seats=self.seats
            ).run_election()
            # Since get_all_eliminated returns reversed eliminations
            new_state = ElectionState(
                curr_round=self.state.curr_round + 1,
                elected=round_state.get_all_winners(),
                eliminated=round_state.get_all_eliminated()[::-1],
                remaining=round_state.remaining,
                profile=round_state.profile,
                previous=self.state,
            )

        # Update election stage to cue next run step
        if stage == "SNTV":
            self.stage = "STV"
        elif stage == "STV":
            self.stage = "Complete"

        self.state = new_state
        return new_state

    def run_election(self) -> ElectionState:
        outcome = None
        while self.stage != "Complete":
            outcome = self.run_step(self.stage)
        return outcome


class TopTwo:
    def __init__(self, profile: PreferenceProfile):
        self.state = ElectionState(curr_round=0, profile=profile)

    """Top Two: Top two eliminates all but the top two plurality vote getters,
    and then conducts a runoff between them, reallocating other ballots."""

    def run_step(self) -> ElectionState:
        # Top 2 is equivalent to a hybrid election for one seat
        # with a cutoff of 2 for the runoff
        hybrid_equivalent = SNTV_STV_Hybrid(
            profile=self.state.profile,
            transfer=fractional_transfer,
            r1_cutoff=2,
            seats=1,
        )
        outcome = hybrid_equivalent.run_election()
        return outcome

    def run_election(self) -> ElectionState:
        outcome = self.run_step()
        return outcome


class DominatingSets:
    def __init__(self, profile: PreferenceProfile):
        self.state = ElectionState(curr_round=0, profile=profile)

    """Dominating sets: Return the tiers of candidates by dominating set,
    which is a set of candidates such that every candidate in the set wins 
    head to head comparisons against candidates outside of it"""

    def run_step(self) -> ElectionState:
        pwc_graph = PairwiseComparisonGraph(self.state.profile)
        dominating_tiers = pwc_graph.dominating_tiers()
        if len(dominating_tiers) == 1:
            new_state = ElectionState(
                curr_round=self.state.curr_round + 1,
                elected=list(),
                eliminated=dominating_tiers,
                remaining=set(),
                profile=PreferenceProfile(),
                previous=self.state,
            )
        else:
            new_state = ElectionState(
                curr_round=self.state.curr_round + 1,
                elected=[set(dominating_tiers[0])],
                eliminated=dominating_tiers[1:][::-1],
                remaining=set(),
                profile=PreferenceProfile(),
                previous=self.state,
            )
        return new_state

    def run_election(self) -> ElectionState:
        outcome = self.run_step()
        return outcome


class CondoBorda:
    def __init__(self, profile: PreferenceProfile, seats: int):
        self.state = ElectionState(curr_round=0, profile=profile)
        self.seats = seats

    """Condo-Borda: Condo-Borda returns candidates ordered by dominating set,
    but breaks ties between candidates """

    def run_step(self) -> ElectionState:
        pwc_graph = PairwiseComparisonGraph(self.state.profile)
        dominating_tiers = pwc_graph.dominating_tiers()
        candidate_borda = borda_scores(self.state.profile)
        ranking = []
        for dt in dominating_tiers:
            ranking += order_candidates_by_borda(dt, candidate_borda)

        new_state = ElectionState(
            curr_round=self.state.curr_round + 1,
            elected=ranking[: self.seats],
            eliminated=ranking[self.seats :],
            remaining=set(),
            profile=PreferenceProfile(),
            previous=self.state,
        )
        return new_state

    def run_election(self) -> ElectionState:
        outcome = self.run_step()
        return outcome


# Election Helper Functions
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


def remove_cand(removed: Union[str, Iterable], ballots: list[Ballot]) -> list[Ballot]:
    remove_set = {}
    if isinstance(removed, str):
        remove_set = {removed}
    elif isinstance(removed, Iterable):
        remove_set = set(removed)

    update = deepcopy(ballots)
    for n, ballot in enumerate(update):
        new_ranking = []
        for s in ballot.ranking:
            new_s = s.difference(remove_set)
            if len(new_s) > 0:
                new_ranking.append(new_s)
        update[n].ranking = new_ranking

    return update


def order_candidates_by_borda(candidate_set, candidate_borda):
    # Sort the candidates in candidate_set based on their Borda values
    ordered_candidates = sorted(
        candidate_set,
        key=lambda candidate: candidate_borda.get(candidate, 0),
        reverse=True,
    )
    return ordered_candidates
