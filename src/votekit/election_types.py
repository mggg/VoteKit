from fractions import Fraction
import itertools as it
import numpy as np
import random
from typing import Callable, Optional

from .ballot import Ballot
from .election_state import ElectionState
from .graphs.pairwise_comparison_graph import PairwiseComparisonGraph
from .pref_profile import PreferenceProfile
from .utils import (
    compute_votes,
    remove_cand,
    fractional_transfer,
    order_candidates_by_borda,
    borda_scores,
    seqRCV_transfer,
)


class STV:
    """
    Class for single-winner IRV and multi-winner STV elections
    """

    def __init__(
        self,
        profile: PreferenceProfile,
        transfer: Callable,
        seats: int,
        quota: str = "droop",
    ):
        """
        profile (PreferenceProfile): initial perference profile
        transfer (function): vote transfer method such as fractional transfer
        seats (int): number of winners/size of committee
        """
        self.__profile = profile
        self.transfer = transfer
        self.seats = seats
        self.election_state = ElectionState(
            curr_round=0,
            elected=[],
            eliminated=[],
            remaining=[
                cand
                for cand, votes in compute_votes(
                    profile.get_candidates(), profile.get_ballots()
                )
            ],
            profile=profile,
        )
        self.threshold = self.get_threshold(quota)

    # can cache since it will not change throughout rounds
    def get_threshold(self, quota: str) -> int:
        quota = quota.lower()
        if quota == "droop":
            return int(self.__profile.num_ballots() / (self.seats + 1) + 1)
        elif quota == "hare":
            return int(self.__profile.num_ballots() / self.seats)
        else:
            raise ValueError("Misspelled or unknown quota type")

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
        fp_votes = compute_votes(remaining, ballots)  ##fp means first place
        elected = []
        eliminated = []

        # if number of remaining candidates equals number of remaining seats,
        # everyone is elected
        if len(remaining) == self.seats - len(self.election_state.get_all_winners()):
            elected = [cand for cand, votes in fp_votes]
            remaining = []
            ballots = []
            # TODO: sort remaining candidates by vote share

        # elect all candidates who crossed threshold
        elif fp_votes[0][1] >= self.threshold:
            for candidate, votes in fp_votes:
                if votes >= self.threshold:
                    elected.append(candidate)
                    remaining.remove(candidate)
                    ballots = self.transfer(
                        candidate,
                        ballots,
                        {cand: votes for cand, votes in fp_votes},
                        self.threshold,
                    )
        # since no one has crossed threshold, eliminate one of the people
        # with least first place votes
        elif self.next_round():
            lp_votes = min([votes for cand, votes in fp_votes])
            lp_candidates = [
                candidate for candidate, votes in fp_votes if votes == lp_votes
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
        remaining = list(set(profile.get_candidates()).difference(cands_removed))
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

        self.state = new_state  # type: ignore
        return new_state  # type: ignore

    def run_election(self) -> ElectionState:
        outcome = None
        while self.stage != "Complete":
            outcome = self.run_step(self.stage)
        return outcome  # type: ignore


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
                remaining=list(),
                profile=PreferenceProfile(),
                previous=self.state,
            )
        else:
            new_state = ElectionState(
                curr_round=self.state.curr_round + 1,
                elected=[set(dominating_tiers[0])],
                eliminated=dominating_tiers[1:][::-1],
                remaining=list(),
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
            remaining=list(),
            profile=PreferenceProfile(),
            previous=self.state,
        )
        return new_state

    def run_election(self) -> ElectionState:
        outcome = self.run_step()
        return outcome


class Plurality:
    """
    Single or multi-winner plurality election
    """

    def __init__(self, profile: PreferenceProfile, seats: int):
        self.seats = seats
        self.election_state = ElectionState(
            curr_round=0,
            elected=[],
            eliminated=[],
            remaining=[
                cand
                for cand, votes in compute_votes(
                    profile.get_candidates(), profile.get_ballots()
                )
            ],
            profile=profile,
        )
        self.__profile = profile

    def run_step(self):
        """
        Simulate 'step' of a plurarity election
        """
        candidates = self.__profile.get_candidates()
        ballots = self.__profile.get_ballots()
        results = compute_votes(candidates, ballots)

        return ElectionState(
            curr_round=1,
            elected=[result.cand for result in results[: self.seats]],
            eliminated=[result.cand for result in results[self.seats :]],
            remaining=[],
            profile=self.__profile,
        )

    run_election = run_step


class SequentialRCV:
    """
    class to run Sequential RCV election
    """

    def __init__(self, profile: PreferenceProfile, seats: int):
        """
        profile (PreferenceProfile): initial perference profile
        seats (int): number of winners/size of committee
        """
        self.seats = seats
        self.profile = profile
        self.election_state = ElectionState(
            curr_round=0,
            elected=[],
            eliminated=[],
            remaining=[],
            profile=profile,
        )

    def run_step(self, old_profile: PreferenceProfile) -> ElectionState:
        """
        Simulates a single step of the sequential RCV contest
        which is a full IRV election run on the current set of candidates
        """
        old_election_state = self.election_state

        IRVrun = STV(old_profile, transfer=seqRCV_transfer, seats=1)
        old_election = IRVrun.run_election()
        elected_cand = old_election.get_all_winners()[0]

        # Removes elected candidate from Ballot List
        updated_ballots = remove_cand(elected_cand, old_profile.get_ballots())

        # Updates profile with removed candidates
        updated_profile = PreferenceProfile(ballots=updated_ballots)

        self.election_state = ElectionState(
            curr_round=old_election_state.curr_round + 1,
            elected=list(elected_cand),
            profile=updated_profile,
            previous=old_election_state,
            remaining=old_election.remaining,
        )
        return self.election_state

    def run_election(self) -> ElectionState:
        """
        Simulates a complete sequential RCV contest.
        Will run rounds of elections until elected seats fill
        """
        old_profile = self.profile
        elected = []  # type: ignore
        seqRCV_step = self.election_state

        while len(elected) < self.seats:
            seqRCV_step = self.run_step(old_profile)
            elected.append(seqRCV_step.elected)
            old_profile = seqRCV_step.profile
        return seqRCV_step


class Borda:
    def __init__(
        self,
        profile: PreferenceProfile,
        seats: int,
        score_vector: Optional[list[Fraction]],
    ):
        self.state = ElectionState(curr_round=0, profile=profile)
        self.seats = seats
        self.score_vector = score_vector

    """Borda: Borda is a positional voting system that assigns a decreasing 
    number of points to candidates based on order and a score vector.
    The conventional score vector is linear (n, n-1, ... 1)"""

    def run_step(self) -> ElectionState:
        candidates = self.state.profile.get_candidates()
        borda_dict = borda_scores(
            profile=self.state.profile, score_vector=self.score_vector
        )
        ranking = order_candidates_by_borda(set(candidates), borda_dict)

        new_state = ElectionState(
            curr_round=self.state.curr_round + 1,
            elected=ranking[: self.seats],
            eliminated=ranking[self.seats :][::-1],
            remaining=list(),
            profile=PreferenceProfile(),
            previous=self.state,
        )
        return new_state

    def run_election(self) -> ElectionState:
        outcome = self.run_step()
        return outcome
