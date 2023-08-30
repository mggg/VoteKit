from fractions import Fraction
import itertools as it
import numpy as np
import random
from typing import Callable, Optional

from .ballot import Ballot
from .models import Election
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


class STV(Election):
    """
    Class for single-winner IRV and multi-winner STV elections

     **Attributes**

    `profile`
    :   PreferenceProfile to run election on

    `transfer`
    :   transfer method (e.g. fractional transfer)

    `seats`
    :   number of seats to be elected

    `qouta`
    :   formula to calculate qouta (defaults to droop)

    `ties`
    :   (Optional) resolves input ties if True, else assumes ballots have no ties

    **Methods**
    """

    def __init__(
        self,
        profile: PreferenceProfile,
        transfer: Callable,
        seats: int,
        quota: str = "droop",
        ties: bool = True,
    ):
        # let parent class handle the og profile and election state
        super().__init__(profile, ties)

        self.transfer = transfer
        self.seats = seats
        self.threshold = self.get_threshold(quota)

    # can cache since it will not change throughout rounds
    def get_threshold(self, quota: str) -> int:
        """
        Calculates threshold required for election

        Args:
            qouta: Type of qouta formula

        Returns:
            Value of the threshold
        """
        quota = quota.lower()
        if quota == "droop":
            return int(self._profile.num_ballots() / (self.seats + 1) + 1)
        elif quota == "hare":
            return int(self._profile.num_ballots() / self.seats)
        else:
            raise ValueError("Misspelled or unknown quota type")

    def next_round(self) -> bool:
        """
        Determines if the number of seats has been met to call an election

        Returns:
            True if number of seats has been met, False otherwise
        """
        return len(self.state.get_all_winners()) != self.seats

    def run_step(self) -> ElectionState:
        """
        Simulates one round an STV election

        Returns:
           An ElectionState object for a given round
        """
        remaining: list[str] = self.state.profile.get_candidates()
        ballots: list[Ballot] = self.state.profile.get_ballots()
        round_votes = compute_votes(remaining, ballots)
        elected = []
        eliminated = []

        # if number of remaining candidates equals number of remaining seats,
        # everyone is elected
        if len(remaining) == self.seats - len(self.state.get_all_winners()):
            elected = [cand for cand, votes in round_votes]
            remaining = []
            ballots = []

        # elect all candidates who crossed threshold
        elif round_votes[0].votes >= self.threshold:
            for candidate, votes in round_votes:
                if votes >= self.threshold:
                    elected.append(candidate)
                    remaining.remove(candidate)
                    ballots = self.transfer(
                        candidate,
                        ballots,
                        {cand: votes for cand, votes in round_votes},
                        self.threshold,
                    )
        # since no one has crossed threshold, eliminate one of the people
        # with least first place votes
        elif self.next_round():
            lp_candidates = [
                candidate
                for candidate, votes in round_votes
                if votes == round_votes[-1].votes
            ]

            lp_cand = random.choice(lp_candidates)
            eliminated.append(lp_cand)
            ballots = remove_cand(lp_cand, ballots)
            remaining.remove(lp_cand)

        self.state = ElectionState(
            curr_round=self.state.curr_round + 1,
            elected=elected,
            eliminated=eliminated,
            remaining=remaining,
            profile=PreferenceProfile(ballots=ballots),
            previous=self.state,
        )
        return self.state

    def run_election(self) -> ElectionState:
        """
        Runs complete STV election

        Returns:
            An ElectionState object with results for a complete election
        """
        if not self.next_round():
            raise ValueError(
                f"Length of elected set equal to number of seats ({self.seats})"
            )

        while self.next_round():
            self.run_step()

        return self.state


class Limited(Election):
    """
    Limited: Elects m (seats) candidates with the highest k-approval scores.
    The k-approval score of a candidate is equal to the number of voters who \n
    rank this candidate among their k top ranked candidates.

    **Attributes**

    `profile`
    :   PreferenceProfile to run election on

    `k`
    :   value of an approval score

    `seats`
    :   number of seats to be elected

    `ties`
    :   (Optional) resolves input ties if True, else assumes ballots have no ties

    **Methods**
    """

    def __init__(
        self, profile: PreferenceProfile, seats: int, k: int, ties: bool = True
    ):
        super().__init__(profile, ties)
        self.seats = seats
        self.k = k

    def run_step(self) -> ElectionState:
        """
        Conducts Limited election in which m-candidates are elected based
        on approval scores

        Returns:
           An ElectionState object for a Limited election
        """
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
        """
        Simulates a complete Limited election

        Returns:
            An ElectionState object with results for a complete election
        """
        outcome = self.run_step()
        return outcome


class Bloc(Election):
    """
    Bloc: Elects m candidates with the highest m-approval scores. The m-approval \n
    score of a candidate is equal to the number of voters who rank this \n
    candidate among their m top ranked candidates.

    **Attributes**

    `profile`
    :   PreferenceProfile to run election on

    `seats`
    :   number of seats to be elected

    `ties`
    :   (Optional) resolves input ties if True, else assumes ballots have no ties

    **Methods**
    """

    def __init__(self, profile: PreferenceProfile, seats: int, ties: bool = True):
        super().__init__(profile, ties)
        self.seats = seats

    def run_step(self) -> ElectionState:
        """
        Conducts a Limited election to elect m-candidates

        Returns:
           An ElectionState object for a Limited election
        """
        limited_equivalent = Limited(
            profile=self.state.profile, seats=self.seats, k=self.seats
        )
        outcome = limited_equivalent.run_election()
        return outcome

    def run_election(self) -> ElectionState:
        """
        Runs complete Bloc election

        Returns:
            An ElectionState object with results for a complete election
        """
        outcome = self.run_step()
        return outcome


class SNTV(Election):
    """
    Single nontransferable vote (SNTV): Elects k-candidates with the highest \n
    Plurality scores

    **Attributes**

    `profile`
    :   PreferenceProfile to run election on

    `seats`
    :   number of seats to be elected

    `ties`
    :   (Optional) resolves input ties if True, else assumes ballots have no ties

    **Methods**
    """

    def __init__(self, profile: PreferenceProfile, seats: int, ties: bool = True):
        super().__init__(profile, ties)
        self.seats = seats

    def run_step(self) -> ElectionState:
        """
        Conducts a Limited election to elect k-candidates

        Returns:
           An ElectionState object for a Limited election
        """
        limited_equivalent = Limited(profile=self.state.profile, seats=self.seats, k=1)
        outcome = limited_equivalent.run_election()
        return outcome

    def run_election(self) -> ElectionState:
        """
        Runs complete SNTV election

        Returns:
            An ElectionState object with results for a complete election
        """
        outcome = self.run_step()
        return outcome


class SNTV_STV_Hybrid(Election):
    """
    SNTV-IRV Hybrid: This method first runs SNTV to a cutoff, then
    runs STV to pick a committee with a given number of seats.

    **Attributes**

    `profile`
    :   PreferenceProfile to run election on

    `transfer`
    :   transfer method (e.g. fractional transfer)

    `r1_cutoff`
    :   first-round cutoff value

    `seats`
    :   number of seats to be elected

    `ties`
    :   (Optional) resolves input ties if True, else assumes ballots have no ties

    **Methods**
    """

    def __init__(
        self,
        profile: PreferenceProfile,
        transfer: Callable,
        r1_cutoff: int,
        seats: int,
        ties: bool = True,
    ):
        super().__init__(profile, ties)
        self.transfer = transfer
        self.r1_cutoff = r1_cutoff
        self.seats = seats
        self.stage = "SNTV"  # SNTV, switches to STV, then Complete

    def run_step(self, stage: str) -> ElectionState:
        """
        Simulates one round an SNTV_STV election

        Args:
            stage: Stage of the hybrid election, can be SNTV or STV

        Returns:
           An ElectionState object for a given round
        """
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
        """
        Runs complete SNTV_STV election

        Returns:
            An ElectionState object with results for a complete election
        """
        outcome = None
        while self.stage != "Complete":
            outcome = self.run_step(self.stage)
        return outcome  # type: ignore


class TopTwo(Election):
    """
    Top Two: Eliminates all but the top two plurality vote getters,and then
    conducts a runoff between them, reallocating other ballots

    **Attributes**

    `profile`
    :   PreferenceProfile to run election on

    `seats`
    :   number of seats to be elected

    `ties`
    :   (Optional) resolves input ties if True, else assumes ballots have no ties

    **Methods**
    """

    def __init__(self, profile: PreferenceProfile, ties: bool = True):
        super().__init__(profile, ties)

    def run_step(self) -> ElectionState:
        """
        Conducts a hybrid election for one seat with a cutoff of 2 for the runoff

        Returns:
            An ElectionState object for the hybrid election
        """
        hybrid_equivalent = SNTV_STV_Hybrid(
            profile=self.state.profile,
            transfer=fractional_transfer,
            r1_cutoff=2,
            seats=1,
        )
        outcome = hybrid_equivalent.run_election()
        return outcome

    def run_election(self) -> ElectionState:
        """
        Simulates a complete TopTwo election

        Returns:
            An ElectionState object for a complete election
        """
        outcome = self.run_step()
        return outcome


class DominatingSets(Election):
    """
    Finds tiers of candidates by dominating set,which is a set of candidates
    such that every candidate in the set wins head to head comparisons against
    candidates outside of it

    **Attributes**

    `profile`
    :   PreferenceProfile to run election on

    `ties`
    :   (Optional) resolves input ties if True, else assumes ballots have no ties

    **Methods**
    """

    def __init__(self, profile: PreferenceProfile, ties: bool = True):
        super().__init__(profile, ties)

    def run_step(self) -> ElectionState:
        """
        Conducts a complete DominatingSets election as it is not a round-by-round
        system

        Returns:
            An ElectionState object for a complete election
        """
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
        """
        Simulates a complete DominatingSets election

        Returns:
            An ElectionState object for a complete election
        """
        outcome = self.run_step()
        return outcome


class CondoBorda(Election):
    """
    Condo-Borda: Elects candidates ordered by dominating set, but breaks ties
    between candidates

    **Attributes**

    `profile`
    :   PreferenceProfile to run election on

    `seats`
    :   number of seats to be elected

    `ties`
    :   (Optional) resolves input ties if True, else assumes ballots have no ties

    **Methods**
    """

    def __init__(self, profile: PreferenceProfile, seats: int, ties: bool = True):
        super().__init__(profile, ties)
        self.seats = seats

    def run_step(self) -> ElectionState:
        """
        Conducts a complete Conda-Borda election as it is not a round-by-round
        system

        Returns:
            An ElectionState object for a complete election
        """
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
        """
        Simulates a complete Conda-Borda election

        Returns:
            An ElectionState object for a complete election
        """
        outcome = self.run_step()
        return outcome


class Plurality(Election):
    """
    Single or multi-winner plurality election

    **Attributes**

    `profile`
    :   PreferenceProfile to run election on

    `seats`
    :   number of seats to be elected

    `ties`
    :   (Optional) resolves input ties if True, else assumes ballots have no ties

    **Methods**
    """

    def __init__(self, profile: PreferenceProfile, seats: int, ties: bool = True):

        super().__init__(profile, ties)
        self.seats = seats

    def run_step(self):
        """
        Simulates a complete Pluarality election as it is not a round-by-round
        system

        Returns:
            An ElectionState object for a complete election
        """
        candidates = self._profile.get_candidates()
        ballots = self._profile.get_ballots()
        results = compute_votes(candidates, ballots)

        return ElectionState(
            curr_round=1,
            elected=[result.cand for result in results[: self.seats]],
            eliminated=[result.cand for result in results[self.seats :]],
            remaining=[],
            profile=self._profile,
        )

    def run_election(self) -> ElectionState:
        """
        Simulates a complete Pluarality election

        Returns:
            An ElectionState object for a complete election
        """
        return self.run_step()


class SequentialRCV(Election):
    """
    Class to conduct Sequential RCV election, in which votes are not transferred
    after a candidate has reached threshold, or been elected

    **Attributes**

    `profile`
    :   PreferenceProfile to run election on

    `seats`
    :   number of seats to be elected

    `ties`
    :   (Optional) resolves input ties if True, else assumes ballots have no ties

    **Methods**
    """

    def __init__(self, profile: PreferenceProfile, seats: int, ties: bool = True):
        super().__init__(profile, ties)
        self.seats = seats

    def run_step(self, old_profile: PreferenceProfile) -> ElectionState:
        """
        Simulates a single step of the sequential RCV contest or a full
        IRV election run on the current set of candidates

         Returns:
           An ElectionState object for a given round
        """
        old_election_state = self.state

        IRVrun = STV(old_profile, transfer=seqRCV_transfer, seats=1)
        old_election = IRVrun.run_election()
        elected_cand = old_election.get_all_winners()[0]

        # Removes elected candidate from Ballot List
        updated_ballots = remove_cand(elected_cand, old_profile.get_ballots())

        # Updates profile with removed candidates
        updated_profile = PreferenceProfile(ballots=updated_ballots)

        self.state = ElectionState(
            curr_round=old_election_state.curr_round + 1,
            elected=list(elected_cand),
            profile=updated_profile,
            previous=old_election_state,
            remaining=old_election.remaining,
        )
        return self.state

    def run_election(self) -> ElectionState:
        """
        Simulates a complete sequential RCV contest.

        Returns:
            An ElectionState object for a complete election
        """
        old_profile = self._profile
        elected = []  # type: ignore
        seqRCV_step = self.state

        while len(elected) < self.seats:
            seqRCV_step = self.run_step(old_profile)
            elected.append(seqRCV_step.elected)
            old_profile = seqRCV_step.profile
        return seqRCV_step


class Borda(Election):
    """
    Positional voting system that assigns a decreasing number of points to
    candidates based on order and a score vector. The conventional score
    vector is linear (n, n-1, ... 1)

    **Attributes**

    `profile`
    :   PreferenceProfile to run election on

    `seats`
    :   number of seats to be elected

    `score_vector`
    :   (Optional) weights assigned to candidate ranking

    `ties`
    :   (Optional) resolves input ties if True, else assumes ballots have no ties

    **Methods**
    """

    def __init__(
        self,
        profile: PreferenceProfile,
        seats: int,
        score_vector: Optional[list[Fraction]],
        ties: bool = True,
    ):
        super().__init__(profile, ties)
        self.seats = seats
        self.score_vector = score_vector

    def run_step(self) -> ElectionState:
        """
        Simulates a complete Borda contest as Borda is not a round-by-round
        system

        Returns:
            An ElectionState object for a complete election
        """
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
        """
        Simulates a complete Borda contest

        Returns:
            An ElectionState object for a complete election
        """
        outcome = self.run_step()
        return outcome
