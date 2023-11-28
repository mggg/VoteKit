from fractions import Fraction
import itertools as it
import numpy as np
from typing import Callable, Optional
from functools import lru_cache

from ..models import Election
from ..election_state import ElectionState
from ..graphs.pairwise_comparison_graph import PairwiseComparisonGraph
from ..pref_profile import PreferenceProfile
from .transfers import fractional_transfer, seqRCV_transfer
from ..utils import (
    compute_votes,
    remove_cand,
    borda_scores,
    scores_into_set_list,
    tie_broken_ranking,
    elect_cands_from_set_ranking,
    first_place_votes,
)

# add ballots attribute // remove preference profile so the original profile is
# not modified in place everytime?


class STV(Election):
    """
    Class for single-winner IRV and multi-winner STV elections.

     **Attributes**

    `profile`
    :   PreferenceProfile to run election on.

    `transfer`
    :   transfer method (e.g. fractional transfer).

    `seats`
    :   number of seats to be elected.

    `quota`
    :   formula to calculate quota (defaults to droop).

    `ballot_ties`
    :   (optional) resolves input ballot ties if True, else assumes ballots have no ties.
                    Defaults to True.

    `tiebreak`
    :   (optional) resolves procedural and final ties by specified tiebreak. Defaults
                to random.

    **Methods**
    """

    def __init__(
        self,
        profile: PreferenceProfile,
        transfer: Callable,
        seats: int,
        quota: str = "droop",
        ballot_ties: bool = True,
        tiebreak: str = "random",
    ):
        # let parent class handle the og profile and election state
        super().__init__(profile, ballot_ties)

        self.transfer = transfer
        self.seats = seats
        self.tiebreak = tiebreak
        self.quota = quota.lower()
        self.threshold = self.get_threshold()

    # can cache since it will not change throughout rounds
    def get_threshold(self) -> int:
        """
        Calculates threshold required for election.

        Returns:
            Value of the threshold.
        """
        quota = self.quota
        if quota == "droop":
            return int(self._profile.num_ballots() / (self.seats + 1) + 1)
        elif quota == "hare":
            return int(self._profile.num_ballots() / self.seats)
        else:
            raise ValueError("Misspelled or unknown quota type")

    def next_round(self) -> bool:
        """
        Determines if the number of seats has been met to call an election.

        Returns:
            True if number of seats has been met, False otherwise.
        """
        cands_elected = 0
        for s in self.state.winners():
            cands_elected += len(s)
        return cands_elected < self.seats

    def run_step(self) -> ElectionState:
        """
        Simulates one round an STV election.

        Returns:
           An ElectionState object for a given round.
        """
        round_num = self.state.curr_round
        remaining = self.state.profile.get_candidates()
        ballots = self.state.profile.get_ballots()
        round_votes, plurality_score = compute_votes(remaining, ballots)

        elected = []
        eliminated = []

        # if number of remaining candidates equals number of remaining seats,
        # everyone is elected
        if len(remaining) == self.seats - len(self.state.winners()):
            elected = [{cand} for cand, _ in round_votes]
            remaining = []
            ballots = []

        # elect all candidates who crossed threshold
        elif round_votes[0].votes >= self.threshold:
            for candidate, votes in round_votes:
                if votes >= self.threshold:
                    elected.append({candidate})
                    remaining.remove(candidate)
                    ballots = self.transfer(
                        candidate,
                        ballots,
                        {cand: votes for cand, votes in round_votes},
                        self.threshold,
                    )
        # since no one has crossed threshold, eliminate one of the people
        # with least first place votes
        else:
            lp_candidates = [
                candidate
                for candidate, votes in round_votes
                if votes == round_votes[-1].votes
            ]

            lp_cand = tie_broken_ranking(
                ranking=[set(lp_candidates)],
                profile=self.state.profile,
                tiebreak=self.tiebreak,
            )[-1]
            eliminated.append(lp_cand)
            ballots = remove_cand(lp_cand, ballots)
            remaining.remove(next(iter(lp_cand)))

        # sort candidates by vote share if multiple are elected
        if len(elected) >= 1:
            if self.state.curr_round > 0:
                score_dict = self.state.get_scores(round_num)
            else:
                score_dict = plurality_score

            elected = scores_into_set_list(score_dict, [c for s in elected for c in s])

        # Make sure list-of-sets have non-empty elements
        elected = [s for s in elected if s != set()]
        eliminated = [s for s in eliminated if s != set()]

        remaining = [set(remaining)]
        remaining = [s for s in remaining if s != set()]

        self.state = ElectionState(
            curr_round=self.state.curr_round + 1,
            elected=elected,
            eliminated_cands=eliminated,
            remaining=remaining,
            scores=plurality_score,
            profile=PreferenceProfile(ballots=ballots),
            previous=self.state,
        )
        return self.state

    @lru_cache
    def run_election(self) -> ElectionState:
        """
        Runs complete STV election.

        Returns:
            An ElectionState object with results for a complete election.
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
    Elects m candidates with the highest k-approval scores.
    The k-approval score of a candidate is equal to the number of voters who
    rank this candidate among their k top ranked candidates.

    **Attributes**

    `profile`
    :   PreferenceProfile to run election on.

    `k`
    :   value of an approval score.

    `seats`
    :   number of seats to be elected.

    `ballot_ties`
    :   (optional) resolves input ballot ties if True, else assumes ballots have no ties.
                    Defaults to True.

    `tiebreak`
    :   (optional) resolves procedural and final ties by specified tiebreak.
                    Defaults to random.

    **Methods**
    """

    def __init__(
        self,
        profile: PreferenceProfile,
        seats: int,
        k: int,
        ballot_ties: bool = True,
        tiebreak: str = "random",
    ):
        super().__init__(profile, ballot_ties)
        self.seats = seats
        self.k = k
        self.tiebreak = tiebreak

    def run_step(self) -> ElectionState:
        """
        Conducts Limited election in which m candidates are elected based
        on approval scores.

        Returns:
           An ElectionState object for a Limited election.
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
        ranking = scores_into_set_list(candidate_approvals)
        ranking = tie_broken_ranking(
            ranking=ranking, profile=self.state.profile, tiebreak=self.tiebreak
        )
        elected, eliminated = elect_cands_from_set_ranking(
            ranking=ranking, seats=self.seats
        )
        new_state = ElectionState(
            curr_round=self.state.curr_round + 1,
            elected=elected,
            eliminated_cands=eliminated,
            remaining=list(),
            scores=candidate_approvals,
            profile=PreferenceProfile(),
            previous=self.state,
        )
        self.state = new_state
        return self.state

    @lru_cache
    def run_election(self) -> ElectionState:
        """
        Simulates a complete Limited election.

        Returns:
            An ElectionState object with results for a complete election.
        """
        self.run_step()
        return self.state


class Bloc(Election):
    """
    Elects m candidates with the highest m-approval scores. The m-approval
    score of a candidate is equal to the number of voters who rank this
    candidate among their m top ranked candidates.

    **Attributes**

    `profile`
    :   PreferenceProfile to run election on.

    `seats`
    :   number of seats to be elected.

    `ballot_ties`
    :   (optional) resolves input ballot ties if True, else assumes ballots have no ties.
                    Defaults to True.

    `tiebreak`
    :   (optional) resolves procedural and final ties by specified tiebreak.
                    Defaults to random.

    **Methods**
    """

    def __init__(
        self,
        profile: PreferenceProfile,
        seats: int,
        ballot_ties: bool = True,
        tiebreak: str = "random",
    ):
        super().__init__(profile, ballot_ties)
        self.seats = seats
        self.tiebreak = tiebreak

    def run_step(self) -> ElectionState:
        """
        Conducts a Limited election to elect m-candidates.

        Returns:
           An ElectionState object for a Limited election.
        """
        limited_equivalent = Limited(
            profile=self.state.profile,
            seats=self.seats,
            k=self.seats,
            tiebreak=self.tiebreak,
        )
        outcome = limited_equivalent.run_election()
        self.state = outcome
        return outcome

    @lru_cache
    def run_election(self) -> ElectionState:
        """
        Runs complete Bloc election.

        Returns:
            An ElectionState object with results for a complete election.
        """
        self.run_step()
        return self.state


class SNTV(Election):
    """
    Single nontransferable vote (SNTV): Elects k candidates with the highest
    Plurality scores.

    **Attributes**

    `profile`
    :   PreferenceProfile to run election on.

    `seats`
    :   number of seats to be elected.

    `ballot_ties`
    :   (optional) resolves input ballot ties if True, else assumes ballots have no ties.
                    Defaults to True.

    `tiebreak`
    :   (optional) resolves procedural and final ties by specified tiebreak.
                    Defaults to random.

    **Methods**
    """

    def __init__(
        self,
        profile: PreferenceProfile,
        seats: int,
        ballot_ties: bool = True,
        tiebreak: str = "random",
    ):
        super().__init__(profile, ballot_ties)
        self.seats = seats
        self.tiebreak = tiebreak

    def run_step(self) -> ElectionState:
        """
        Conducts an SNTV election to elect candidates.

        Returns:
           An ElectionState object for a SNTV election.
        """
        limited_equivalent = Limited(
            profile=self.state.profile, seats=self.seats, k=1, tiebreak=self.tiebreak
        )
        outcome = limited_equivalent.run_election()
        self.state = outcome
        return outcome

    @lru_cache
    def run_election(self) -> ElectionState:
        """
        Runs complete SNTV election.

        Returns:
            An ElectionState object with results for a complete election.
        """
        self.run_step()
        return self.state


class SNTV_STV_Hybrid(Election):
    """
    Election method that first runs SNTV to a cutoff, then runs STV to
    pick a committee with a given number of seats.

    **Attributes**

    `profile`
    :   PreferenceProfile to run election on.

    `transfer`
    :   transfer method (e.g. fractional transfer).

    `r1_cutoff`
    :   first-round cutoff value.

    `seats`
    :   number of seats to be elected.

    `ballot_ties`
    :   (optional) resolves input ballot ties if True, else assumes ballots have no ties.
                    Defaults to True.

    `tiebreak`
    :   (optional) resolves procedural and final ties by specified tiebreak.
                    Defaults to random.

    **Methods**
    """

    def __init__(
        self,
        profile: PreferenceProfile,
        transfer: Callable,
        r1_cutoff: int,
        seats: int,
        ballot_ties: bool = True,
        tiebreak: str = "random",
    ):
        super().__init__(profile, ballot_ties)
        self.transfer = transfer
        self.r1_cutoff = r1_cutoff
        self.seats = seats
        self.tiebreak = tiebreak
        self.stage = "SNTV"  # SNTV, switches to STV, then Complete

    def run_step(self, stage: str) -> ElectionState:
        """
        Simulates one round an SNTV_STV election.

        Args:
            stage: Stage of the hybrid election, can be SNTV or STV.

        Returns:
           An ElectionState object for a given round.
        """
        profile = self.state.profile

        new_state = None
        if stage == "SNTV":
            round_state = SNTV(
                profile=profile, seats=self.r1_cutoff, tiebreak=self.tiebreak
            ).run_election()

            # The STV election will be run on the new election state
            # Therefore we should not add any winners, but rather
            # set the SNTV winners as remaining candidates and update pref profiles
            new_profile = PreferenceProfile(
                ballots=remove_cand(
                    set().union(*round_state.eliminated_cands), profile.get_ballots()
                )
            )
            new_state = ElectionState(
                curr_round=self.state.curr_round + 1,
                elected=list(),
                eliminated_cands=round_state.eliminated_cands,
                remaining=[set(new_profile.get_candidates())],
                profile=new_profile,
                scores=round_state.get_scores(round_state.curr_round),
                previous=self.state,
            )
        elif stage == "STV":
            round_state = STV(
                profile=profile,
                transfer=self.transfer,
                seats=self.seats,
                tiebreak=self.tiebreak,
            ).run_election()

            new_state = ElectionState(
                curr_round=self.state.curr_round + 1,
                elected=round_state.winners(),
                eliminated_cands=round_state.eliminated(),
                remaining=round_state.remaining,
                scores=round_state.get_scores(round_state.curr_round),
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

    @lru_cache
    def run_election(self) -> ElectionState:
        """
        Runs complete SNTV_STV election.

        Returns:
            An ElectionState object with results for a complete election.
        """
        while self.stage != "Complete":
            self.run_step(self.stage)
        return self.state  # type: ignore


class TopTwo(Election):
    """
    Eliminates all but the top two plurality vote getters, and then
    conducts a runoff between them, reallocating other ballots.

    **Attributes**

    `profile`
    :   PreferenceProfile to run election on.

    `seats`
    :   number of seats to be elected.

    `ballot_ties`
    :   (optional) resolves input ballot ties if True, else assumes ballots have no ties.
                    Defaults to True.

    `tiebreak`
    :   (optional) resolves procedural and final ties by specified tiebreak.
                    Defaults to random.

    **Methods**
    """

    def __init__(
        self,
        profile: PreferenceProfile,
        ballot_ties: bool = True,
        tiebreak: str = "random",
    ):
        super().__init__(profile, ballot_ties)
        self.tiebreak = tiebreak

    def run_step(self) -> ElectionState:
        """
        Conducts a TopTwo election for one seat with a cutoff of 2 for the runoff.

        Returns:
            An ElectionState object for the TopTwo election.
        """
        hybrid_equivalent = SNTV_STV_Hybrid(
            profile=self.state.profile,
            transfer=fractional_transfer,
            r1_cutoff=2,
            seats=1,
            tiebreak=self.tiebreak,
        )
        outcome = hybrid_equivalent.run_election()
        self.state = outcome
        return outcome

    @lru_cache
    def run_election(self) -> ElectionState:
        """
        Simulates a complete TopTwo election.

        Returns:
            An ElectionState object for a complete election.
        """
        self.run_step()
        return self.state


class DominatingSets(Election):
    """
    Finds tiers of candidates by dominating set, which is a set of candidates
    such that every candidate in the set wins head to head comparisons against
    candidates outside of it.

    **Attributes**

    `profile`
    :   PreferenceProfile to run election on.

    `ballot_ties`
    :   (optional) resolves input ballot ties if True, else assumes ballots have no ties.
                    Defaults to True.


    **Methods**
    """

    def __init__(self, profile: PreferenceProfile, ballot_ties: bool = True):
        super().__init__(profile, ballot_ties)

    def run_step(self) -> ElectionState:
        """
        Conducts a complete DominatingSets election as it is not a round-by-round
        system.

        Returns:
            An ElectionState object for a complete election.
        """
        pwc_graph = PairwiseComparisonGraph(self.state.profile)
        dominating_tiers = pwc_graph.dominating_tiers()
        if len(dominating_tiers) == 1:
            new_state = ElectionState(
                curr_round=self.state.curr_round + 1,
                elected=list(),
                eliminated_cands=dominating_tiers,
                remaining=list(),
                scores=pwc_graph.pairwise_dict,
                profile=PreferenceProfile(),
                previous=self.state,
            )
        else:
            new_state = ElectionState(
                curr_round=self.state.curr_round + 1,
                elected=[set(dominating_tiers[0])],
                eliminated_cands=dominating_tiers[1:],
                remaining=list(),
                scores=pwc_graph.pairwise_dict,
                profile=PreferenceProfile(),
                previous=self.state,
            )
        self.state = new_state
        return new_state

    @lru_cache
    def run_election(self) -> ElectionState:
        """
        Simulates a complete DominatingSets election.

        Returns:
            An ElectionState object for a complete election.
        """
        self.run_step()
        return self.state


class CondoBorda(Election):
    """
    Elects candidates ordered by dominating set, but breaks ties
    between candidates with Borda.

    **Attributes**

    `profile`
    :   PreferenceProfile to run election on.

    `seats`
    :   number of seats to be elected.

    `ballot_ties`
    :   (optional) resolves input ballot ties if True, else assumes ballots have no ties.
                Defaults to True.

    `tiebreak`
    :   (optional) resolves procedural and final ties by specified tiebreak.
                Defaults to random.

    **Methods**
    """

    def __init__(
        self,
        profile: PreferenceProfile,
        seats: int,
        ballot_ties: bool = True,
        tiebreak: str = "random",
    ):
        super().__init__(profile, ballot_ties)
        self.seats = seats
        self.tiebreak = tiebreak

    def run_step(self) -> ElectionState:
        """
        Conducts a complete Conda-Borda election as it is not a round-by-round
        system.

        Returns:
            An `ElectionState` object for a complete election.
        """
        pwc_graph = PairwiseComparisonGraph(self.state.profile)
        dominating_tiers = pwc_graph.dominating_tiers()
        ranking = tie_broken_ranking(
            ranking=dominating_tiers, profile=self.state.profile, tiebreak="borda"
        )
        elected, eliminated = elect_cands_from_set_ranking(
            ranking=ranking, seats=self.seats
        )

        new_state = ElectionState(
            curr_round=self.state.curr_round + 1,
            elected=elected,
            eliminated_cands=eliminated,
            remaining=list(),
            scores=pwc_graph.pairwise_dict,
            profile=PreferenceProfile(),
            previous=self.state,
        )
        self.state = new_state
        return new_state

    @lru_cache
    def run_election(self) -> ElectionState:
        """
        Simulates a complete Conda-Borda election.

        Returns:
            An ElectionState object for a complete election.
        """
        self.run_step()
        return self.state


class SequentialRCV(Election):
    """
    Class to conduct Sequential RCV election, in which votes are not transferred
    after a candidate has reached threshold, or been elected.

    **Attributes**

    `profile`
    :   PreferenceProfile to run election on.

    `seats`
    :   number of seats to be elected.

    `ballot_ties`
    :   (optional) resolves input ballot ties if True, else assumes ballots have no ties.
                Defaults to True.

    `tiebreak`
    :   (optional) resolves procedural and final ties by specified tiebreak.
                Defaults to random.

    **Methods**
    """

    def __init__(
        self,
        profile: PreferenceProfile,
        seats: int,
        ballot_ties: bool = True,
        tiebreak: str = "random",
    ):
        super().__init__(profile, ballot_ties)
        self.seats = seats
        self.tiebreak = tiebreak

    def run_step(self, old_profile: PreferenceProfile) -> ElectionState:
        """
        Simulates a single step of the sequential RCV contest or a full
        IRV election run on the current set of candidates.

         Returns:
           An ElectionState object for a given round.
        """
        old_election_state = self.state

        IRVrun = STV(
            old_profile, transfer=seqRCV_transfer, seats=1, tiebreak=self.tiebreak
        )
        old_election = IRVrun.run_election()
        elected_cand = old_election.winners()[0]

        # Removes elected candidate from Ballot List
        updated_ballots = remove_cand(elected_cand, old_profile.get_ballots())

        # Updates profile with removed candidates
        updated_profile = PreferenceProfile(ballots=updated_ballots)

        self.state = ElectionState(
            curr_round=old_election_state.curr_round + 1,
            elected=[elected_cand],
            profile=updated_profile,
            previous=old_election_state,
            scores=first_place_votes(updated_profile),
            remaining=old_election.remaining,
        )
        return self.state

    @lru_cache
    def run_election(self) -> ElectionState:
        """
        Simulates a complete sequential RCV contest.

        Returns:
            An ElectionState object for a complete election.
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
    vector is $(n, n-1, \dots, 1)$, where $n$ is the number of candidates.
    If a ballot is incomplete, the remaining points of the score vector
    are evenly distributed to the unlisted candidates (see `borda_scores` function in `utils`).

    **Attributes**

    `profile`
    :   PreferenceProfile to run election on.

    `seats`
    :   number of seats to be elected.

    `score_vector`
    :   (optional) weights assigned to candidate ranking, should be a list of `Fractions`.
                    Defaults to $(n,n-1,\dots,1)$.

    `ballot_ties`
    :   (optional) resolves input ballot ties if True, else assumes ballots have no ties.
                    Defaults to True.

    `tiebreak`
    :   (optional) resolves procedural and final ties by specified tiebreak.
                    Defaults to random.

    **Methods**
    """

    def __init__(
        self,
        profile: PreferenceProfile,
        seats: int,
        score_vector: Optional[list[Fraction]],
        ballot_ties: bool = True,
        tiebreak: str = "random",
    ):
        super().__init__(profile, ballot_ties)
        self.seats = seats
        self.tiebreak = tiebreak
        self.score_vector = score_vector

    def run_step(self) -> ElectionState:
        """
        Simulates a complete Borda contest as Borda is not a round-by-round
        system.

        Returns:
            An ElectionState object for a complete election.
        """
        borda_dict = borda_scores(
            profile=self.state.profile, score_vector=self.score_vector
        )

        ranking = scores_into_set_list(borda_dict)
        ranking = tie_broken_ranking(
            ranking=ranking, profile=self.state.profile, tiebreak=self.tiebreak
        )

        elected, eliminated = elect_cands_from_set_ranking(
            ranking=ranking, seats=self.seats
        )

        new_state = ElectionState(
            curr_round=self.state.curr_round + 1,
            elected=elected,
            eliminated_cands=eliminated,
            remaining=list(),
            scores=borda_dict,
            profile=PreferenceProfile(),
            previous=self.state,
        )
        self.state = new_state
        return new_state

    @lru_cache
    def run_election(self) -> ElectionState:
        """
        Simulates a complete Borda contest.

        Returns:
            An ElectionState object for a complete election.
        """
        self.run_step()
        return self.state


class Plurality(SNTV):
    """
    Simulates a single or multi-winner plurality election. Inherits
    methods from `SNTV` to run election.
    """

    def __init__(
        self,
        profile: PreferenceProfile,
        seats: int,
        ballot_ties: bool = True,
        tiebreak: str = "random",
    ):
        super().__init__(profile, ballot_ties)
        self.seats = seats
        self.tiebreak = tiebreak
