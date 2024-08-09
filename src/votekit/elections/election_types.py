from fractions import Fraction
import itertools as it
import random
import numpy as np
from typing import Callable, Optional, Union
from functools import lru_cache

from ..models import Election
from ..election_state import ElectionState
from ..graphs.pairwise_comparison_graph import PairwiseComparisonGraph
from ..pref_profile import PreferenceProfile
from ..ballot import Ballot
from .transfers import fractional_transfer
from ..utils import (
    compute_votes,
    remove_cand,
    scores_into_set_list,
    tie_broken_ranking,
    elect_cands_from_set_ranking,
    first_place_votes,
    compute_scores_from_vector,
    validate_score_vector,
    borda_scores,
    ballots_by_first_cand,
)


class STV(Election):
    """
    Class for multi-winner STV elections.

    Args:
        profile (PreferenceProfile):   PreferenceProfile to run election on.
        transfer (Callable): Transfer method (e.g. fractional transfer).
        seats (int): Number of seats to be elected.
        quota (str, optional): Formula to calculate quota. Accepts "droop" or "hare".
            Defaults to "droop".
        ballot_ties (bool, optional): Resolves input ballot ties if True, else assumes ballots have
            no ties. Defaults to True.
        tiebreak (Union[str, Callable], optional): Resolves procedural and final ties by specified
            tiebreak. Can either be a custom tiebreak function or a string. Supported strings are
            given in ``tie_broken_ranking`` documentation. The custom function must take as
            input two named parameters; ``ranking``, a list-of-sets ranking of candidates and
            ``profile``, the original ``PreferenceProfile``. It must return a list-of-sets
            ranking of candidates with no ties. Defaults to random tiebreak.

    Attributes:
        _profile (PreferenceProfile):   PreferenceProfile to run election on.
        state (ElectionState): Current state of the election.
        transfer (Callable): Transfer method (e.g. fractional transfer).
        seats (int): Number of seats to be elected.
        quota (str): Formula to calculate quota.
        tiebreak (Union[str, Callable]): Resolves procedural and final ties by specified
            tiebreak.
        threshold (int): Threshold number of votes to be elected.

    """

    def __init__(
        self,
        profile: PreferenceProfile,
        transfer: Callable,
        seats: int,
        quota: str = "droop",
        ballot_ties: bool = True,
        tiebreak: Union[str, Callable] = "random",
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
            int: Value of the threshold.
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
            bool: True if number of seats has not been met, False otherwise.
        """
        cands_elected = 0
        for s in self.state.winners():
            cands_elected += len(s)
        return cands_elected < self.seats

    def run_step(self) -> ElectionState:
        """
        Simulates one round an STV election.

        Returns:
           ElectionState: An ElectionState object for a round.
        """
        remaining = self.state.profile.get_candidates()
        ballots = self.state.profile.get_ballots()
        round_votes, plurality_score = compute_votes(remaining, ballots)

        elected = []
        eliminated = []

        # if number of remaining candidates equals number of remaining seats,
        # everyone is elected
        if len(remaining) == self.seats - len(
            [c for s in self.state.winners() for c in s]
        ):
            elected = [{cand} for cand, _ in round_votes]
            remaining = []
            ballots = []

        # elect all candidates who crossed threshold
        elif round_votes[0].votes >= self.threshold:
            # partition ballots by first place candidate
            cand_to_ballot = ballots_by_first_cand(remaining, ballots)
            new_ballots = []
            for candidate, votes in round_votes:
                if votes >= self.threshold:
                    elected.append({candidate})
                    remaining.remove(candidate)
                    # only transfer on ballots where winner is first
                    new_ballots += self.transfer(
                        candidate,
                        cand_to_ballot[candidate],
                        plurality_score,
                        self.threshold,
                    )

            # add in remaining ballots where non-winners are first
            for cand in remaining:
                new_ballots += cand_to_ballot[cand]

            # remove winners from all ballots
            ballots = remove_cand([c for s in elected for c in s], new_ballots)

        # since no one has crossed threshold, eliminate one of the people
        # with least first place votes
        else:
            lp_candidates = [
                candidate
                for candidate, votes in round_votes
                if votes == round_votes[-1].votes
            ]

            if isinstance(self.tiebreak, str):
                lp_cand = tie_broken_ranking(
                    ranking=[set(lp_candidates)],
                    profile=self.state.profile,
                    tiebreak=self.tiebreak,
                )[-1]
            else:
                lp_cand = self.tiebreak(
                    ranking=[set(lp_candidates)], profile=self.state.profile
                )[-1]

            eliminated.append(lp_cand)
            ballots = remove_cand(lp_cand, ballots)
            remaining.remove(next(iter(lp_cand)))

        # sorts remaining based on their current first place votes
        _, score_dict = compute_votes(remaining, ballots)
        remaining = scores_into_set_list(score_dict, remaining)

        # sort candidates by vote share if multiple are elected
        if len(elected) >= 1:
            elected = scores_into_set_list(
                plurality_score, [c for s in elected for c in s]
            )

        # Make sure list-of-sets have non-empty elements
        elected = [s for s in elected if s != set()]
        eliminated = [s for s in eliminated if s != set()]

        self.state = ElectionState(
            curr_round=self.state.curr_round + 1,
            elected=elected,
            eliminated_cands=eliminated,
            remaining=remaining,
            scores=score_dict,
            profile=PreferenceProfile(ballots=ballots),
            previous=self.state,
        )
        return self.state

    @lru_cache
    def run_election(self) -> ElectionState:
        """
        Runs complete STV election.

        Returns:
            ElectionState: An ElectionState object with results for a complete election.
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

    Args:
        profile (PreferenceProfile):   PreferenceProfile to run election on.
        seats (int): Number of seats to be elected.
        k (int): The number of top ranked candidates to consider as approved.
        ballot_ties (bool, optional): Resolves input ballot ties if True, else assumes ballots have
            no ties. Defaults to True.
        tiebreak (Union[str, Callable], optional): Resolves procedural and final ties by specified
            tiebreak. Can either be a custom tiebreak function or a string. Supported strings are
            given in ``tie_broken_ranking`` documentation. The custom function must take as
            input two named parameters; ``ranking``, a list-of-sets ranking of candidates and
            ``profile``, the original ``PreferenceProfile``. It must return a list-of-sets
            ranking of candidates with no ties. Defaults to random tiebreak.

    Attributes:
        _profile (PreferenceProfile):   PreferenceProfile to run election on.
        state (ElectionState): Current state of the election.
        seats (int): Number of seats to be elected.
        k (int): The number of top ranked candidates to consider as approved.
        tiebreak (Union[str, Callable]): Resolves procedural and final ties by specified
            tiebreak.

    """

    def __init__(
        self,
        profile: PreferenceProfile,
        seats: int,
        k: int,
        ballot_ties: bool = True,
        tiebreak: Union[Callable, str] = "random",
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
           ElectionState: An ElectionState object for a Limited election.
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
                    # NOTE: I think I notice a bug here! If a single candidate is named 'ABC' this
                    #       counts three candidates 'A', 'B', and 'C' -- i.e. it breaks up strings
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

        if isinstance(self.tiebreak, str):
            ranking = tie_broken_ranking(
                ranking=ranking, profile=self.state.profile, tiebreak=self.tiebreak
            )
        else:
            ranking = self.tiebreak(ranking=ranking, profile=self.state.profile)

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
            ElectionState: An ElectionState object with results for a complete election.
        """
        self.run_step()
        return self.state


class Bloc(Limited):
    """
    Elects m candidates with the highest m-approval scores. Specific case of Limited election
    where k = m.

    Args:
        profile (PreferenceProfile):   PreferenceProfile to run election on.
        seats (int): Number of seats to be elected.
        ballot_ties (bool, optional): Resolves input ballot ties if True, else assumes ballots have
            no ties. Defaults to True.
        tiebreak (Union[str, Callable], optional): Resolves procedural and final ties by specified
            tiebreak. Can either be a custom tiebreak function or a string. Supported strings are
            given in ``tie_broken_ranking`` documentation. The custom function must take as
            input two named parameters; ``ranking``, a list-of-sets ranking of candidates and
            ``profile``, the original ``PreferenceProfile``. It must return a list-of-sets
            ranking of candidates with no ties. Defaults to random tiebreak.

    Attributes:
        _profile (PreferenceProfile):   PreferenceProfile to run election on.
        state (ElectionState): Current state of the election.
        seats (int): Number of seats to be elected.
        tiebreak (Union[str, Callable]): Resolves procedural and final ties by specified
            tiebreak.
    """

    def __init__(
        self,
        profile: PreferenceProfile,
        seats: int,
        ballot_ties: bool = True,
        tiebreak: Union[Callable, str] = "random",
    ):
        super().__init__(profile, seats, seats, ballot_ties, tiebreak)


class SNTV(Limited):
    """
    Single nontransferable vote (SNTV): Elects k candidates with the highest
    Plurality scores. Equivalent to Limited with k=1.

    Args:
        profile (PreferenceProfile):   PreferenceProfile to run election on.
        seats (int): Number of seats to be elected.
        ballot_ties (bool, optional): Resolves input ballot ties if True, else assumes ballots have
            no ties. Defaults to True.
        tiebreak (Union[str, Callable], optional): Resolves procedural and final ties by specified
            tiebreak. Can either be a custom tiebreak function or a string. Supported strings are
            given in ``tie_broken_ranking`` documentation. The custom function must take as
            input two named parameters; ``ranking``, a list-of-sets ranking of candidates and
            ``profile``, the original ``PreferenceProfile``. It must return a list-of-sets
            ranking of candidates with no ties. Defaults to random tiebreak.

    Attributes:
        _profile (PreferenceProfile):   PreferenceProfile to run election on.
        state (ElectionState): Current state of the election.
        seats (int): Number of seats to be elected.
        tiebreak (Union[str, Callable]): Resolves procedural and final ties by specified
            tiebreak.
    """

    def __init__(
        self,
        profile: PreferenceProfile,
        seats: int,
        ballot_ties: bool = True,
        tiebreak: Union[Callable, str] = "random",
    ):
        super().__init__(profile, seats, 1, ballot_ties, tiebreak)


class Plurality(SNTV):
    """
    Simulates a single or multi-winner plurality election. Wrapper for SNTV.
    """


class SNTV_STV_Hybrid(Election):
    """
    Election method that first runs SNTV to a cutoff number of candidates, then runs STV to
    pick a committee with a given number of seats.

    Args:
        profile (PreferenceProfile):   PreferenceProfile to run election on.
        transfer (Callable): Transfer method (e.g. fractional transfer).
        r1_cutoff (int): First round cutoff value.
        seats (int): Number of seats to be elected.
        ballot_ties (bool, optional): Resolves input ballot ties if True, else assumes ballots have
            no ties. Defaults to True.
        tiebreak (Union[str, Callable], optional): Resolves procedural and final ties by specified
            tiebreak. Can either be a custom tiebreak function or a string. Supported strings are
            given in ``tie_broken_ranking`` documentation. The custom function must take as
            input two named parameters; ``ranking``, a list-of-sets ranking of candidates and
            ``profile``, the original ``PreferenceProfile``. It must return a list-of-sets
            ranking of candidates with no ties. Defaults to random tiebreak.

    Attributes:
        _profile (PreferenceProfile):   PreferenceProfile to run election on.
        state (ElectionState): Current state of the election.
        transfer (Callable): Transfer method (e.g. fractional transfer).
        r1_cutoff (int): First round cutoff value.
        seats (int): Number of seats to be elected.
        tiebreak (Union[str, Callable]): Resolves procedural and final ties by specified
            tiebreak.


    """

    def __init__(
        self,
        profile: PreferenceProfile,
        transfer: Callable,
        r1_cutoff: int,
        seats: int,
        ballot_ties: bool = True,
        tiebreak: Union[Callable, str] = "random",
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
            stage (str): Stage of the hybrid election, can be "SNTV" or "STV".

        Returns:
            ElectionState: An ElectionState object for a given round.
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
            ElectionState: An ElectionState object with results for a complete election.
        """
        while self.stage != "Complete":
            self.run_step(self.stage)
        return self.state  # type: ignore


class TopTwo(Election):
    """
    Eliminates all but the top two plurality vote getters, and then
    conducts a runoff between them, reallocating other ballots.

    Args:
        profile (PreferenceProfile):   PreferenceProfile to run election on.
        seats (int): Number of seats to be elected.
        ballot_ties (bool, optional): Resolves input ballot ties if True, else assumes ballots have
            no ties. Defaults to True.
        tiebreak (Union[str, Callable], optional): Resolves procedural and final ties by specified
            tiebreak. Can either be a custom tiebreak function or a string. Supported strings are
            given in ``tie_broken_ranking`` documentation. The custom function must take as
            input two named parameters; ``ranking``, a list-of-sets ranking of candidates and
            ``profile``, the original ``PreferenceProfile``. It must return a list-of-sets
            ranking of candidates with no ties. Defaults to random tiebreak.

    Attributes:
        _profile (PreferenceProfile):   PreferenceProfile to run election on.
        state (ElectionState): Current state of the election.
        seats (int): Number of seats to be elected.
        tiebreak (Union[str, Callable]): Resolves procedural and final ties by specified
            tiebreak.
    """

    def __init__(
        self,
        profile: PreferenceProfile,
        ballot_ties: bool = True,
        tiebreak: Union[str, Callable] = "random",
    ):
        super().__init__(profile, ballot_ties)
        self.tiebreak = tiebreak

    def run_step(self) -> ElectionState:
        """
        Conducts a TopTwo election for one seat with a cutoff of 2 for the runoff.

        Returns:
            ElectionState: An ElectionState object for the TopTwo election.
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
            ElectionState: An ElectionState object for a complete election.
        """
        self.run_step()
        return self.state


class DominatingSets(Election):
    """
    Finds tiers of candidates by dominating set, which is a set of candidates
    such that every candidate in the set wins head to head comparisons against
    candidates outside of it. Elects all candidates in the top tier.

    Args:
        profile (PreferenceProfile):   PreferenceProfile to run election on.
        ballot_ties (bool, optional): Resolves input ballot ties if True, else assumes ballots have
            no ties. Defaults to True.

    Attributes:
        _profile (PreferenceProfile):   PreferenceProfile to run election on.
        state (ElectionState): Current state of the election.
    """

    def __init__(self, profile: PreferenceProfile, ballot_ties: bool = True):
        super().__init__(profile, ballot_ties)

    def run_step(self) -> ElectionState:
        """
        Conducts a complete DominatingSets election as it is not a round-by-round
        system.

        Returns:
            ElectionState: An ElectionState object for a complete election.
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
            ElectionState: An ElectionState object for a complete election.
        """
        self.run_step()
        return self.state


class CondoBorda(Election):
    """
    Finds tiers of candidates by dominating set, which is a set of candidates
    such that every candidate in the set wins head to head comparisons against
    candidates outside of it. Elects as many candidates as specified, in order from
    first to last dominating set. If the number of seats left is smaller than the next
    dominating set, CondoBorda breaks ties between candidates with Borda.

    Args:
        profile (PreferenceProfile):   PreferenceProfile to run election on.
        seats (int): Number of seats to be elected.
        ballot_ties (bool, optional): Resolves input ballot ties if True, else assumes ballots have
            no ties. Defaults to True.
        tiebreak (Union[str, Callable], optional): Resolves procedural and final ties by specified
            tiebreak. Can either be a custom tiebreak function or a string. Supported strings are
            given in ``tie_broken_ranking`` documentation. The custom function must take as
            input two named parameters; ``ranking``, a list-of-sets ranking of candidates and
            ``profile``, the original ``PreferenceProfile``. It must return a list-of-sets
            ranking of candidates with no ties. Defaults to random tiebreak.

    Attributes:
        _profile (PreferenceProfile):   PreferenceProfile to run election on.
        state (ElectionState): Current state of the election.
        seats (int): Number of seats to be elected.
        tiebreak (Union[str, Callable]): Resolves procedural and final ties by specified
            tiebreak.
    """

    def __init__(
        self,
        profile: PreferenceProfile,
        seats: int,
        ballot_ties: bool = True,
        tiebreak: Union[Callable, str] = "random",
    ):
        super().__init__(profile, ballot_ties)
        self.seats = seats
        self.tiebreak = tiebreak

    def run_step(self) -> ElectionState:
        """
        Conducts a complete Conda-Borda election as it is not a round-by-round
        system.

        Returns:
            ElectionState: An `ElectionState` object for a complete election.
        """
        pwc_graph = PairwiseComparisonGraph(self.state.profile)
        dominating_tiers = pwc_graph.dominating_tiers()

        if isinstance(self.tiebreak, str):
            ranking = tie_broken_ranking(
                ranking=dominating_tiers, profile=self.state.profile, tiebreak="borda"
            )
        else:
            ranking = self.tiebreak(
                ranking=dominating_tiers, profile=self.state.profile
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
            ElectionState: An ElectionState object for a complete election.
        """
        self.run_step()
        return self.state


class SequentialRCV(Election):
    """
    Class to conduct Sequential RCV election, in which votes are not transferred
    after a candidate has reached threshold, or been elected.

    Args:
        profile (PreferenceProfile):   PreferenceProfile to run election on.
        seats (int): Number of seats to be elected.
        ballot_ties (bool, optional): Resolves input ballot ties if True, else assumes ballots have
            no ties. Defaults to True.
        tiebreak (Union[str, Callable], optional): Resolves procedural and final ties by specified
            tiebreak. Can either be a custom tiebreak function or a string. Supported strings are
            given in ``tie_broken_ranking`` documentation. The custom function must take as
            input two named parameters; ``ranking``, a list-of-sets ranking of candidates and
            ``profile``, the original ``PreferenceProfile``. It must return a list-of-sets
            ranking of candidates with no ties. Defaults to random tiebreak.

    Attributes:
        _profile (PreferenceProfile):   PreferenceProfile to run election on.
        state (ElectionState): Current state of the election.
        seats (int): Number of seats to be elected.
        tiebreak (Union[str, Callable]): Resolves procedural and final ties by specified
            tiebreak.
    """

    def __init__(
        self,
        profile: PreferenceProfile,
        seats: int,
        ballot_ties: bool = True,
        tiebreak: Union[Callable, str] = "random",
    ):
        super().__init__(profile, ballot_ties)
        self.seats = seats
        self.tiebreak = tiebreak

    def run_step(self, old_profile: PreferenceProfile) -> ElectionState:
        """
        Simulates a single step of the sequential RCV contest or a full
        IRV election run on the current set of candidates.

        Returns:
            ElectionState: An ElectionState object.
        """
        old_election_state = self.state

        IRVrun = STV(
            old_profile,
            transfer=(lambda winner, ballots, votes, threshold: ballots),
            seats=1,
            tiebreak=self.tiebreak,
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
            ElectionState: An ElectionState object for a complete election.
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
    vector is :math:`(n, n-1, \dots, 1)`, where `n` is the number of candidates.
    If a ballot is incomplete, the remaining points of the score vector
    are evenly distributed to the unlisted candidates (see ``borda_scores`` function in ``utils``).

    Args:
        profile (PreferenceProfile):   PreferenceProfile to run election on.
        seats (int): Number of seats to be elected.
        score_vector (list[Fraction], optional): Weights assigned to candidate ranking.
            Defaults to :math:`(n,n-1,\dots,1)`.
        ballot_ties (bool, optional): Resolves input ballot ties if True, else assumes ballots have
            no ties. Defaults to True.
        tiebreak (Union[str, Callable], optional): Resolves procedural and final ties by specified
            tiebreak. Can either be a custom tiebreak function or a string. Supported strings are
            given in ``tie_broken_ranking`` documentation. The custom function must take as
            input two named parameters; ``ranking``, a list-of-sets ranking of candidates and
            ``profile``, the original ``PreferenceProfile``. It must return a list-of-sets
            ranking of candidates with no ties. Defaults to random tiebreak.

    Attributes:
        _profile (PreferenceProfile):   PreferenceProfile to run election on.
        state (ElectionState): Current state of the election.
        seats (int): Number of seats to be elected.
        score_vector (list[Fraction], optional): Weights assigned to candidate ranking.
        tiebreak (Union[str, Callable]): Resolves procedural and final ties by specified
            tiebreak.
    """

    def __init__(
        self,
        profile: PreferenceProfile,
        seats: int,
        score_vector: Optional[list[Fraction]] = None,
        ballot_ties: bool = True,
        tiebreak: Union[Callable, str] = "random",
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
            ElectionState: An ElectionState object for a complete election.
        """
        borda_dict = borda_scores(
            profile=self.state.profile, score_vector=self.score_vector
        )

        ranking = scores_into_set_list(borda_dict)

        if isinstance(self.tiebreak, str):
            ranking = tie_broken_ranking(
                ranking=ranking, profile=self.state.profile, tiebreak=self.tiebreak
            )
        else:
            ranking = self.tiebreak(ranking=ranking, profile=self.state.profile)

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
            ElectionState: An ElectionState object for a complete election.
        """
        self.run_step()
        return self.state


class IRV(STV):
    """
    A class for conducting IRV elections, which are mathematically equivalent to STV for one seat.

    Args:
        profile (PreferenceProfile):   PreferenceProfile to run election on.
        seats (int): Number of seats to be elected.
        quota (str, optional): Formula to calculate quota. Accepts "droop" or "hare".
            Defaults to "droop".
        ballot_ties (bool, optional): Resolves input ballot ties if True, else assumes ballots have
            no ties. Defaults to True.
        tiebreak (Union[str, Callable], optional): Resolves procedural and final ties by specified
            tiebreak. Can either be a custom tiebreak function or a string. Supported strings are
            given in ``tie_broken_ranking`` documentation. The custom function must take as
            input two named parameters; ``ranking``, a list-of-sets ranking of candidates and
            ``profile``, the original ``PreferenceProfile``. It must return a list-of-sets
            ranking of candidates with no ties. Defaults to random tiebreak.

    Attributes:
        _profile (PreferenceProfile):   PreferenceProfile to run election on.
        state (ElectionState): Current state of the election.
        seats (int): Number of seats to be elected.
        quota (str): Formula to calculate quota.
        tiebreak (Union[str, Callable]): Resolves procedural and final ties by specified
            tiebreak.
        threshold (int): Threshold number of votes to be elected.

    """

    def __init__(
        self,
        profile: PreferenceProfile,
        quota: str = "droop",
        ballot_ties: bool = True,
        tiebreak: Union[Callable, str] = "random",
    ):
        # let parent class handle the construction
        super().__init__(
            profile=profile,
            ballot_ties=ballot_ties,
            seats=1,
            tiebreak=tiebreak,
            quota=quota,
            transfer=fractional_transfer,
        )


class HighestScore(Election):
    """
    Conducts an election based on points from score vector.
    Chooses the m candidates with highest scores.
    Ties are broken by randomly permuting the tied candidates.

    Args:
        profile (PreferenceProfile):   PreferenceProfile to run election on.
        seats (int): Number of seats to be elected.
        score_vector (list[float]): List of floats where `i`th entry denotes the number of points
            given to candidates ranked in position `i`.
        ballot_ties (bool, optional): Resolves input ballot ties if True, else assumes ballots have
            no ties. Defaults to True.
        tiebreak (Union[str, Callable], optional): Resolves procedural and final ties by specified
            tiebreak. Can either be a custom tiebreak function or a string. Supported strings are
            given in ``tie_broken_ranking`` documentation. The custom function must take as
            input two named parameters; ``ranking``, a list-of-sets ranking of candidates and
            ``profile``, the original ``PreferenceProfile``. It must return a list-of-sets
            ranking of candidates with no ties. Defaults to random tiebreak.

    Attributes:
        _profile (PreferenceProfile):   PreferenceProfile to run election on.
        state (ElectionState): Current state of the election.
        seats (int): Number of seats to be elected.
        score_vector (list[float]): List of floats where `i`th entry denotes the number of points
            given to candidates ranked in position `i`.
        tiebreak (Union[str, Callable]): Resolves procedural and final ties by specified
            tiebreak.

    """

    def __init__(
        self,
        profile: PreferenceProfile,
        seats: int,
        score_vector: list[float],
        tiebreak: Union[Callable, str] = "random",
        ballot_ties: bool = False,
    ):
        super().__init__(profile, ballot_ties)
        # check for valid score vector
        validate_score_vector(score_vector)

        self.seats = seats
        self.score_vector = score_vector
        self.tiebreak = tiebreak

    def run_step(self):
        """
        Simulates a complete Borda contest as Borda is not a round-by-round
        system.

        Returns:
            ElectionState: An ElectionState object for a complete election.
        """
        # a dictionary whose keys are candidates and values are scores
        vote_tallies = compute_scores_from_vector(
            profile=self.state.profile, score_vector=self.score_vector
        )

        # translate scores into ranking of candidates, tie break
        ranking = scores_into_set_list(score_dict=vote_tallies)

        if isinstance(self.tiebreak, str):
            untied_ranking = tie_broken_ranking(
                ranking=ranking, profile=self.state.profile, tiebreak=self.tiebreak
            )
        else:
            untied_ranking = self.tiebreak(ranking=ranking, profile=self.state.profile)

        elected, eliminated = elect_cands_from_set_ranking(
            ranking=untied_ranking, seats=self.seats
        )

        self.state = ElectionState(
            curr_round=1,
            elected=elected,
            eliminated_cands=eliminated,
            remaining=[],
            profile=self.state.profile,
            previous=self.state,
        )
        return self.state

    @lru_cache
    def run_election(self):
        """
        Simulates a complete Borda contest.

        Returns:
            ElectionState: An ElectionState object for a complete election.
        """
        self.run_step()
        return self.state


class Cumulative(HighestScore):
    """
    Voting system where voters are allowed to vote for candidates with multiplicity.
    Each ranking position should have one candidate, and every candidate ranked will receive
    one point, i.e., the score vector is :math:`(1,\dots,1)`.

    Args:
        profile (PreferenceProfile):   PreferenceProfile to run election on.
        seats (int): Number of seats to be elected.
        ballot_ties (bool, optional): Resolves input ballot ties if True, else assumes ballots have
            no ties. Defaults to True.
        tiebreak (Union[str, Callable], optional): Resolves procedural and final ties by specified
            tiebreak. Can either be a custom tiebreak function or a string. Supported strings are
            given in ``tie_broken_ranking`` documentation. The custom function must take as
            input two named parameters; ``ranking``, a list-of-sets ranking of candidates and
            ``profile``, the original ``PreferenceProfile``. It must return a list-of-sets
            ranking of candidates with no ties. Defaults to random tiebreak.

    Attributes:
        _profile (PreferenceProfile):   PreferenceProfile to run election on.
        state (ElectionState): Current state of the election.
        seats (int): Number of seats to be elected.
        tiebreak (Union[str, Callable]): Resolves procedural and final ties by specified
            tiebreak.
    """

    def __init__(
        self,
        profile: PreferenceProfile,
        seats: int,
        ballot_ties: bool = True,
        tiebreak: Union[str, Callable] = "random",
    ):
        longest_ballot = 0
        for ballot in profile.ballots:
            if len(ballot.ranking) > longest_ballot:
                longest_ballot = len(ballot.ranking)

        score_vector = [1.0 for _ in range(longest_ballot)]
        super().__init__(
            profile=profile,
            ballot_ties=ballot_ties,
            score_vector=score_vector,
            seats=seats,
            tiebreak=tiebreak,
        )


class RandomDictator(Election):
    """
    Choose a winner randomly from the distribution of first place votes. For multi-winner elections
    repeat this process for every winner, removing that candidate from every voter's ballot
    once they have been elected.

    Args:
      profile (PreferenceProfile): PreferenceProfile to run election on.
      m (int): Number of seats to elect.

    Attributes:
      _profile (PreferenceProfile): PreferenceProfile to run election on.
      m (int): Number of seats to be elected.

    """

    def __init__(self, profile: PreferenceProfile, m: int):
        super().__init__(profile, ballot_ties=False)
        if m <= 0 or m > len(profile.get_candidates()):
            raise ValueError("Invalid number of candidates to elect")

        self.m = m

    def next_round(self) -> bool:
        """
        Determines if another round is needed.

        Returns:
            (bool) True if number of seats has not been met, False otherwise.
        """
        cands_elected = [len(s) for s in self.state.winners()]
        return sum(cands_elected) < self.m

    def run_step(self):
        """
        If m candidates have not yet been elected:
        finds a single winning candidate to add to the list of elected
        candidates by sampling from the distribution of first place votes.
        Removes that candidate from all ballots in the preference profile.

        Returns:
            ElectionState: An ElectionState object for a complete election.
        """
        if self.next_round():
            remaining = self.state.profile.get_candidates()

            ballots = self.state.profile.ballots
            weights = [b.weight for b in ballots]
            random_ballot = random.choices(
                self.state.profile.ballots, weights=weights, k=1
            )[0]

            winning_candidate = random.choice(list(random_ballot.ranking[0]))
            elected = [{winning_candidate}]
            new_ballots = remove_cand(winning_candidate, self.state.profile.ballots)
            new_profile = PreferenceProfile(ballots=new_ballots)
            remaining = [{c} for c in remaining if c != winning_candidate]

            self.state = ElectionState(
                curr_round=self.state.curr_round + 1,
                elected=elected,
                eliminated_cands=[],
                remaining=remaining,
                profile=new_profile,
                previous=self.state,
            )
            return self.state

    def run_election(self):
        # run steps until we elect the required number of candidates
        while self.next_round():
            self.run_step()

        return self.state


class BoostedRandomDictator(Election):
    """
    Modified random dictator where
        - With probability (1 - 1/(n_candidates - 1))
        choose a winner randomly from the distribution of first place votes.
        - With probability 1/(n_candidates - 1)
          Choose a winner via a proportional to squares rule.

    For multi-winner elections
    repeat this process for every winner, removing that candidate from every
    voter's ballot once they have been elected.

    Args:
      profile (PreferenceProfile): PreferenceProfile to run election on.
      m (int): Number of seats to elect.

    Attributes:
      _profile (PreferenceProfile): PreferenceProfile to run election on.
      m (int): Number of seats to be elected.

    """

    def __init__(self, profile: PreferenceProfile, m: int):
        super().__init__(profile, ballot_ties=False)
        if m <= 0 or m > len(profile.get_candidates()):
            raise ValueError("Invalid number of candidates to elect")

        self.m = m

    def next_round(self) -> bool:
        """
        Determines if another round is needed.

        Returns:
            (bool) True if number of seats has not been met, False otherwise.
        """
        cands_elected = [len(s) for s in self.state.winners()]
        return sum(cands_elected) < self.m

    def run_step(self):
        """
        If m candidates have not yet been elected:
        finds a single winning candidate to add to the list of elected
        candidates by sampling from the distribution induced by the combination
        of random dictator and proportional to squares election rules.
        Removes that candidate from all ballots in the preference profile.

        Returns:
            ElectionState: An ElectionState object for a complete election.
        """
        if self.next_round():
            remaining = self.state.profile.get_candidates()
            u = random.uniform(0, 1)

            if len(remaining) == 1:
                winning_candidate = remaining[0]

            elif u <= 1 / (len(remaining) - 1):
                candidate_votes = first_place_votes(self.state.profile, to_float=True)
                p = np.array([i for i in candidate_votes.values()])
                p /= float(self.state.profile.num_ballots())
                p = np.power(p, 2)
                p /= np.sum(p)
                winning_candidate = np.random.choice(remaining, p=p)

            else:
                ballots = self.state.profile.ballots
                weights = [b.weight for b in ballots]
                random_ballot = random.choices(
                    self.state.profile.ballots, weights=weights, k=1
                )[0]
                winning_candidate = random.choice(list(random_ballot.ranking[0]))

            elected = [{winning_candidate}]
            new_ballots = remove_cand(winning_candidate, self.state.profile.ballots)
            new_profile = PreferenceProfile(ballots=new_ballots)
            remaining = [{c} for c in remaining if c != winning_candidate]

            self.state = ElectionState(
                curr_round=self.state.curr_round + 1,
                elected=elected,
                eliminated_cands=[],
                remaining=remaining,
                profile=new_profile,
                previous=self.state,
            )
            return self.state

    def run_election(self):
        while self.next_round():
            self.run_step()

        return self.state


class PluralityVeto(Election):
    """
    Scores each candidate by their plurality (number of first) place votes,
    then in a randomized order it lets each voter decrement the score of their
    least favorite candidate. The candidate with the largest score at the end is
    then chosen as the winner.

    Args:
      profile (PreferenceProfile): PreferenceProfile to run election on. Note that
        ballots muts have integer weights to be considered valid for this mechanism.
      m (int): Number of seats to elect.

    Attributes:
      _profile (PreferenceProfile): PreferenceProfile to run election on.
      m (int): Number of seats to be elected.
      random_order (list[int]): randomly shuffled order of voter indices
      candidate_approvals (dict[str,int]): dictionary tracking current candidate scores


    """

    def __init__(self, profile: PreferenceProfile, m: int):
        super().__init__(profile, ballot_ties=False)
        if m <= 0 or m > len(profile.get_candidates()):
            raise ValueError("Invalid number of candidates to elect.")

        self.m = m

        # Decondense ballots
        ballots = self.state.profile.get_ballots()
        new_ballots = [Ballot() for i in range(int(self.state.profile.num_ballots()))]
        bidx = 0
        for i, b in enumerate(ballots):
            if not int(b.weight) == b.weight:
                raise ValueError("Ballots must have integer weight.")

            for j in range(int(b.weight)):
                new_ballots[bidx] = Ballot(b.ranking, weight=Fraction(1, 1))
                bidx += 1

        new_profile = PreferenceProfile(
            ballots=new_ballots, candidates=self.state.profile.get_candidates()
        )

        self.state = ElectionState(
            curr_round=self.state.curr_round,
            elected=[],
            eliminated_cands=[],
            remaining=self.state.remaining,
            profile=new_profile,
            previous=self.state.previous,
        )

        self.random_order = None
        self.candidate_approvals = None

    def next_round(self) -> bool:
        """
        Determines if another round is needed.

        Returns:
            (bool) True if number of seats has not been met, False otherwise.
        """
        cands_elected = [len(s) for s in self.state.winners()]
        return sum(cands_elected) < self.m

    def run_step(self):
        """
        If m candidates have not yet been elected:
        if the current round is 0, count plurality scores and remove all
        candidates with a score of 0 to eliminated. Otherwise, this method
        eliminates a single candidate by decrementing scores of candidates on
        the current ticket until someone falls to a score of 0. Once we find
        that there are exactly m candidates remaining on the ticket, elect all
        of them.

        Returns:
            ElectionState: An ElectionState object for a complete election.
        """
        if self.next_round():
            candidates = self.state.profile.get_candidates()
            ballots = self.state.profile.get_ballots()

            if len(candidates) == self.m:
                # move all to elected, this is the last round
                elected = [{c} for c in candidates]
                self.state = ElectionState(
                    curr_round=self.state.curr_round + 1,
                    elected=elected,
                    eliminated_cands=[],
                    remaining=[],
                    profile=self.state.profile,
                    previous=self.state,
                )

            else:
                if self.state.curr_round == 0:
                    eliminated = [
                        c for c, score in self.candidate_approvals.items() if score <= 0
                    ]

                else:
                    eliminated = ["c"]
                    last_idx = 0
                    for i in range(len(self.random_order)):
                        last_idx = i
                        r = self.random_order[i]
                        ballot = ballots[r]
                        # Need to make a random choice between last place candidates,
                        # because voters should only get one veto
                        # if len(ballot.ranking) > 0:
                        least_preferred = random.choice(list(ballot.ranking[-1]))
                        self.candidate_approvals[least_preferred] -= 1
                        if self.candidate_approvals[least_preferred] <= 0:
                            eliminated[0] = least_preferred
                            # Only want to eliminate one candidate per round so we break here
                            break

                    # Update the randomized order so that
                    # we can continue where we left off next round
                    self.random_order = (
                        self.random_order[last_idx + 1 :]
                        + self.random_order[: last_idx + 1]
                    )

                new_ballots = remove_cand(eliminated, self.state.profile.ballots)
                new_profile = PreferenceProfile(ballots=new_ballots)
                eliminated = [{c} for c in eliminated]
                remaining = [{c} for c in new_profile.get_candidates()]
                self.state = ElectionState(
                    curr_round=self.state.curr_round + 1,
                    elected=[],
                    eliminated_cands=eliminated,
                    remaining=remaining,
                    profile=new_profile,
                    previous=self.state,
                )

            return self.state

    def run_election(self):
        self.random_order = list(range(int(self.state.profile.num_ballots())))
        np.random.shuffle(self.random_order)
        self.candidate_approvals = first_place_votes(self.state.profile, to_float=True)

        while self.next_round():
            self.run_step()

        return self.state
