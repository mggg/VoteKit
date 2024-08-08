from .abstract_ranking import RankingElection
from ...transfers import fractional_transfer
from ....pref_profile import PreferenceProfile
from ...election_state import ElectionState
from ....ballot import Ballot
from ....utils import (
    remove_cand,
    first_place_votes,
    ballots_by_first_cand,
    tiebreak_set,
    elect_cands_from_set_ranking,
    score_dict_to_ranking,
)
from typing import Optional, Callable, Union
from fractions import Fraction


class STV(RankingElection):
    """
    STV elections. Currently implements simultaneous election. All ballots must have no
    ties.

    Args:
        profile (PreferenceProfile):   PreferenceProfile to run election on.
        m (int, optional): Number of seats to be elected. Defaults to 1.
        transfer (Callable[[str, Union[Fraction, float], Union[tuple[Ballot], list[Ballot]], int], tuple[Ballot,...]], optional):
        Transfer method. Defaults to fractional transfer.
            Function signature is elected candidate, their number of first-place votes, the list of
            ballots with them ranked first, and the threshold value. Returns the list of ballots
            after transfer.
        quota (str, optional): Formula to calculate quota. Accepts "droop" or "hare".
            Defaults to "droop".
        simultaneous (bool, optional): True if all candidates who cross threshold in a round are
            elected simultaneously, False if only the candidate with highest first-place votes
            who crosses the threshold is elected in a round. Defaults to True.
        tiebreak (str, optional): Method to be used if a tiebreak is needed. Accepts
            'borda' and 'random'. Defaults to None, in which case a ValueError is raised if
            a tiebreak is needed.

    """  # noqa

    def __init__(
        self,
        profile: PreferenceProfile,
        m: int = 1,
        transfer: Callable[
            [str, Union[Fraction, float], Union[tuple[Ballot], list[Ballot]], int],
            tuple[Ballot, ...],
        ] = fractional_transfer,
        quota: str = "droop",
        simultaneous: bool = True,
        tiebreak: Optional[str] = None,
    ):
        self._stv_validate_profile(profile)

        if m <= 0 or m > len(profile.candidates):
            raise ValueError(
                "m must be non-negative and less than or equal to the number of candidates."
            )

        self.m = m
        self.transfer = transfer
        self.quota = quota
        # Set to 0 initially so that first call to `get_threshold` returns the
        # proper threshold.
        self.threshold = 0
        self.threshold = self.get_threshold(profile.total_ballot_wt)
        self.simultaneous = simultaneous
        self.tiebreak = tiebreak
        super().__init__(profile, score_function=first_place_votes, sort_high_low=True)

    def _stv_validate_profile(self, profile: PreferenceProfile):
        """
        Validate that each ballot has a ranking, and that there are no ties in ballots.
        """

        for ballot in profile.ballots:
            if not ballot.ranking:
                raise TypeError("Ballots must have rankings.")
            if len(ballot.ranking) == 0:
                raise TypeError("All ballots must have rankings.")
            elif any(len(s) > 1 for s in ballot.ranking):
                raise TypeError(f"Ballot {ballot} contains a tied ranking.")

    def get_threshold(self, total_ballot_wt: Fraction) -> int:
        """
        Calculates threshold required for election.

        Args:
            total_ballot_wt (Fraction): Total weight of ballots to compute threshold.
        Returns:
            int: Value of the threshold.
        """
        if self.threshold == 0:
            if self.quota == "droop":
                return int(total_ballot_wt / (self.m + 1) + 1)  # takes floor
            elif self.quota == "hare":
                return int(total_ballot_wt / self.m)  # takes floor
            else:
                raise ValueError("Misspelled or unknown quota type.")
        else:
            return self.threshold

    def _is_finished(self):
        elected_cands = [c for s in self.get_elected() for c in s]

        if len(elected_cands) == self.m:
            return True
        return False

    def _simultaneous_elect_step(
        self, profile: PreferenceProfile, prev_state: ElectionState
    ) -> tuple[tuple[frozenset[str], ...], PreferenceProfile]:
        """
        Run one step of an election from the given profile and previous state.
        Used for simultaneous STV election if candidates cross threshold.

        Args:
            profile (PreferenceProfile): Profile of ballots.
            prev_state (ElectionState): The previous ElectionState.

        Returns:
            tuple[tuple[frozenset[str],...], PreferenceProfile]:
                A tuple whose first entry is the elected candidates, ranked by first-place votes,
                and whose second entry is the profile of ballots after transfers.
        """
        ranking_by_fpv = prev_state.remaining

        elected = []
        for s in ranking_by_fpv:
            c = list(s)[0]  # all cands in set have same score
            if prev_state.scores[c] >= self.threshold:
                elected.append(s)

            # since ranking is ordered by fpv, once below threshold we are done
            else:
                break

        ballots_by_fpv = ballots_by_first_cand(profile)
        new_ballots = [Ballot()] * profile.num_ballots
        ballot_index = 0

        for s in elected:
            for candidate in s:
                transfer_ballots = self.transfer(
                    candidate,
                    prev_state.scores[candidate],
                    ballots_by_fpv[candidate],
                    self.threshold,
                )
                new_ballots[
                    ballot_index : (ballot_index + len(transfer_ballots))
                ] = transfer_ballots
                ballot_index += len(transfer_ballots)

        for candidate in set([c for s in ranking_by_fpv for c in s]).difference(
            [c for s in elected for c in s]
        ):
            transfer_ballots = tuple(ballots_by_fpv[candidate])
            new_ballots[
                ballot_index : (ballot_index + len(transfer_ballots))
            ] = transfer_ballots
            ballot_index += len(transfer_ballots)

        cleaned_ballots = remove_cand(
            [c for s in elected for c in s],
            tuple([b for b in new_ballots if b.ranking]),
        )

        remaining_cands = set(profile.candidates).difference(
            [c for s in elected for c in s]
        )
        new_profile = PreferenceProfile(
            ballots=cleaned_ballots, candidates=tuple(remaining_cands)
        )
        return (tuple(elected), new_profile)

    def _single_elect_step(
        self, profile: PreferenceProfile, prev_state: ElectionState
    ) -> tuple[
        tuple[frozenset[str], ...],
        dict[frozenset[str], tuple[frozenset[str], ...]],
        PreferenceProfile,
    ]:
        """
        Run one step of an election from the given profile and previous state.
        Used for one-by-one STV election if candidates cross threshold.

        Args:
            profile (PreferenceProfile): Profile of ballots.
            prev_state (ElectionState): The previous ElectionState.

        Returns:
            tuple[tuple[frozenset[str],...], dict[frozenset[str], tuple[frozenset[str],...]],
            PreferenceProfile]:
                A tuple whose first entry is the elected candidate, second is the tiebreak dict,
                and whose third entry is the profile of ballots after transfers.
        """
        ranking_by_fpv = prev_state.remaining

        elected, remaining, tiebreak = elect_cands_from_set_ranking(
            ranking_by_fpv, m=1, profile=profile, tiebreak=self.tiebreak
        )
        if tiebreak:
            tiebreaks = {tiebreak[0]: tiebreak[1]}
        else:
            tiebreaks = {}

        ballots_by_fpv = ballots_by_first_cand(profile)
        new_ballots = [Ballot()] * profile.num_ballots
        ballot_index = 0

        elected_c = list(elected[0])[0]

        transfer_ballots = self.transfer(
            elected_c,
            prev_state.scores[elected_c],
            ballots_by_fpv[elected_c],
            self.threshold,
        )
        new_ballots[
            ballot_index : (ballot_index + len(transfer_ballots))
        ] = transfer_ballots
        ballot_index += len(transfer_ballots)

        for s in remaining:
            for candidate in s:
                transfer_ballots = tuple(ballots_by_fpv[candidate])
                new_ballots[
                    ballot_index : (ballot_index + len(transfer_ballots))
                ] = transfer_ballots
                ballot_index += len(transfer_ballots)

        cleaned_ballots = remove_cand(
            elected_c, tuple([b for b in new_ballots if b.ranking])
        )

        remaining_cands = set(profile.candidates).difference(
            [c for s in elected for c in s]
        )
        new_profile = PreferenceProfile(
            ballots=cleaned_ballots, candidates=tuple(remaining_cands)
        )
        return elected, tiebreaks, new_profile

    def _run_step(
        self, profile: PreferenceProfile, prev_state: ElectionState, store_states=False
    ) -> PreferenceProfile:
        """
        Run one step of an election from the given profile and previous state.
        STV sets a threshold for first-place votes. If a candidate passes it, they are elected.
        We remove them from all ballots and transfer any surplus ballots to other candidates.
        If no one passes, we eliminate the lowest ranked candidate and reallocate their ballots.

        Can be run 1-by-1 or simultaneous, which determines what happens if multiple people cross
        threshold.

        Args:
            profile (PreferenceProfile): Profile of ballots.
            prev_state (ElectionState): The previous ElectionState.
            store_states (bool, optional): True if `self.election_states` should be updated with the
                ElectionState generated by this round. This should only be True when used by
                `self._run_election()`. Defaults to False.

        Returns:
            PreferenceProfile: The profile of ballots after the round is completed.
        """
        tiebreaks: dict[frozenset[str], tuple[frozenset[str], ...]] = {}

        above_thresh_cands = [
            c for c, score in prev_state.scores.items() if score >= self.threshold
        ]

        if len(above_thresh_cands) > 0:
            if self.simultaneous:
                elected, new_profile = self._simultaneous_elect_step(
                    profile, prev_state
                )

            else:
                elected, tiebreaks, new_profile = self._single_elect_step(
                    profile, prev_state
                )

            # no on eliminated in elect round
            eliminated: tuple[frozenset[str], ...] = (frozenset(),)

        # catches the possibility that we exhaust all ballots
        # without candidates reaching threshold
        elif len(profile.candidates) == self.m - len(
            [c for s in self.get_elected() for c in s]
        ):
            elected = prev_state.remaining
            eliminated = (frozenset(),)
            new_profile = PreferenceProfile()

        else:
            lowest_fpv_cands = prev_state.remaining[-1]

            if len(lowest_fpv_cands) > 1:
                tiebroken_ranking = tiebreak_set(
                    lowest_fpv_cands, self.get_profile(0), tiebreak="first_place"
                )
                tiebreaks = {lowest_fpv_cands: tiebroken_ranking}

                eliminated_cand = list(tiebroken_ranking[-1])[0]

            else:
                eliminated_cand = list(lowest_fpv_cands)[0]

            new_profile = remove_cand(eliminated_cand, profile)
            elected = (frozenset(),)
            eliminated = (frozenset([eliminated_cand]),)

        if store_states:
            if self.score_function:
                scores = self.score_function(new_profile)

            remaining = score_dict_to_ranking(scores)

            new_state = ElectionState(
                round_number=prev_state.round_number + 1,
                remaining=remaining,
                elected=elected,
                eliminated=eliminated,
                scores=scores,
                tiebreaks=tiebreaks,
            )

            self.election_states.append(new_state)

        return new_profile


class IRV(STV):
    """
    IRV (Instant-runoff voting) elections.  Elect 1 seat. All ballots must have no ties.
    Equivalent to STV for m = 1.

    Args:
        profile (PreferenceProfile):   PreferenceProfile to run election on.
        quota (str, optional): Formula to calculate quota. Accepts "droop" or "hare".
            Defaults to "droop".
        tiebreak (str, optional): Method to be used if a tiebreak is needed. Accepts
            'borda' and 'random'. Defaults to None, in which case a ValueError is raised if
            a tiebreak is needed.

    """

    def __init__(
        self,
        profile: PreferenceProfile,
        quota: str = "droop",
        tiebreak: Optional[str] = None,
    ):
        super().__init__(profile, m=1, quota=quota, tiebreak=tiebreak)


class SequentialRCV(STV):
    """
    An STV election in which votes are not transferred after a candidate has reached threshold, or
    been elected. This system is actually used in parts of Utah.

    Args:
        profile (PreferenceProfile):   PreferenceProfile to run election on.
        m (int, optional): Number of seats to be elected. Defaults to 1.
        quota (str, optional): Formula to calculate quota. Accepts "droop" or "hare".
            Defaults to "droop".
        simultaneous (bool, optional): True if all candidates who cross threshold in a round are
            elected simultaneously, False if only the candidate with highest first-place votes
            who crosses the threshold is elected in a round. Defaults to True.
        tiebreak (str, optional): Method to be used if a tiebreak is needed. Accepts
            'borda' and 'random'. Defaults to None, in which case a ValueError is raised if
            a tiebreak is needed.

    """

    def __init__(
        self,
        profile: PreferenceProfile,
        m: int = 1,
        quota: str = "droop",
        simultaneous: bool = True,
        tiebreak: Optional[str] = None,
    ):
        super().__init__(
            profile,
            m=m,
            transfer=(
                lambda winner, fpv, ballots, threshold: remove_cand(
                    winner, tuple(ballots)
                )
            ),
            quota=quota,
            simultaneous=simultaneous,
            tiebreak=tiebreak,
        )
