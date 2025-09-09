from .abstract_ranking import RankingElection
from votekit.pref_profile import RankProfile
from votekit.elections.election_state import ElectionState
from votekit.cleaning import remove_and_condense_ranked_profile
from votekit.utils import first_place_votes
from votekit.elections.election_types.ranking import Plurality, STV
from votekit.elections.transfers import fractional_transfer
from votekit.ballot import RankBallot
from typing import Optional, Callable, Union, Literal
from functools import partial


class Alaska(RankingElection):
    """
    Alaska election. Election method that first runs a Plurality election to choose a user-specified
    number of final-round candidates, then runs STV to choose :math:`m` winners.

    Args:
        profile (PreferenceProfile): Profile to conduct election on.
        m_1 (int, optional): Number of final round candidates, i.e. number of winners of Plurality
            round. Defaults to 2.
        m_2 (int, optional): Number of seats to elect in STV round, i.e. number of overall winners.
            Defaults to 1.
        transfer (Callable[[str, float], Union[tuple[Ballot], list[Ballot]], int], tuple[Ballot,...]], optional):
            Transfer method. Defaults to fractional transfer.
            Function signature is elected candidate, their number of first-place votes, the list of
            ballots with them ranked first, and the threshold value. Returns the list of ballots
            after transfer.
        quota (str, optional): Formula to calculate quota. Accepts "droop" or "hare".
            Defaults to "droop".
        simultaneous (bool, optional): True if all candidates who cross threshold in a round are
            elected simultaneously, False if only the candidate with highest first-place votes
            who crosses the threshold is elected in a round. Defaults to True.
        tiebreak (str, optional): Tiebreak method to use. Options are None, 'random', and 'borda'.
            Defaults to None, in which case a tie raises a ValueError.
        fpv_tie_convention (Literal["high", "average", "low"], optional): How to award points
            for tied first place votes. Defaults to "average", where if n candidates are tied for
            first, each receives 1/n points. "high" would award them each one point, and "low" 0.
            Only used by ``score_function`` parameter.

    """

    def __init__(
        self,
        profile: PreferenceProfile,
        m_1: int = 2,
        m_2: int = 1,
        transfer: Callable[
            [str, float, Union[tuple[Ballot], list[Ballot]], int],
            tuple[Ballot, ...],
        ] = fractional_transfer,
        quota: str = "droop",
        simultaneous: bool = True,
        tiebreak: Optional[str] = None,
        fpv_tie_convention: Literal["high", "low", "average"] = "average",
    ):
        if m_1 <= 0:
            raise ValueError("m_1 must be positive.")
        elif m_2 <= 0:
            raise ValueError("m_2 must be positive.")
        elif m_1 < m_2:
            raise ValueError("m_1 must be greater than or equal to m_2.")
        self.m_1 = m_1
        self.m_2 = m_2
        self.transfer = transfer
        self.quota = quota
        self.simultaneous = simultaneous
        self.tiebreak = tiebreak
        super().__init__(
            profile,
            score_function=partial(
                first_place_votes, tie_convention=fpv_tie_convention
            ),
            sort_high_low=True,
        )

    def get_profile(self, round_number: int = -1) -> PreferenceProfile:
        """
        Fetch the PreferenceProfile of the given round number.

        Args:
            round_number (int, optional): The round number. Supports negative indexing. Defaults to
                -1, which accesses the final profile.

        Returns:
            PreferenceProfile

        """
        if (
            round_number < -len(self.election_states)
            or round_number > len(self.election_states) - 1
        ):
            raise IndexError("round_number out of range.")

        round_number = round_number % len(self.election_states)

        profile = self._profile

        if round_number in [0, 1]:
            for i in range(round_number):
                profile = self._run_step(profile, self.election_states[i])

        else:
            stv = STV(
                self.get_profile(1),  # plurality profile
                self.m_2,
                self.transfer,
                self.quota,
                self.simultaneous,
                self.tiebreak,
            )

            # e.g., round 2 of Alaska is equivalent to round 1 of stv
            profile = stv.get_profile(round_number - 1)

        return profile

    def _is_finished(self):
        elected_cands = [c for s in self.get_elected() for c in s]

        if len(elected_cands) == self.m_2:
            return True
        return False

    def _run_step(
        self, profile: PreferenceProfile, prev_state: ElectionState, store_states=False
    ) -> PreferenceProfile:
        """
        Run one step of an election from the given profile and previous state.

        Args:
            profile (PreferenceProfile): Profile of ballots.
            prev_state (ElectionState): The previous ElectionState.
            store_states (bool, optional): True if `self.election_states` should be updated with the
                ElectionState generated by this round. This should only be True when used by
                `self._run_election()`. Defaults to False.

        Returns:
            PreferenceProfile: The profile of ballots after the round is completed.
        """
        if prev_state.round_number == 0:
            plurality = Plurality(profile, self.m_1, self.tiebreak)
            remaining = plurality.get_elected()
            eliminated = plurality.get_remaining()
            tiebreaks = plurality.election_states[-1].tiebreaks
            new_profile: PreferenceProfile = remove_and_condense_ranked_profile(
                [c for s in eliminated for c in s],
                profile,
            )

            if self.score_function is None:
                raise ValueError("score_function must be defined for Alaska election.")
            scores = self.score_function(new_profile)

            if store_states:
                new_state = ElectionState(
                    round_number=prev_state.round_number + 1,
                    remaining=remaining,
                    eliminated=eliminated,
                    scores=scores,
                    tiebreaks=tiebreaks,
                )

                self.election_states.append(new_state)

        else:
            stv = STV(
                profile,
                self.m_2,
                self.transfer,
                self.quota,
                self.simultaneous,
                self.tiebreak,
            )
            new_profile = stv.get_profile()

            if store_states:
                # first state was already stored by Plurality
                for state in stv.election_states[1:]:
                    state.round_number += 1
                self.election_states += stv.election_states[1:]

        return new_profile
