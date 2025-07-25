from .abstract_ranking import RankingElection
from ....pref_profile import PreferenceProfile
from ...election_state import ElectionState
from ....cleaning import remove_and_condense_ranked_profile
from ....utils import (
    first_place_votes,
    score_dict_to_ranking,
)
import random
from typing import Literal
from functools import partial


class BoostedRandomDictator(RankingElection):
    """
    Modified random dictator where with probability (1 - 1/(n_candidates - 1))
    choose a winner randomly from the distribution of first place votes.
    With probability 1/(n_candidates - 1), choose a winner via a proportional to
    squares rule.

    For multi-winner elections
    repeat this process for every winner, removing that candidate from every
    voter's ballot once they have been elected.

    Args:
      profile (PreferenceProfile): PreferenceProfile to run election on.
      m (int): Number of seats to elect.
      fpv_tie_convention (Literal["high", "average", "low"], optional): How to award points
            for tied first place votes. Defaults to "average", where if n candidates are tied for
            first, each receives 1/n points. "high" would award them each one point, and "low" 0.

    """

    def __init__(
        self,
        profile: PreferenceProfile,
        m: int,
        fpv_tie_convention: Literal["high", "average", "low"] = "average",
    ):
        if m <= 0:
            raise ValueError("m must be positive.")
        elif len(profile.candidates_cast) < m:
            raise ValueError("Not enough candidates received votes to be elected.")
        self.m = m

        super().__init__(
            profile,
            score_function=partial(
                first_place_votes, tie_convention=fpv_tie_convention
            ),
        )

    def _is_finished(self) -> bool:
        cands_elected = [len(s) for s in self.get_elected()]
        return sum(cands_elected) >= self.m

    def _run_step(
        self, profile: PreferenceProfile, prev_state: ElectionState, store_states=False
    ) -> PreferenceProfile:
        """
        Run one step of an election from the given profile and previous state.
        If m candidates have not yet been elected:
        finds a single winning candidate to add to the list of elected
        candidates by sampling from the distribution induced by the combination
        of random dictator and proportional to squares election rules.
        Removes that candidate from all ballots in the preference profile.

        Args:
            profile (PreferenceProfile): Profile of ballots.
            prev_state (ElectionState): The previous ElectionState.
            store_states (bool, optional): True if `self.election_states` should be updated with the
                ElectionState generated by this round. This should only be True when used by
                `self._run_election()`. Defaults to False.

        Returns:
            PreferenceProfile: The profile of ballots after the round is completed.
        """
        remaining_cands = profile.candidates
        u = random.uniform(0, 1)

        if len(remaining_cands) == 1:
            winning_candidate = remaining_cands[0]

        elif u <= 1 / (len(remaining_cands) - 1):
            fpv = prev_state.scores
            candidates = list(fpv.keys())
            weights: list[float] = list(fpv.values())
            sq_weights = [float(x) ** 2 for x in weights]
            sq_wt_total = sum(sq_weights)
            sq_weights = [x / sq_wt_total for x in sq_weights]
            winning_candidate = random.choices(candidates, weights=sq_weights, k=1)[0]

        else:
            fpv = prev_state.scores
            candidates = list(fpv.keys())
            weights = list(fpv.values())
            winning_candidate = random.choices(candidates, weights=weights, k=1)[0]

        new_profile = remove_and_condense_ranked_profile(
            winning_candidate,
            profile,
        )

        if store_states:
            elected = (frozenset({winning_candidate}),)
            if self.score_function:
                scores = self.score_function(new_profile)
            remaining = score_dict_to_ranking(scores)

            new_state = ElectionState(
                round_number=prev_state.round_number + 1,
                elected=elected,
                remaining=remaining,
                scores=scores,
                tiebreaks={},
            )
            self.election_states.append(new_state)

        return new_profile
