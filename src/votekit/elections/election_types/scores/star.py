from typing import Dict, List, Tuple, Optional
import numpy as np
import random
from ....models import Election
from ....pref_profile import PreferenceProfile
from ...election_state import ElectionState
from ....utils import (
    score_profile_from_ballot_scores,
    elect_cands_from_set_ranking,
)
from ....cleaning import remove_and_condense
from typing import Optional

class Star(Election):
    """
    STAR (Score Then Automatic Runoff) voting method. Voters score each candidate with
    any numerical value up to :math:`L`, where :math:`L` is a user-specified maximum score,
    using votekit.Ballot objects with score dictionaries. Scores may be positive or negative,
    and missing scores are treated as 0. The two candidates with the highest total scores
    advance to an automatic runoff, where the candidate preferred by more voters (based on
    higher scores) wins. Ties in the runoff are resolved using the specified tiebreaker.

    Args:
        profile (PreferenceProfile): Profile from votekit, containing a list of Ballot objects
            with score dictionaries mapping candidates to scores and optional weights.
        L (float, optional): Maximum score per candidate. Defaults to 5.
        tiebreak (str, optional): Tiebreak method to use. Options are None or 'most_top_ratings'.
            If 'most_top_ratings', the finalist with more ballots assigning the maximum score wins.
            If None, a tie results in no winner (None). Defaults to None.

    Raises:
        ValueError: If L is not positive or there are fewer than two candidates.
        TypeError: If ballots lack score dictionaries or have non-positive weights.

    Returns:
        dict: A dictionary containing:
            - winner (str or None): The winning candidate, or None if a tie occurs and tiebreak is None.
            - finalists (tuple): The two candidates in the runoff.
            - scores (dict): Total scores per candidate.
            - runoff (dict): Runoff vote counts for finalists and "No Preference".
    """

    def __init__(
        self,
        profile: PreferenceProfile,
        L: float = 5,
        tiebreak: Optional[str] = None,
    ):
        """Initialize a STAR election."""
        if L <= 0:
            raise ValueError("L must be positive.")
        self.L = L
        if tiebreak not in [None, 'most_top_ratings']:
            raise ValueError("tiebreak must be None or 'most_top_ratings'.")
        
        self.tiebreak = tiebreak
        self._validate_profile(profile)

        self._cands = list(profile.candidates_cast)
        df = profile.df
        self._scores_mat = df[self._cands].fillna(0).to_numpy()
        self._weights    = df["Weight"].to_numpy()  
        
        super().__init__(
            profile, score_function=score_profile_from_ballot_scores, sort_high_low=True
        )

    def _validate_profile(self, profile: PreferenceProfile):
        """
        Ensures that every ballot has a score dictionary and weights are valid.

        Args:
            profile (PreferenceProfile): Profile from votekit to validate, containing Ballot objects.

        Raises:
            ValueError: Fewer than two candidates provided (STAR requires at least two).
            TypeError: Ballots lack score dictionaries or have non-positive weights.
        """
        # Need at least two candidates
        cands = list(profile.candidates_cast)
        if len(cands) <= 1:
            raise ValueError("STAR requires at least two candidates.")
        
        # Every Ballot must actually carry a .scores dict
        if any(b.scores is None for b in profile.ballots):
            raise TypeError("All ballots must have score dictionary.")
        
        df = profile.df

        # No negative weights allowed
        if (df["Weight"] < 0).any():
            raise TypeError("Ballot must have positive weight.")

        # Catch any ballot whose explicit scores exceed L
        for b in profile.ballots:
            if any(score > self.L for score in b.scores.values()):
                raise TypeError(f"Ballot violates score limit {self.L} per candidate.")

    def _tiebreak_most_top_ratings(self, finalists: List[str], ballots: List) -> Optional[str]:
        """
        Resolves a tie by selecting the finalist with more ballots assigning the maximum score.

        Args:
            finalists (List[str]): The two candidates in the runoff.
            ballots (List): List of votekit.Ballot objects from the PreferenceProfile.

        Returns:
            Optional[str]: The winning candidate, or None if the tie persists.
        """
        top_score = max(
            (b.scores.get(f, 0) for b in ballots for f in finalists),
            default=0
        )

        top_ratings = {f: 0 for f in finalists}

        for ballot in ballots:
            weight = getattr(ballot, 'weight', 1)
            for f in finalists:
                if ballot.scores.get(f, 0) == top_score:
                    top_ratings[f] += weight

        
        if top_ratings[finalists[0]] != top_ratings[finalists[1]]:
            return max(top_ratings, key=top_ratings.get)
        
        return None

    def _is_finished(self) -> bool:
        """
        Checks if the election is finished (single-round election).

        Returns:
            bool: True if the election has completed (two states exist), False otherwise.
        """
        return len(self.election_states) == 2

    def _run_step(self, profile, prev_state, store_states=False):
        # Score totals
        totals = self._weights @ self._scores_mat

        # Pick finalists
        fianlists = np.argsort(-totals)[:2]
        finalist1, finalist2 = fianlists

        # get per‑ballot scores for fianlists
        score_finalist1 = self._scores_mat[:, finalist1]
        score_finalist2 = self._scores_mat[:, finalist2]
        weights  = self._weights

        # runoff counts
        runoff_counts1 = (weights * (score_finalist1 > score_finalist2)).sum()
        runoff_counts2 = (weights * (score_finalist2 > score_finalist1)).sum()
        runoff_counts_nopref = (weights * (score_finalist1 == score_finalist2)).sum()

        runoff = {
            self._cands[finalist1]: runoff_counts1,
            self._cands[finalist2]: runoff_counts2,
            "No Preference": runoff_counts_nopref
        }

        # decide winner
        if runoff_counts1 > runoff_counts2:
            winner = self._cands[finalist1]
        elif runoff_counts2 > runoff_counts1:
            winner = self._cands[finalist2]
        else:
            if self.tiebreak == 'most_top_ratings':
                topv = max(score_finalist1.max(), score_finalist2.max())
                t1   = (weights * (score_finalist1 == topv)).sum()
                t2   = (weights * (score_finalist2 == topv)).sum()
                winner = self._cands[finalist1] if t1 > t2 else self._cands[finalist2] if t2 > t1 else None
            else:
                winner = None
            if self.tiebreak == 'most_top_ratings':
                winner = self._tiebreak_most_top_ratings([finalist1, finalist2], profile.ballots)

        # store for later
        self._runoff_counts = runoff
        elected = [frozenset({winner})] if winner else []
        remaining = [c for c in profile.candidates_cast if c != winner]

        # Update profile by removing elected candidate
        elected_cands = [winner] if winner else []
        new_profile = remove_and_condense(elected_cands, profile)

        if store_states:
            scores_dict = dict(zip(self._cands, totals.tolist()))
            new_state = ElectionState(
                round_number=1,
                remaining=tuple(),
                eliminated=[frozenset({c}) for c in remaining],
                elected=elected,
                scores=scores_dict,
                tiebreaks={} if winner else {f"{finalist1},{finalist2}": None},
                #metadata={"runoff_counts": runoff_counts}
            )
            self.election_states.append(new_state)

        return new_profile

    def run_election(self) -> Dict:
        """
        Run the STAR election and return the results.

        Returns:
            dict: A dictionary containing:
                - winner (str or None): The winning candidate, or None if a tie occurs and tiebreak is None.
                - finalists (tuple): The two candidates in the runoff.
                - scores (dict): Total scores per candidate.
                - runoff (dict): Runoff vote counts for finalists and "No Preference".
        """
        #super()._run_election()
        final_state = self.election_states[-1] if self.election_states else None
        if not final_state:
            return {
                "winner": None,
                "finalists": (),
                "scores": {},
                "runoff": {}
            }
        
        winner = next(iter(final_state.elected[0])) if final_state.elected else None
        sorted_cands = sorted(final_state.scores, key=final_state.scores.get, reverse=True)
        finalists = tuple(sorted_cands[:2])

        #finalists = tuple(final_state.scores.keys()[:2]) if len(final_state.scores) >= 2 else tuple(final_state.scores.keys())
        return {
            "winner": winner,
            "finalists": finalists,
            "scores": final_state.scores,
            "runoff": getattr(self, "_runoff_counts", {}),
        }