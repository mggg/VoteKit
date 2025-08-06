from typing import List, Optional
import numpy as np
import pandas as pd
from votekit.elections.election_types.scores.rating import GeneralRating
from ....pref_profile import PreferenceProfile
from ...election_state import ElectionState
from ....cleaning import remove_and_condense_scored
from typing import Optional

class Star(GeneralRating):
    """
    STAR (Score Then Automatic Runoff) voting method. Voters score each candidate with
    any numerical value up to L, where L is a user-specified maximum score,
    using votekit.Ballot objects with score dictionaries. Scores may be positive,
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
            winner (str or None): The winning candidate, or None if a tie occurs and tiebreak is None.
            finalists (tuple): The two candidates in the runoff.
            scores (dict): Total scores per candidate.
            runoff (dict): Runoff vote counts for finalists and "No Preference".
    """

    def __init__(
        self,
        profile: PreferenceProfile,
        L: float = 5,
        *,
        tiebreak: str = "most_top_ratings",
        m: int = 1,
        k: Optional[float] = None,
    ):
        
        """Initialize a STAR election."""
        if L <= 0:
            raise ValueError("L must be positive.")
        self.L = L
        if tiebreak != 'most_top_ratings':
            raise ValueError("tiebreak must be 'most_top_ratings'. No other methods supported at this time")
        
        self.tiebreak = tiebreak
        self.m = m
        self.k = k
        self._validate_profile(profile)

        self._cands = list(profile.candidates_cast)
        self.df = profile.df
        self._scores_mat = self.df[self._cands].fillna(0).to_numpy()
        self._weights = self.df["Weight"].to_numpy()
        

        if any(self._weights < 0):
            raise ValueError("All ballot weights must be positive.")
        
        super().__init__(
            profile,
            m=1,
            tiebreak=tiebreak,
            L=L,
        )        

    def _tiebreak_most_top_ratings(self, finalist_indices: List[int]) -> Optional[str]:
        """
        Resolves a tie by selecting the finalist with more ballots assigning the maximum score.

        Args:
            finalists (List[str]): The two candidates in the runoff.

        Returns:
            Optional[str]: The winning candidate, or None if the tie persists.
        """
        scores = self._scores_mat[:, finalist_indices]
        weights = self._weights
        
        print(finalist_indices)

        # Check and return which of the finalists has more top ratings, secondary ratings, triary ratings, ..., and then return randomly if all are equal. 
        for score_rank in range(int(self.L) - 1, -1, -1):
            mask = (scores == score_rank)  # shape: (num_ballots, 2)
            counts = np.dot(weights, mask) # shape: (2,)
            if counts[0] != counts[1]:
                winner_idx = finalist_indices[np.argmax(counts)]
                return self._cands[winner_idx]
        return self._cands[np.random.choice(finalist_indices)]

    def _is_finished(self) -> bool:
        """
        Determines if the election process is complete.

        Returns:
            bool: True if the number of winners equals the number of seats to be filled (m),
            indicating that the election has finished. False otherwise.
        """
        elected_cands = [c for s in self.get_elected() for c in s]

        if len(elected_cands) == self.m:
            return True
        return False

    def _run_step(self, profile: PreferenceProfile, prev_state: dict, store_states: bool = False):
        scores_mat = self._scores_mat
        weights = self._weights
        cands = self._cands

        # Total scores per candidate
        totals = np.dot(weights, scores_mat)
        totals = pd.Series(totals, index=cands)

        # Find top 2 finalists
        top2 = totals.sort_values(ascending=False).index[:2]
        idx1, idx2 = cands.index(top2[0]), cands.index(top2[1])
        finalist_1, finalist_2 = top2[0], top2[1]

        # Runoff counts
        finalist_scores1 = scores_mat[:, idx1] 
        finalist_scores2 = scores_mat[:, idx2]

        finalist_wins1 = weights[finalist_scores1 > finalist_scores2].sum()
        finalist_wins2 = weights[finalist_scores2 > finalist_scores1].sum()

        # Determine winner and handle tie
        if finalist_wins1 > finalist_wins2:
            winner = finalist_1
        elif finalist_wins2 > finalist_wins1:
            winner = finalist_2
        else:
            winner = self._tiebreak_most_top_ratings([idx1, idx2])

        # Store results
        elected = [frozenset({winner})] if winner else []
        remaining = [c for c in profile.candidates_cast if c != winner]

        new_profile = remove_and_condense_scored([winner], profile)

        if store_states:
            new_state = ElectionState(
                round_number=1,
                remaining=tuple(),
                eliminated=[frozenset({c}) for c in remaining],
                elected=elected,
                scores=totals.to_dict(),
                tiebreaks={} if winner else {f"{finalist_1},{finalist_2}": None},
            )
            self.election_states.append(new_state)

        return new_profile