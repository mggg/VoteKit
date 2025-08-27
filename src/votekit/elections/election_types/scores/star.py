from typing import List, Optional
from typing import cast
import numpy as np
from votekit.elections.election_types.scores.rating import GeneralRating
from ....pref_profile import PreferenceProfile
from ...election_state import ElectionState
from ....cleaning import remove_cand_scored

class Star(GeneralRating):
    """
    STAR (Score Then Automatic Runoff) voting method. Voters score each candidate with
    any numerical value up to L, where L is a user-specified maximum score,
    using `votekit.Ballot` objects with score dictionaries. Scores may be positive,
    and missing scores are treated as 0. The two candidates with the highest total scores
    advance to an automatic runoff, where the candidate preferred by more voters (based on
    higher scores) wins. Ties in the runoff are resolved using the specified tiebreaker.

    Args:
        profile (PreferenceProfile): Profile from VoteKit, containing a list of Ballot objects
        L (float, optional): Maximum score per candidate. Defaults to 5.
        tiebreak (str, optional): Tiebreak method to use. Options are None or 'most_top_ratings'.
            If 'most_top_ratings', the finalist with more ballots assigning the maximum score wins.
            If None, a tie results in no winner (None). Defaults to None.

    Raises:
        ValueError: If L is not positive or there are fewer than two candidates.
        TypeError: If ballots lack score dictionaries or have non-positive weights
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
        if tiebreak != 'most_top_ratings':
            raise ValueError("tiebreak must be 'most_top_ratings'. No other methods supported at this time")
        
        # Store parameters
        self.L = L
        self.tiebreak = tiebreak
        self.m = m
        self.k = k
        
        # Gather candidates considered in election
        self._cands = list(profile.candidates_cast)
        self._cand_index = {cand: i for i, cand in enumerate(self._cands)}

        # Store profile information
        self.df = profile.df
        self._scores_mat = self.df.loc[:, self._cands].fillna(0).to_numpy()
        self._weights = self.df["Weight"].to_numpy(copy=False)
        
        super().__init__(
            profile,
            m=1,
            tiebreak=tiebreak,
            L=L,
        )
            
    def _tiebreak_most_top_ratings(self, finalist_indices: List[int]) -> str:
        """
        Resolves a tie by selecting the finalist with more ballots assigning the maximum score.

        Args:
            finalists (List[str]): The two candidates in the runoff.

        Returns:
            Optional[str]: The winning candidate, or None if the tie persists.
        """
        finalist_score_matrix = self._scores_mat[:, finalist_indices]
        
        # Compare candidates by the number of votes at each score until a winner is found
        for score_rank in range(int(self.L) - 1, -1, -1):
            mask = (finalist_score_matrix == score_rank)
            counts = np.dot(self._weights, mask)
            if counts[0] != counts[1]:
                winner_idx = finalist_indices[np.argmax(counts)]
                return self._cands[winner_idx]
        return self._cands[np.random.choice(finalist_indices)] # random tiebreak if all else fails
    
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
    
    def _run_step(self, profile: PreferenceProfile, prev_state: ElectionState, store_states: bool = False):
        """
        Runs the STAR voting method.
        
        Args:
            profile (PreferenceProfile): Profile from votekit.
            prev_state (dict): Previous state of the election.
            store_states (bool, optional): Whether to store states. Defaults to False.

        Returns:
            PreferenceProfile: Updated profile.
        """
        # Get score totals
        score_totals = self._scores_mat.T @ self._weights

        # Find the top two candidates
        idx1, idx2 = np.argsort(score_totals)[-2:]
        int_idx1, int_idx2 = int(idx1), int(idx2)
        finalist_1, finalist_2 = self._cands[int_idx1], self._cands[int_idx2]

        # Runoff Election
        diff = self._scores_mat[:, int_idx1] - self._scores_mat[:, int_idx2]
        wins1 = self._weights[diff > 0].sum()
        wins2 = self._weights[diff < 0].sum()

        tiebreak = False
        if wins1 > wins2:
            winner = finalist_1
        elif wins2 > wins1:
            winner = finalist_2
        else:
            tiebreak = True
            winner = self._tiebreak_most_top_ratings([int_idx1, int_idx2])          

        # Build the new profile
        new_profile = remove_cand_scored([winner], profile)

        if store_states:

            eliminated = tuple(frozenset({candidate}) for candidate in profile.candidates_cast if candidate != winner)
            elected = (frozenset({winner}),)
            scores = dict(zip(self._cands, score_totals))
            tiebreaks = ({frozenset({finalist_1, finalist_2}): cast(tuple[frozenset[str], ...], (frozenset({winner}),))} if tiebreak else {})

            self.election_states.append(ElectionState(
                round_number=prev_state.round_number + 1,
                remaining=tuple(),
                eliminated=eliminated,
                elected=elected,
                scores=scores,
                tiebreaks=tiebreaks,
            ))

        return new_profile
