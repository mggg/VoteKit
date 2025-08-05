from typing import Dict, List, Tuple, Optional
from votekit import Ballot
import numpy as np
import pandas as pd
from votekit.elections.election_types.scores.rating import GeneralRating as Election
from ....pref_profile import PreferenceProfile
from ...election_state import ElectionState
from typing import Optional

class Star(Election):
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
        self._weights    = self.df["Weight"].to_numpy()  

        if any(self._weights < 0):
            raise ValueError("All ballot weights must be positive.")
        
        super().__init__(
            profile,
            m=1,
            tiebreak=tiebreak,
            L=L,
        )        

    def remove_and_condense_ratings(self,removed: List[str], profile: PreferenceProfile, remove_empty_ballots: bool = True, remove_zero_weight_ballots: bool = True,
        ) -> PreferenceProfile:
            """
            Faster version of remove_and_condense for OpenListPR elections.

            Args:
                removed (List[str]): List of candidates to be removed from the profile.
                profile (PreferenceProfile): The original preference profile.
                remove_empty_ballots (bool, optional): If True, removes ballots with no votes.
                remove_zero_weight_ballots (bool, optional): If True, removes ballots with zero weight.

            Returns:
                PreferenceProfile: A new profile with the specified candidates removed and ballots condensed.
            """
            if isinstance(removed, str):
                removed = [removed]

            # pull out candidate list, df, and weight vector
            all_cands_list = list(profile.candidates_cast)
            kept_cands_list = [c for c in all_cands_list if c not in removed]
            df = profile.df

            # build data arrays
            weights = df["Weight"].fillna(1).to_numpy()
            candidate_scores = df[kept_cands_list].fillna(0).to_numpy() 

            # filter out zero‑weight and empty ballots
            filter = np.ones_like(weights, dtype=bool)
            if remove_zero_weight_ballots:
                filter &= (weights > 0)
            if remove_empty_ballots:
                filter &= (candidate_scores.sum(axis=1) > 0)
            weight_filtered = weights[filter] 
            cand_scores_filtered = candidate_scores[filter, :]  

            # Build out information
            rows, cols = np.where(cand_scores_filtered > 0)
            ballot_dicts = [{} for _ in range(cand_scores_filtered.shape[0])]
            for i, j in zip(rows, cols):
                ballot_dicts[i][kept_cands_list[j]] = cand_scores_filtered[i, j]

            new_ballots = [
                Ballot(scores=d, weight=w)
                for d, w in zip(ballot_dicts, weight_filtered)
                if d
            ]
                    
            # return new profile
            return PreferenceProfile(
                ballots=new_ballots,
                candidates=kept_cands_list,
            )

    def _tiebreak_most_top_ratings(self, finalists: List[str]) -> Optional[str]:
        """
        Resolves a tie by selecting the finalist with more ballots assigning the maximum score.

        Args:
            finalists (List[str]): The two candidates in the runoff.

        Returns:
            Optional[str]: The winning candidate, or None if the tie persists.
        """
        df = self._profile.df
        scores_df = df[finalists].fillna(0)
        weights = df["Weight"].fillna(1)
        # Check and retrun which of the finalists has more top ratings, secondary ratings, triary ratings, and then return randomly if all are equal. 
        for score_rank in range(int(self.L) - 1, -1, -1):
            counts = {
                cand: weights[scores_df[cand] == score_rank].sum()
                for cand in finalists
            }

            if counts[finalists[0]] != counts[finalists[1]]:
                return max(counts, key=counts.get)

        return np.random.choice(finalists)

    def _run_step(self, profile: PreferenceProfile, prev_state: dict, store_states: bool = False):
        df = self._profile.df
        weights = df["Weight"].fillna(1)
        scores_df = df[self._cands].fillna(0)

        # Total scores per candidate
        totals = (scores_df.values * weights.values[:, None]).sum(axis=0)
        totals = pd.Series(totals, index=scores_df.columns)

        # Find top 2 finalists
        top2 = totals.sort_values(ascending=False).index[:2]
        finalist_1, finalist_2 = top2[0], top2[1]

        # Runoff counts
        runoff_counts1 = df[finalist_1].fillna(0)
        runoff_counts2 = df[finalist_2].fillna(0)

        runoff_counts = {
            finalist_1: weights[runoff_counts1 > runoff_counts2].sum(),
            finalist_2: weights[runoff_counts2 > runoff_counts1].sum(),
            "No Preference": weights[runoff_counts1 == runoff_counts2].sum(),
        }

        # Determine winner
        if runoff_counts[finalist_1] > runoff_counts[finalist_2]:
            winner = finalist_1
        elif runoff_counts[finalist_2] > runoff_counts[finalist_1]:
            winner = finalist_2
        else:
            top_votes = max(runoff_counts1.max(), runoff_counts2.max())
            top_votes1 = weights[runoff_counts1 == top_votes].sum()
            top_votes2 = weights[runoff_counts2 == top_votes].sum()
            if top_votes1 > top_votes2:
                winner = finalist_1
            elif top_votes2 > top_votes1:
                winner = finalist_2
            else:
                winner = self._tiebreak_most_top_ratings([finalist_1, finalist_2])

        # Store results
        self._runoff_counts = runoff_counts
        elected = [frozenset({winner})] if winner else []
        remaining = [c for c in profile.candidates_cast if c != winner]

        new_profile = self.remove_and_condense_ratings([winner], profile)

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