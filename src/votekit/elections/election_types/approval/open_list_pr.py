from typing import Dict, List, Optional, Union, List
import numpy as np
import pandas as pd
import random
from votekit import PreferenceProfile, Ballot
from votekit.elections.election_types.scores.rating import GeneralRating as Election
from ....pref_profile import PreferenceProfile
from ...election_state import ElectionState

class OpenListPR(Election):
    """
    Open-list party-list proportional representation (PR) election method.

    In Open-list, each party gives a list of candidates to elect. Voters can cast votes for individual candidates
    as well as whole parties (as opposed to closed-list PR where voters can only vote for parties). The total number
    of seats allocated to each party is determined by a function that takes in the sum of votes recieved by
    each party and their individual candidates as well as the current party seat share. Once a party's seats are determined, the candidates are elected from the party's
    list in order of their individual vote totals. 
    """
    def __init__(
        self,
        profile: PreferenceProfile,
        party_map: Dict[str, str],
        m: int = 1,
        tiebreak: Optional[str] = "random",
        divisor_method: str = "dhondt",
        L: float = 1,
        k: float = 1,
    ): 
        """
        Initialize an OpenListPR election.

        Args:
        profile (PreferenceProfile): Profile from votekit, containing a list of Ballot objects
            with rankings of candidates and optional weights.
        party_map Dict[str, str]: A dictionary mapping each candidate to a party. 
        m (int, optional): Number of seats to be filled. Defaults to 1.
        tiebreak (str, optional): Tiebreak method to use. Options are None or 'random'.
        divisor_method (str, optional): Method to use for calculating party seat allocation.
            Options are "dhondt", "saint-lague", or "imperiali".
        """

        if m <= 0:
            raise ValueError("m must be positive.")
        if party_map is None:
            raise ValueError("A party map must be provided for OpenListPR.")

        self.m = m
        self.tiebreak = tiebreak
        self.party_map = party_map
        self.divisor_method = divisor_method
        self.L = L
        self.k = k

        self._validate_profile(profile)

        # Precompute once before the real run
        (
            self._candidate_vote_totals,
            self._party_vote_totals,
            self._party_to_sorted_candidates,
        ) = self._precompute_scores(profile)

        # Will be updated only in the real run during each step
        self._party_seat_totals: Dict[str, int] = {p: 0 for p in self._party_vote_totals}
        self._winners: List[str] = []
        self._last_scores: Dict[str, float] = {}

        self._base_party_df = pd.DataFrame({
            "total_votes": pd.Series(self._party_vote_totals),
            "num_candidates": pd.Series({p: len(cands) for p, cands in self._party_to_sorted_candidates.items()}),
            "seats_so_far": 0,
        })

        super().__init__(profile=profile, m=m, tiebreak=tiebreak)
        
    def _validate_profile(self, profile: PreferenceProfile) -> None:
        """
        Validates the profile to ensure that each ballot ranks exactly one candidate
        and all weights are non-negative.

        Args:
            profile (PreferenceProfile): The profile containing ballots to be validated.

        Returns:
            None

        Raises:
            ValueError: If a ballot does not rank exactly one candidate or if any ballot
            has a negative weight.
        """
        df = profile.df

        # Grab the candidate columns 
        cand_cols = list(profile.candidates_cast)
        cand_df = df[cand_cols].fillna(0)

        # No ballot may award more than 1 point to any candidate
        if cand_df.gt(1).values.any():
            raise ValueError("Ballots can only accept candidates (scores of one).")

    def _precompute_scores(self, profile: PreferenceProfile) -> tuple[Dict[str, int], Dict[str, int], Dict[str, List[str]]]:
        """
        This function computes the total number of votes for each candidate, the
        total number of votes for each party, and sorts the list of candidates
        for each party in descending order of votes.

        Args:
            profile (PreferenceProfile): The input preference profile.

        Returns:
            tuple[Dict[str, int], Dict[str, int], Dict[str, List[str]]]:
                A tuple of three dictionaries. The first dictionary maps each candidate
                to its total number of votes. The second dictionary maps each party to
                its total number of votes. The third dictionary maps each party to a
                sorted list of its candidates in descending order of votes.
        """
        df = profile.df.copy()
        candidate_cols = list(profile.candidates_cast)

        ## tally candidates aggregated
        weights = df["Weight"].fillna(1).to_numpy()
        df_cleaned = df[candidate_cols].fillna(0).to_numpy()

        # Calculate weighted votes for each candidate
        weighted_votes = df_cleaned.T.dot(weights)
        candidate_vote_totals = dict(zip(candidate_cols, weighted_votes))
        
        # Party vote totals
        candidate_votes = pd.Series(candidate_vote_totals)
        party_series = pd.Series(self.party_map)
        party_vote_totals = candidate_votes.groupby(party_series).sum().to_dict()

        # Build sorted party lists
        cand_df = pd.DataFrame({
            "candidate": list(profile.candidates_cast),
            "party": [self.party_map[c] for c in profile.candidates_cast],
            "votes": [candidate_vote_totals[c] for c in profile.candidates_cast]
        })

        party_to_sorted_candidates = (
            cand_df.sort_values(["party", "votes"], ascending=[True, False])
            .groupby("party")["candidate"]
            .apply(list)
            .to_dict()
        )

        # Candidate tie notice
        candidate_tie_notices = {}
        for party, cand_list in party_to_sorted_candidates.items():
            votes = [candidate_vote_totals[c] for c in cand_list]

            # Find ties: candidates with the same vote as the top candidate
            if len(cand_list) > 1:
                top_vote = votes[0]
                tied = [c for c, v in zip(cand_list, votes) if v == top_vote]

                if len(tied) > 1:
                    
                    candidate_tie_notices[party] = {
                        "tied_candidates": tied,
                        "votes": top_vote,
                        "notice": f"Tie among candidates {tied} in party {party} with {top_vote} votes each."
                    }

        return candidate_vote_totals, party_vote_totals, party_to_sorted_candidates

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

    def remove_and_condense_scored(self,removed: List[str], profile: PreferenceProfile, remove_empty_ballots: bool = True, remove_zero_weight_ballots: bool = True,
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
        weight_f = weights[filter] 
        cand_scores_f = candidate_scores[filter, :]  

        totals = cand_scores_f.T.dot(weight_f)
        new_ballots = [Ballot(scores={cand: 1}, weight=wt) for cand, wt in zip(kept_cands_list, totals) if wt > 0]

        # return new profile
        return PreferenceProfile(
            ballots=new_ballots,
            candidates=kept_cands_list,
        )

    def _run_step(self, profile: PreferenceProfile, prev_state: ElectionState, store_states: bool = False,
    ) -> PreferenceProfile:
        """
        Run one step of an OpenListPR election from the given profile and previous state.

        Args:
            profile (PreferenceProfile): Profile of ballots.
            prev_state (ElectionState): The previous ElectionState.
            store_states (bool, optional): True if `self.election_states` should be updated with the
                ElectionState generated by this round. This should only be True when used by
                `self._run_election()`. Defaults to False.

        Returns:
            PreferenceProfile: The profile of ballots after the round is completed.
        """
        seats = {p: 0 for p in self._party_vote_totals}
        for state in self.election_states[1: prev_state.round_number + 1]:
            for s in state.elected:
                for c in s:
                    seats[self.party_map[c]] += 1

        df = self._base_party_df.copy()
        df["seats_so_far"] = pd.Series(seats)

        # compute party scores
        party_scores: Dict[str, float] = {}
        if self.divisor_method == "dhondt":
            df["divisor"] = 1 + df["seats_so_far"]
        elif self.divisor_method == "saint-lague":
            df["divisor"] = 2 * df["seats_so_far"] + 1
        elif self.divisor_method == "imperiali":
            df["divisor"] = df["seats_so_far"] + 2
        else:
            raise ValueError(f"Unknown divisor method: {self.divisor_method}")

        # Compute scores
        df["score"] = df.apply(
            lambda row: 0.0 if row["seats_so_far"] >= row["num_candidates"]
            else row["total_votes"] / row["divisor"],
            axis=1
        )
        party_scores = df["score"].to_dict()

        # pick party leader
        max_score = max(party_scores.values())
        leaders = [p for p, s in party_scores.items() if s == max_score and s > 0]

        if not leaders:
            raise ValueError("Not enough nominees to fill all seats.")
        if len(leaders) > 1:
            if self.tiebreak == "random":
                winner_party = random.choice(leaders)
            else:
                raise ValueError("Tiebreak policy provided not supported.")
        else:
            winner_party = leaders[0]

        tiebreaks = {}
        if len(leaders) > 1:
            tiebreaks = {
                "type": "party",
                "tied_parties": leaders,
                "winner": winner_party,
                "method": self.tiebreak
            }

        # fill seat with candidate from party leader
        idx = seats[winner_party]
        winner = self._party_to_sorted_candidates[winner_party][idx]

        # update profile
        next_prof = self.remove_and_condense_scored([winner], profile)

        # update state
        if store_states:
            self._winners.append(winner)
            self._party_seat_totals[winner_party] += 1
            self._last_scores = party_scores

            rem = [frozenset({c}) for c in profile.candidates_cast if c not in self._winners]
            rem.sort(key=lambda s: next(iter(s)))

            self.election_states.append(
                ElectionState(
                    round_number=prev_state.round_number + 1,
                    remaining=rem,
                    eliminated=tuple(),
                    elected=[frozenset({winner})],
                    scores=dict(party_scores),
                    tiebreaks=tiebreaks,
                )
            )
        return next_prof
    
    def _run_election(self):
        """
        Run the OpenListPR election process (overwrite parent _run_election).
        """
        # initialize
        self.election_states.append(
            ElectionState(
                round_number=0,
                remaining=tuple(frozenset({c}) for c in self._candidate_vote_totals),
                scores={},
                tiebreaks={},
            )
        )
        
        # run election steps until finished
        profile = self._profile
        prev_state = self.election_states[0]
        while not self._is_finished():
            profile = self._run_step(profile, prev_state, store_states=True)
            prev_state = self.election_states[-1]