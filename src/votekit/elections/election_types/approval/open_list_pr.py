from typing import Dict, List, Optional
import pandas as pd
from collections import defaultdict, Counter
import random
from votekit.elections.election_types.scores.rating import GeneralRating
from ....pref_profile import PreferenceProfile
from ...election_state import ElectionState

class OpenListPR(GeneralRating):
    """
    Open-list party-list proportional representation (PR) election method.

    In Open-list, each party gives a list of candidates to elect. The total number of seats 
    allocated to each party is determined by a function that takes in the sum of votes recieved by
    each party and their individual candidates as well as the current party seat share. Once a 
    party's seats are determined, the candidates are elected from the party's list in order of 
    their individual vote totals. 
    """
    def __init__(
        self,
        profile: PreferenceProfile,
        party_to_candidate_map: (Dict[str, List[str]]),
        m: int = 1,
        tiebreak: Optional[str] = "random",
        divisor_method: str = "dhondt",
        L: float = 1,
        k: float = 1,
    ): 
        """
        Initialize an OpenListPR election.

        Args:
             profile (PreferenceProfile): Profile containing a list of Ballot objects with approval scores of candidates.
             party_to_candidate_map (Dict[str, list[str]]): A dictionary mapping each party to list of candidates.
             m (int): Number of seats to be filled. Defaults to 1.
             tiebreak (str): Tiebreak method to use. Options are None or 'random'. Defaults to 'random'.
             divisor_method (str): Method to use for calculating party seat allocation.
                 Options are "dhondt" (i.e. "Jefferson Method), "saint-lague", or "imperiali". Defaults to "dhondt".
             L (float): Score limit. Defaults to 1.
             k (float): Total score limit for each ballot. Defaults to 1.
        """

        if m <= 0:
            raise ValueError("m must be positive.")
        self.m = m
        self.tiebreak = tiebreak
        self.divisor_method = divisor_method
        self.L = L
        self.k = k

        self.party_to_candidate_map = party_to_candidate_map
        self._validate_profile(profile)

        (
            self._candidate_vote_totals,
            self._party_vote_totals,
            self._party_to_sorted_candidates,
            self.candidate_tie_notices
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

        self._party_scores: dict[str, float] = {}
        self._initialize_party_scores()
        super().__init__(profile=profile, m=m, tiebreak=tiebreak)
        
    def _precompute_scores(self, profile: PreferenceProfile) -> tuple[Dict[str, int], Dict[str, int], Dict[str, List[str]], Dict[str, List[str]]]:
        """
        This function computes the total number of votes for each candidate, the
        total number of votes for each party, and sorts the list of candidates
        for each party in descending order of votes.

        Args:
            profile (PreferenceProfile): Profile containing a list of Ballot objects with approval scores of candidates.

        Returns:
            tuple[Dict[str, int], Dict[str, int], Dict[str, List[str]]]:
                A tuple of three dictionaries. The first dictionary maps each candidate
                to its total number of votes. The second dictionary maps each party to
                its total number of votes. The third dictionary maps each party to a
                sorted list of its candidates in descending order of votes.
        """
        df = profile.df
        candidate_cols = list(profile.candidates_cast)

        ## tally candidates aggregated
        weights = df["Weight"].to_numpy()
        ballot_votes = df[candidate_cols].fillna(0).to_numpy()
        candidate_totals_array = ballot_votes.T.dot(weights)

        # Calculate weighted votes for each candidate
        candidate_vote_totals = dict(zip(candidate_cols, candidate_totals_array))
        
        # Party vote totals
        party_vote_totals = {p : sum(candidate_vote_totals[c] for c in cand_list) for p, cand_list in self.party_to_candidate_map.items()}  

        # Build sorted party lists
        party_to_sorted_candidates = {
            party: sorted(
                cands,
                key=lambda c: candidate_vote_totals[c],
                reverse=True,
            )
            for party, cands in self.party_to_candidate_map.items()
        }

        # Candidate tie notice
        candidate_tie_notices = {}
        for party, cand_list in party_to_sorted_candidates.items():
            votes = [candidate_vote_totals[c] for c in cand_list]
            vote_counts = Counter(votes)
            top_vote_count = max(votes)
            top_ties = [c for c in cand_list if vote_counts[candidate_vote_totals[c]] == top_vote_count]

            # Candidate tie notice
            if len(top_ties) > 1:
                candidate_tie_notices[party] = top_ties

        return candidate_vote_totals, party_vote_totals, party_to_sorted_candidates, candidate_tie_notices

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

    def _initialize_party_scores(self):
        """
        Setup the running party seat totals and initial scores for the divisor method.
        """
        self._party_seat_totals = {p: 0 for p in self._party_vote_totals}
        for p in self._party_vote_totals:
            if self.divisor_method in {"dhondt", "jefferson"}:
                divisor = 1
            elif self.divisor_method == "saint-lague":
                divisor = 1
            elif self.divisor_method == "imperiali":
                divisor = 2
            else:
                raise ValueError("Not known divisor method provided.")
            self._party_scores[p] = self._party_vote_totals[p] / divisor

    def _update_party_scores(self, party):
        """
        Update the score for a single party after a seat win.

        Args:
            party (str): The party whose score is to be updated.
        """
        seats = self._party_seat_totals[party]
        num_candidates = len(self._party_to_sorted_candidates[party])

        if seats >= num_candidates:
            self._party_scores[party] = 0.0
            return

        if self.divisor_method in {"dhondt", "jefferson"}:
            divisor = 1 + seats
        elif self.divisor_method == "saint-lague":
            divisor = 2 * seats + 1
        elif self.divisor_method == "imperiali":
            divisor = seats + 2
        else:
            raise ValueError("Not known divisor method provided.")
        
        self._party_scores[party] = self._party_vote_totals[party] / divisor

    def _run_step(
        self, profile: PreferenceProfile, prev_state: ElectionState, store_states: bool = False
    ) -> PreferenceProfile:
        """
        Run one step of an OpenListPR election from the given profile and previous state.
        """
        # Determine available parties to elect a candidate
        available_parties = [p for p in self._party_scores if self._party_seat_totals[p] < len(self._party_to_sorted_candidates[p])]
        if not available_parties:
            raise ValueError("No parties have remaining candidates to elect.")

        # Find the leading party
        max_score = max(self._party_scores[p] for p in available_parties)
        leading_parties = [p for p in available_parties if self._party_scores[p] == max_score and self._party_scores[p] > 0]

        # Settle ties between leading parties and record tiebreak information
        tiebreaks: dict[frozenset[str], tuple[frozenset[str], ...]] = {}
        if len(leading_parties)==0:
            raise RuntimeError("All parties have a score of 0 before the end of the election.")

        if len(leading_parties) > 1:
            if self.tiebreak == "random":
                winner_party = random.choice(leading_parties)
            else:
                raise ValueError("Tiebreak policy provided not supported.")
            tiebreaks[frozenset(leading_parties)] = (frozenset({winner_party}),)

        else:
            winner_party = leading_parties[0]

        # Find the highest scoring candidate from the winning party
        idx = self._party_seat_totals[winner_party]
        winner = self._party_to_sorted_candidates[winner_party][idx]

        candidate_tie = self.candidate_tie_notices.get(winner_party)
        if candidate_tie is not None:
             tiebreaks[frozenset(candidate_tie)] = (frozenset({winner}),)

        # Update state and record the result
        if store_states:
            self._winners.append(winner)
            self._party_seat_totals[winner_party] += 1

            remaining_candidates = [c for c in profile.candidates_cast if c not in self._winners]
            candidate_groups_by_score: dict[int, List[str]] = defaultdict(list)
            for c in remaining_candidates:
                candidate_groups_by_score[self._candidate_vote_totals[c]].append(c)

            remaining_groups_by_score: list[frozenset[str]] = [frozenset(sorted(group)) for total, group in sorted(candidate_groups_by_score.items(), key=lambda gv: gv[0], reverse=True)]
            remaining_groups_by_score_tuple = tuple(remaining_groups_by_score)
            self.election_states.append(

                ElectionState(
                    round_number=prev_state.round_number + 1,
                    remaining=remaining_groups_by_score_tuple,
                    eliminated=tuple(),
                    elected=tuple([frozenset({winner}),]),
                    scores=dict(self._party_scores),
                    tiebreaks=tiebreaks,
                )
            )
            self._update_party_scores(winner_party)   
        return profile
    
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


