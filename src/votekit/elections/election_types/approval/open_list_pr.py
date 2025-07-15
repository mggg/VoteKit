from typing import Dict, List, Tuple, Optional
import random
from ....models import Election
from ....pref_profile import PreferenceProfile
from ...election_state import ElectionState
from ....utils import (
    elect_cands_from_set_ranking,
)
from ....cleaning import remove_and_condense
from typing import Optional


class OpenListPR(Election):
    r"""
    Open-list party-list PR (one-vote version).

    • Each ballot chooses exactly one candidate.  
    • Seats are filled one-by-one using the divisor `party_votes / (1 + seats_won)`.  
      If a party has no nominees left, it simply drops out of the scoring
      pool for the remaining rounds.

    Args
    ----
    profile : PreferenceProfile
    m       : int                total seats
    party_map : dict[candidate, party], optional
    tiebreak : {"random", None}, optional  how to break party-score ties
    """
    def __init__(
            self,
            profile: PreferenceProfile,
            m: float = 1,
            party_map: Optional[Dict[str, str]] = None,
            tiebreak: Optional[str] = None,
    ):
        if m <= 0:
            raise ValueError("`m` must be positive.")
        
        self.m = m
        self.tiebreak = tiebreak
        self.party_map = party_map
        self._validate_profile(profile)

        self.party_seats: Dict[str, int] = {}
        self._party_lists: Dict[str, List[str]] = {}
        self.winners: List[str] = []
        self.party_scores: Dict[str, float] = {}
        
        super().__init__(
            profile=profile, 
        )

    def _validate_profile(self, profile):
        # All ballots must have a single vote
        for ballot in profile.ballots:
            if len(ballot.ranking) != 1: 
                raise ValueError("Each ballot must rank exactly one candidate.")
            
    def _is_finished(self) -> bool:
        """
        Checks if the election is finished (single-round election).

        Returns:
            bool: True if the election has completed (two states exist), False otherwise.
        """
        return len(self.winners) == self.m
            
    def _run_step(self, profile: PreferenceProfile, prev_state: ElectionState, store_states: bool = False
    ) -> PreferenceProfile:
        # Count candidate votes:
        self.candidate_votes: Dict[str, int] = {candidate: 0 for candidate in profile.candidates_cast}
        for ballot in profile.ballots:
            cand = ballot.ranking[0]
            
            if isinstance(cand, (set, frozenset)):
                cand = next(iter(cand))
            cand = str(cand)
            self.candidate_votes[cand] += 1

        # Count party votes
        self.party_votes: Dict[str, int] = {}
        for candidate, num_votes in self.candidate_votes.items():
            party = self.party_map[candidate] # Identify party of the candidate evaluated
            self.party_votes[party] = self.party_votes.get(party, 0) + num_votes # Add votes to the party total (for seat allocation)

        # Build & sort each party’s nominee list by votes
        for candidate in self.candidate_votes:
            self._party_lists.setdefault(self.party_map[candidate], []).append(candidate)

        for party, list_of_cands in self._party_lists.items():
            self._party_lists[party] = sorted(
                list_of_cands, key=lambda candidate: (-self.candidate_votes[candidate], random.random())
            )

        # Fill the seats:
        self.party_seats = {party: 0 for party in self._party_lists}
        round_no = 1

        while len(self.winners) < self.m:
            
            # Calculate scores for the currnet round
            self.party_scores = {}

            for party, nominees in self._party_lists.items():
                if self.party_seats[party] >= len(nominees):
                    self.party_scores[party] = 0.0 # If no nominees left, score is 0
                else:
                    divisor = 1 + self.party_seats[party]
                    self.party_scores[party] = self.party_votes[party] / divisor

            max_score = max(self.party_scores.values())
            top_parties = [party for party, score in self.party_scores.items() if score == max_score and score > 0]
            if not top_parties:                       # no one left to seat
                raise ValueError("Not enough nominees to fill all seats.")

            if len(top_parties) > 1:
                if self.tiebreak == "random":
                    winner_party = random.choice(top_parties)
                    tiebreak_note = {tuple(sorted(top_parties)): winner_party}
                else:
                    raise ValueError(
                        f"Tie between parties {top_parties} with no tiebreak strategy."
                    )
            else:
                winner_party = top_parties[0]
                tiebreak_note = {}

            seat_idx = self.party_seats[winner_party]
            winner_cand = self._party_lists[winner_party][seat_idx]

            self.winners.append(winner_cand)
            self.party_seats[winner_party] += 1

            if store_states:
                remaining = [candidate for candidate in self._profile.candidates if candidate not in self.winners]
                eliminated = [frozenset({c}) for c in self._profile.candidates if c not in self.winners]

                self.election_states.append(
                    ElectionState(
                        round_number=round_no,
                        remaining=[frozenset({c}) for c in remaining],
                        eliminated=eliminated,
                        elected=[frozenset({winner_cand}),],
                        scores=dict(self.party_scores),
                        tiebreaks=tiebreak_note,
                    )
                )

            round_no += 1

        return profile
    
    def run_election(self) -> Dict:
        """
        Return a compact summary once the election has already been executed
        (it is executed automatically by Election.__init__).

        Returns
        -------
        dict
            winners : tuple[str] final list of seated candidates
            party_seats : dict[str,int]
            last_scores : dict[str,float]  party scores
        """
        if not self.election_states:
            self._run_election()
        final_state = self.election_states[-1]
        winners = tuple(self.winners)
        last_scores = dict(self.party_scores)
        party_seats = dict(self.party_seats)

        return {
            "winners": winners,
            "party_seats": party_seats,
            "last_scores": last_scores,
        }
