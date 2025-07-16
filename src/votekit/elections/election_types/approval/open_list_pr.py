from typing import Dict, List, Optional
import random
from ....models import Election
from ....pref_profile import PreferenceProfile
from ...election_state import ElectionState
from ....cleaning import remove_and_condense_ranked_profile


class OpenListPR(Election):
    def __init__(
        self,
        profile: PreferenceProfile,
        m: int = 1,
        party_map: Optional[Dict[str, str]] = None,
        tiebreak: Optional[str] = None,
    ):
        if m <= 0:
            raise ValueError("m must be positive.")
        if party_map is None:
            raise ValueError("A party map must be provided for OpenListPR.")

        self.m = m
        self.tiebreak = tiebreak
        self.party_map = party_map
        self._validate_profile(profile)

        # Precompute once before the real run
        (
            self._candidate_votes,
            self._party_votes,
            self._party_lists,
        ) = self._precompute(profile)

        # Will be updated only in the real run during each step
        self._party_seats: Dict[str, int] = {p: 0 for p in self._party_votes}
        self._winners: List[str] = []
        self._last_scores: Dict[str, float] = {}

        super().__init__(profile=profile)
        
    def _validate_profile(self, profile: PreferenceProfile) -> None:
        for b in profile.ballots:
            if len(b.ranking) != 1:
                raise ValueError("Each ballot must rank exactly one candidate.")
            if b.weight < 0:
                raise ValueError("All ballots must have non‑negative weight.")

    def _precompute(self, profile: PreferenceProfile) -> tuple[Dict[str, int], Dict[str, int], Dict[str, List[str]]]:
        # tally candidates aggregates
        cand_votes: Dict[str, int] = {c: 0 for c in profile.candidates_cast}
        for b in profile.ballots:
            c = next(iter(b.ranking[0])) if isinstance(b.ranking[0], (set, frozenset)) else b.ranking[0]
            cand_votes[c] += b.weight

        # aggregate parties and sort party-candidate lists
        party_votes: Dict[str, int] = {}
        party_lists: Dict[str, List[str]] = {}
        for c, v in cand_votes.items():
            p = self.party_map[c]
            party_votes[p] = party_votes.get(p, 0) + v
            party_lists.setdefault(p, []).append(c)

        for p, lst in party_lists.items():
            party_lists[p] = sorted(lst, key=lambda c: (-cand_votes[c], random.random()))
        return cand_votes, party_votes, party_lists

    def _is_finished(self) -> bool:
        return len(self._winners) == self.m

    def _run_step(self, profile: PreferenceProfile, prev_state: ElectionState, store_states: bool = False,
    ) -> PreferenceProfile:
        
        # compute how many seats each party has so far (from prev_state history)
        seats_so_far: Dict[str, int] = {p: 0 for p in self._party_votes}
        for state in self.election_states[1: prev_state.round_number + 1]:
            for s in state.elected:
                for c in s:
                    seats_so_far[self.party_map[c]] += 1

        # compute party scores
        scores: Dict[str, float] = {}
        for p, tot in self._party_votes.items():
            scores[p] = 0.0 if seats_so_far[p] >= len(self._party_lists[p]) else tot / (1 + seats_so_far[p])

        # pick party leader
        max_score = max(scores.values())
        leaders = [p for p, s in scores.items() if s == max_score and s > 0]
        if not leaders:
            raise ValueError("Not enough nominees to fill all seats.")
        if len(leaders) > 1:
            if self.tiebreak == "random":
                winner_party = random.choice(leaders)
            else:
                raise ValueError("Tie between parties with no tiebreak.")
        else:
            winner_party = leaders[0]

        # fill seat with candidate from party leader
        idx = seats_so_far[winner_party]
        winner = self._party_lists[winner_party][idx]

        # update profile
        next_prof = remove_and_condense_ranked_profile([winner], profile)

        # update state
        if store_states:
            self._winners.append(winner)
            self._party_seats[winner_party] += 1
            self._last_scores = scores

            rem = [frozenset({c}) for c in profile.candidates_cast if c not in self._winners]
            rem.sort(key=lambda s: next(iter(s)))

            self.election_states.append(
                ElectionState(
                    round_number=prev_state.round_number + 1,
                    remaining=rem,
                    eliminated=tuple(),
                    elected=[frozenset({winner})],
                    scores=dict(scores),
                    tiebreaks={},
                )
            )

        return next_prof

    def run_election(self) -> Dict:
        if not self.election_states:
            self._run_election()
        return {
            "winners": tuple(self._winners),
            "party_seats": dict(self._party_seats),
            "last_scores": dict(self._last_scores),
        }
