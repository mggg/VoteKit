from typing import Dict, List, Optional
import random
from ....models import Election
from ....pref_profile import PreferenceProfile
from ...election_state import ElectionState
from ....cleaning import remove_and_condense_ranked_profile


class OpenListPR(Election):
    """
    Open-list party-list PR.
    """
    def __init__(
        self,
        profile: PreferenceProfile,
        m: int = 1,
        party_map: Optional[Dict[str, str]] = None,
        tiebreak: Optional[str] = None,
        divisor_method: Optional[str] = "dhondt",
    ): 
        """
        Initialize an OpenListPR election.

        Parameters
        ----------
        profile : PreferenceProfile
            The input preference profile.
        m : int, optional
            The number of seats to be filled. Defaults to 1.
        party_map : Dict[str, str], optional
            A dictionary mapping each candidate to a party. Defaults to None.
        tiebreak : str, optional
            A string indicating the method to break ties. Defaults to None.
        divisor_method : str, optional
            A string indicating the divisor method to use. Defaults to "dhondt".

        Raises
        ------
        ValueError
            If m is not positive or if party_map is None.
        """

        if m <= 0:
            raise ValueError("m must be positive.")
        if party_map is None:
            raise ValueError("A party map must be provided for OpenListPR.")

        self.m = m
        self.tiebreak = tiebreak
        self.party_map = party_map
        self.divisor_method = divisor_method
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
        """
        Validates the profile to ensure that each ballot ranks exactly one candidate
        and all weights are non-negative.

        Args:
            profile (PreferenceProfile): The profile containing ballots to be validated.

        Raises:
            ValueError: If a ballot does not rank exactly one candidate or if any ballot
            has a negative weight.
        """
        if profile.df["Weight"].min() < 0:
            raise ValueError("All ballots must have non‑negative weight.")
        
        if "Ranking_2" in profile.df.columns:
            raise ValueError("Each ballot must rank exactly one candidate.")

    def _precompute(self, profile: PreferenceProfile) -> tuple[Dict[str, int], Dict[str, int], Dict[str, List[str]]]:
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
        ## tally candidates aggregates
        cand_votes: Dict[str, int] = {c: 0 for c in profile.candidates_cast}
        df = profile.df.copy()
        df['Candidate'] = df['Ranking_1'].apply(lambda r: next(iter(r)) if isinstance(r, (set, frozenset)) else r[0])
        cand_votes = df.groupby('Candidate')['Weight'].sum().to_dict()

        df['Party'] = df['Candidate'].map(self.party_map)
        party_votes = df.groupby('Party')['Weight'].sum().to_dict()
        def sorted_candidates(group):
            votes = group.groupby('Candidate')['Weight'].sum()
            sorted_cands = votes.sort_values(ascending=False).index.tolist()
            return sorted_cands

        party_lists = df.groupby('Party').apply(sorted_candidates).to_dict()

        return cand_votes, party_votes, party_lists

    def _is_finished(self) -> bool:
        """
        Determines if the election process is complete.

        Returns:
            bool: True if the number of winners equals the number of seats to be filled (m),
            indicating that the election has finished. False otherwise.
        """
        return len(self._winners) == self.m

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
        # compute how many seats each party has so far (from prev_state history)
        seats_so_far: Dict[str, int] = {p: 0 for p in self._party_votes}
        for state in self.election_states[1: prev_state.round_number + 1]:
            for s in state.elected:
                for c in s:
                    seats_so_far[self.party_map[c]] += 1

        # compute party scores
        scores: Dict[str, float] = {}
        for party, total in self._party_votes.items():
            divisor = 0
            if self.divisor_method == "dhondt":
                divisor = 1 + seats_so_far[party]
            elif self.divisor_method == "saint-lague":
                divisor = 2 * seats_so_far[party] + 1
            elif self.divisor_method == "imperiali":
                divisor = seats_so_far[party] + 2
            else:
                raise ValueError(f"Unknown divisor method: {self.divisor_method}")
            
            scores[party] = 0.0 if seats_so_far[party] >= len(self._party_lists[party]) \
                else total / (1 + seats_so_far[party])

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
        return {
            "winners": tuple(self._winners),
            "party_seats": dict(self._party_seats),
            "last_scores": dict(self._last_scores),
        }
