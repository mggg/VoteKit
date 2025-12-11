import networkx as nx
import numpy as np

from votekit.pref_profile import RankProfile
from votekit.graphs.pairwise_comparison_graph import (
    pairwise_dict,
    get_dominating_tiers_digraph,
)
from votekit.utils import tiebreak_set

from votekit.elections.election_types.ranking.abstract_ranking import RankingElection
from votekit.elections.election_state import ElectionState


class Schulze(RankingElection):
    """
    See <https://link.springer.com/article/10.1007/s00355-010-0475-4> and <https://arxiv.org/pdf/1804.02973>

    The Schulze method uses the widest path algorithm to determine winners. For each pair
    of candidates, it computes the strength of the strongest path (where the strength of
    a path is the strength of its weakest link). Candidate A is preferred to candidate B
    if the strongest path from A to B is stronger than the strongest path from B to A.

    The Schulze method computes the strongest paths between all pairs of candidates:
    1. Initialize p[i,j] = d[i,j] - d[j,i] (margin of victory)
    2. For each intermediate candidate k, update p[i,j] = max(p[i,j], min(p[i,k], p[k,j]))
    3. Candidate i beats j if p[i,j] > p[j,i]

    Args:
        profile (RankProfile): Profile to conduct election on.
        m (int, optional): Number of seats to elect. Defaults to 1.
        tiebreak (str, optional): Method for breaking ties. Defaults to "lexicographic".
    """

    def __init__(
        self,
        profile: RankProfile,
        tiebreak: str = "lexicographic",
        m: int = 1,
    ):
        if m <= 0:
            raise ValueError("m must be strictly positive")
        if len(profile.candidates_cast) < m:
            raise ValueError("Not enough candidates received votes to be elected.")
        self.m = m
        self.tiebreak = tiebreak

        def quick_tiebreak_candidates(profile: RankProfile) -> dict[str, float]:
            candidate_set = frozenset(profile.candidates)
            tiebroken_candidates = tiebreak_set(candidate_set, tiebreak=self.tiebreak)

            if len(tiebroken_candidates) != len(profile.candidates):
                raise RuntimeError("Tiebreak did not resolve all candidates.")

            return {next(iter(c)): i for i, c in enumerate(tiebroken_candidates[::-1])}

        super().__init__(
            profile,
            score_function=quick_tiebreak_candidates,
            sort_high_low=True,
        )

    def _is_finished(self):
        """
        Check if the election is finished.
        """
        # single round election
        elected_cands = [c for s in self.get_elected() for c in s]

        if len(elected_cands) == self.m:
            return True
        return False

    def _run_step(
        self, profile: RankProfile, prev_state: ElectionState, store_states=False
    ) -> RankProfile:
        """
        Run one step of an election from the given profile and previous state. Since this is
        a single-round election, this will complete the election and return the final profile.

        The Schulze method computes the strongest paths between all pairs of candidates:
        1. Initialize p[i,j] = d[i,j] - d[j,i] (margin of victory)
        2. For each intermediate candidate k, update p[i,j] = max(p[i,j], min(p[i,k], p[k,j]))
        3. Candidate i beats j if p[i,j] > p[j,i]

        Args:
            profile (RankProfile): Profile of ballots.
            prev_state (ElectionState): The previous ElectionState.
            store_states (bool, optional): Included for compatibility with the base class but not
                 used in this election type.

        Returns:
            RankProfile: The profile of ballots after the round is completed.
        """
        # Get pairwise comparison data: d[i,j] = number of voters who prefer i to j
        pairwise = pairwise_dict(profile)
        candidates = list(profile.candidates_cast)
        n = len(candidates)

        # Create candidate index mapping
        cand_to_idx = {cand: idx for idx, cand in enumerate(candidates)}

        # Initialize p[i,j] matrix (strongest path strengths) using NumPy
        # p[i,j] represents the strength of the strongest path from i to j
        p = np.zeros((n, n), dtype=np.float64)

        # Step 1: Initialize p[i,j] = d[i,j] - d[j,i] for all pairs (i != j)
        # pairwise_dict returns (a, b): (weight_a, weight_b) where:
        #   weight_a = number of voters preferring a to b
        #   weight_b = number of voters preferring b to a
        for (a, b), (weight_a, weight_b) in pairwise.items():
            i = cand_to_idx[a]
            j = cand_to_idx[b]
            # p[i,j] is the margin by which i beats j (can be negative if j beats i)
            p[i, j] = weight_a - weight_b
            # Also set the reverse direction
            p[j, i] = weight_b - weight_a

        # Step 2: Floyd-Warshall style algorithm to compute strongest (widest) paths
        # Schulze requires: p[i,j] = max(p[i,j], min(p[i,k], p[k,j]))
        # We use NumPy broadcasting to vectorize the inner two loops for performance.
        for k in range(n):
            # p[:, k:k+1] is column k (shape n x 1), p[k:k+1, :] is row k (shape 1 x n)
            p = np.maximum(p, np.minimum(p[:, k : k + 1], p[k : k + 1, :]))

        # Step 3: Build directed graph where i -> j if p[i,j] > p[j,i]
        graph: nx.DiGraph = nx.DiGraph()
        graph.add_nodes_from(candidates)

        for i in range(n):
            for j in range(n):
                if i != j and p[i, j] > p[j, i]:
                    graph.add_edge(candidates[i], candidates[j])

        # Get dominating tiers from the graph
        dominating_tiers = get_dominating_tiers_digraph(graph)

        tiebreak_resolutions = {}
        for candidate_tier_set in dominating_tiers:
            if len(candidate_tier_set) > 1:
                tiebreak_resolutions[frozenset(candidate_tier_set)] = tiebreak_set(
                    frozenset(candidate_tier_set), tiebreak=self.tiebreak
                )

        ordered_candidates = [
            candidate
            for candidate_set in dominating_tiers
            for candidate in sorted(candidate_set)
        ]

        elected = tuple(frozenset({c}) for c in ordered_candidates[: self.m])
        remaining = tuple(frozenset({c}) for c in ordered_candidates[self.m :])

        if store_states:
            new_state = ElectionState(
                round_number=prev_state.round_number + 1,
                elected=elected,
                remaining=remaining,
                tiebreaks=tiebreak_resolutions,
            )

            self.election_states.append(new_state)

        return profile
