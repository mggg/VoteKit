import networkx as nx

from votekit.pref_profile import RankProfile
from votekit.graphs.pairwise_comparison_graph import (
    pairwise_dict,
    get_dominating_tiers_digraph,
)
from votekit.utils import tiebreak_set

from votekit.elections.election_types.ranking.abstract_ranking import RankingElection
from votekit.elections.election_state import ElectionState


class RankedPairs(RankingElection):
    """
    See `Ranked Pairs <https://en.wikipedia.org/wiki/Ranked_pairs>`_ for more details.
    Any ambiguity is resolved lexicographically, so that the candidate with the
    first name alphabetically is preferred in the case of a tie.

    The idea of the Ranked Pairs election is to take the head-to-head results of the candidates
    and then sort them by the margin of victory. We then lock this order in and construct
    a directed graph from the head-to-head results by traversing the locked order and skipping
    any edges that would create a cycle in the directed graph. The final ranking of the election
    is then determined by the dominating tiers of the directed graph with ties broken
    lexicographically.

    Args:
        profile (RankProfile): Profile to conduct election on.
        m (int, optional): Number of seats to elect. Defaults to 1.
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

        Args:
            profile (RankProfile): Profile of ballots.
            prev_state (ElectionState): The previous ElectionState.
            store_states (bool, optional): Included for compatibility with the base class but not
                 used in this election type.

        Returns:
            RankProfile: The profile of ballots after the round is completed.
        """
        pairwise = pairwise_dict(profile)
        ordered_winners = {}
        # Determine the Digraph edges
        for (a, b), (weight_a, weight_b) in pairwise.items():
            if weight_a > weight_b:
                ordered_winners[(a, b)] = weight_a - weight_b
            if weight_b > weight_a:
                ordered_winners[(b, a)] = weight_b - weight_a

        # Lock the order
        sorted_winners = sorted(
            ordered_winners.items(), key=lambda x: x[1], reverse=True
        )

        graph: nx.DiGraph = nx.DiGraph()
        graph.add_nodes_from(profile.candidates_cast)

        for edge, _ in sorted_winners:
            # Skip cycles. Only need to check one direction of the path since we are
            # in a directed graph.
            if nx.has_path(graph, edge[1], edge[0]):
                continue
            graph.add_edge(edge[0], edge[1])

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
