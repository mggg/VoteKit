from fractions import Fraction
from itertools import permutations, combinations
import matplotlib.pyplot as plt  # type: ignore
import networkx as nx  # type: ignore

from ..ballot import Ballot
from .base_graph import Graph
from ..pref_profile import PreferenceProfile


class PairwiseComparisonGraph(Graph):
    def __init__(self, profile: PreferenceProfile, ballot_length=None):
        self.ballot_length = ballot_length
        if ballot_length is None:
            self.ballot_length = len(profile.get_candidates())
        full_profile = self.ballot_fill(profile, self.ballot_length)
        self.profile = full_profile
        self.candidates = self.profile.get_candidates()
        self.pairwise_dict = self.compute_pairwise_dict()
        self.pairwise_graph = self.build_graph()

    # Function to fill in incomplete ballots for pairwise comparison
    def ballot_fill(self, profile: PreferenceProfile, ballot_length: int):
        cand_list = [{cand} for cand in profile.get_candidates()]
        updated_ballot_list = []

        for ballot in profile.get_ballots():
            if len(ballot.ranking) < ballot_length:
                missing_cands = [
                    cand for cand in cand_list if cand not in ballot.ranking
                ]
                missing_cands_perms = list(
                    permutations(missing_cands, len(missing_cands))
                )
                frac_freq = ballot.weight / (len(missing_cands_perms))
                for perm in missing_cands_perms:
                    updated_rank = ballot.ranking + list(perm)
                    updated_ballot = Ballot(
                        ranking=updated_rank, weight=Fraction(frac_freq, 1)
                    )
                    updated_ballot_list.append(updated_ballot)
            else:
                updated_ballot_list.append(ballot)
        return PreferenceProfile(ballots=updated_ballot_list)

    # Helper functions to make pairwise comparison graph
    def head2head_count(self, cand1, cand2) -> Fraction:
        count = 0
        ballots_list = self.profile.get_ballots()
        for ballot in ballots_list:
            rank_list = ballot.ranking
            for s in rank_list:
                if cand1 in s:
                    count += ballot.weight
                    break
                elif cand2 in s:
                    break
        return Fraction(count)

    def compute_pairwise_dict(self) -> dict:
        pairwise_dict = {}  # {(cand_a, cand_b): freq cand_a is preferred over cand_b}
        cand_pairs = combinations(self.candidates, 2)

        for pair in cand_pairs:
            cand_a, cand_b = pair[0], pair[1]
            head_2_head_dict = {
                (cand_a, cand_b): self.head2head_count(cand_a, cand_b),
                (cand_b, cand_a): self.head2head_count(cand_b, cand_a),
            }
            max_pair = max(zip(head_2_head_dict.values(), head_2_head_dict.keys()))
            pairwise_dict[max_pair[1]] = abs(
                self.head2head_count(cand_a, cand_b)
                - self.head2head_count(cand_b, cand_a)
            )

        return pairwise_dict

    def build_graph(self) -> nx.DiGraph:
        G = nx.DiGraph()
        G.add_nodes_from(self.candidates)
        for e in self.pairwise_dict.keys():
            G.add_edge(e[0], e[1], weight=self.pairwise_dict[e])
        return G

    def draw(self, outfile=None):
        G = self.pairwise_graph

        pos = nx.circular_layout(G)
        nx.draw_networkx(
            G,
            pos,
            with_labels=True,
            node_size=500,
            node_color="skyblue",
            edgelist=list(),
        )
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=G.edges,
            width=1.5,
            edge_color="b",
            arrows=True,
            alpha=1,
            node_size=1000,
            arrowsize=25,
        )
        edge_labels = {(i, j): G[i][j]["weight"] for i, j in G.edges()}
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels, label_pos=0.5, font_size=10
        )
        # Out stuff
        if outfile is not None:
            plt.savefig(outfile)
        else:
            plt.show()
        plt.close()

    # More complicated Requests
    def has_condorcet(self) -> bool:
        dominating_tiers = self.dominating_tiers()
        if len(dominating_tiers[0]) == 1:
            return True
        return False

    def dominating_tiers(self) -> list[set]:
        beat_set_size_dict = {}
        for i, cand in enumerate(self.candidates):
            beat_set = set()
            for j, other_cand in enumerate(self.candidates):
                if i != j:
                    if nx.has_path(self.pairwise_graph, cand, other_cand):
                        beat_set.add(other_cand)
            beat_set_size_dict[cand] = len(beat_set)

        # We want to return candidates sorted and grouped by beat set size
        tier_dict: dict = {}
        for k, v in beat_set_size_dict.items():
            if v in tier_dict.keys():
                tier_dict[v].add(k)
            else:
                tier_dict[v] = {k}
        tier_list = [tier_dict[k] for k in sorted(tier_dict.keys(), reverse=True)]
        return tier_list
