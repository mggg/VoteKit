from fractions import Fraction
from itertools import permutations, combinations
import matplotlib.pyplot as plt  # type: ignore
import networkx as nx  # type: ignore
from functools import cache
from typing import Optional

from ..ballot import Ballot
from .base_graph import Graph
from ..pref_profile import PreferenceProfile


class PairwiseComparisonGraph(Graph):
    """
    Class to construct the pairwise comparison graph where nodes are candidates
    and edges are pairwise preferences.

    Args:
        profile (PreferenceProfile): ``PreferenceProfile`` to construct graph from.
        ballot_length (int, optional): Max length of ballot. If not provided,
            defaults to number of candidates.

    Attributes:
        profile (PreferenceProfile): ``PreferenceProfile`` to construct graph from.
        ballot_length (int): Max length of ballot.
        candidates (list): List of candidates.
        pairwise_dict (dict): Dictionary constructed from ``compute_pairwise_dict``.
        pairwise_graph (networkx.DiGraph): Underlying graph.
    """

    def __init__(self, profile: PreferenceProfile, ballot_length: Optional[int] = None):
        if ballot_length is None:
            self.ballot_length = len(profile.candidates)
        else:
            self.ballot_length = ballot_length
        full_profile = self.ballot_fill(profile, self.ballot_length)
        self.profile = full_profile
        self.candidates = self.profile.candidates
        self.pairwise_dict = self.compute_pairwise_dict()
        self.pairwise_graph = self.build_graph()

    def ballot_fill(self, profile: PreferenceProfile, ballot_length: int):
        """
        Fills incomplete ballots for pairwise comparison.

        Args:
            profile (PreferenceProfile): ``PreferenceProfile`` to fill.
            ballot_length (int): How long a ballot is.

        Returns:
            PreferenceProfile: A ``PreferenceProfile`` with incomplete ballots filled in.
        """
        cand_list = [{cand} for cand in profile.candidates]
        updated_ballot_list = []

        for ballot in profile.ballots:
            if not ballot.ranking:
                raise TypeError("Ballots must have rankings.")
            if len(ballot.ranking) < ballot_length:
                missing_cands = [
                    cand for cand in cand_list if cand not in ballot.ranking
                ]
                missing_cands_perms = list(
                    permutations(missing_cands, len(missing_cands))
                )
                frac_freq = ballot.weight / (len(missing_cands_perms))
                for perm in missing_cands_perms:
                    updated_rank = ballot.ranking + tuple([frozenset(c) for c in perm])
                    updated_ballot = Ballot(
                        ranking=updated_rank, weight=Fraction(frac_freq, 1)
                    )
                    updated_ballot_list.append(updated_ballot)
            else:
                updated_ballot_list.append(ballot)
        return PreferenceProfile(ballots=tuple(updated_ballot_list))

    # Helper functions to make pairwise comparison graph
    def head2head_count(self, cand1: str, cand2: str) -> Fraction:
        """
        Counts head to head comparisons between two candidates. Note that the given order
        of the candidates matters here.

        Args:
            cand1 (str): The first candidate to compare.
            cand2 (str): The second candidate to compare.

        Returns:
            Fraction: A count of the number of times cand1 is preferred to cand2.
        """
        count = 0
        ballots_list = self.profile.ballots
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
        """
        Constructs dictionary where keys are tuples ``(cand_a, cand_b)`` containing
        two candidates and values is the frequency cand_a is preferred to
        cand_b.

        Returns:
            dict:
                A dictionary with keys = (cand_a, cand_b) and values = frequency cand_a is
                preferred to cand_b.
        """
        pairwise_dict = {}  # {(cand_a, cand_b): freq cand_a is preferred over cand_b}
        cand_pairs = combinations(self.candidates, 2)

        for pair in cand_pairs:
            cand_a, cand_b = pair[0], pair[1]
            head_2_head_dict = {
                (cand_a, cand_b): self.head2head_count(cand_a, cand_b),
                (cand_b, cand_a): self.head2head_count(cand_b, cand_a),
            }
            max_pair = max(zip(head_2_head_dict.values(), head_2_head_dict.keys()))
            if (
                self.head2head_count(cand_a, cand_b)
                - self.head2head_count(cand_b, cand_a)
                != 0
            ):
                pairwise_dict[max_pair[1]] = abs(
                    self.head2head_count(cand_a, cand_b)
                    - self.head2head_count(cand_b, cand_a)
                )
            else:
                pairwise_dict[(cand_a, cand_b)] = Fraction(0)
                pairwise_dict[(cand_b, cand_a)] = Fraction(0)

            ## would display x:y instead of abs(x-y)
            # winner, loser = max_pair[1]
            # pairwise_dict[max_pair[1]] = f"{head_2_head_dict[(winner, loser)]}: \
            # {head_2_head_dict[(loser, winner)]}"

        return pairwise_dict

    def build_graph(self) -> nx.DiGraph:
        """
        Builds the networkx pairwise comparison graph.

        Returns:
            networkx.DiGraph: The networkx digraph representing the pairwise comparison graph.
        """
        G = nx.DiGraph()
        G.add_nodes_from(self.candidates)
        for e in self.pairwise_dict.keys():
            G.add_edge(e[0], e[1], weight=self.pairwise_dict[e])
        return G

    def draw(self, outfile=None):
        """
        Draws pairwise comparison graph.

        Args:
            outfile (str): The filepath to save the graph. Defaults to not saving.
        """
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
            G, pos, edge_labels=edge_labels, label_pos=1 / 3, font_size=10
        )
        # Out stuff
        if outfile is not None:
            plt.savefig(outfile)
        else:
            plt.show()
        plt.close()

    def has_condorcet_winner(self) -> bool:
        """
        Checks if graph has a condorcet winner.

        Returns:
            bool: True if condorcet winner exists, False otherwise.
        """
        dominating_tiers = self.dominating_tiers()
        if len(dominating_tiers[0]) == 1:
            return True
        return False

    def get_condorcet_winner(self) -> str:
        """
        Returns the condorcet winner. Raises a ValueError if no condorcet winner.

        Returns:
            str: The condorcet winner.
        """

        if self.has_condorcet_winner():
            return list(self.dominating_tiers()[0])[0]

        else:
            raise ValueError("There is no condorcet winner.")

    @cache
    def dominating_tiers(self) -> list[set]:
        """
        Finds dominating tiers within an election.

        Returns:
            list[set]: A list of dominating tiers.
        """
        beat_set_size_dict = {}
        for cand in self.candidates:
            beat_set = set()
            for other_cand in self.candidates:
                if cand != other_cand:
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

    def has_condorcet_cycles(self) -> bool:
        """
            Checks if graph has any condorcet cycles, which we define as any cycle of length
            greater than 2 in the graph.

        Returns:
            bool: True if condorcet cycles exists, False otherwise.
        """

        if len(self.get_condorcet_cycles()) > 0:
            return True

        else:
            return False

    @cache
    def get_condorcet_cycles(self) -> list:
        """
            Returns a list of condorcet cycles in the graph, which we define as any cycle of length
            greater than 2.

        Returns:
            list: List of condorcet cycles sorted by length.
        """

        G = self.pairwise_graph
        list_of_cycles = nx.recursive_simple_cycles(G)
        return sorted(list_of_cycles, key=lambda x: len(x))
