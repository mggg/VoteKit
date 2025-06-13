from itertools import combinations
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.axes import Axes
import matplotlib.patches as mpatches
import networkx as nx  # type: ignore
from functools import cache
from typing import Optional
from ..pref_profile import PreferenceProfile
from ..pref_profile.utils import _convert_ranking_cols_to_ranking


def pairwise_dict(
    profile: PreferenceProfile,
) -> dict[tuple[str, ...], list[float]]:
    """
    Computes a dictionary whose keys are candidate pairs (A,B) and whose values are lists [a,b]
    where 'a' denotes the number of times A beats B head to head, and 'b' is the reverse.

    Args:
        profile (PreferenceProfile): Profile to compute dict on.

    Returns:
        dict[tuple[str, ...], list[float]]: Pairwise comparison dictionary.

    """

    if profile.contains_scores:
        raise ValueError("Profile must only contain rankings, not scores.")
    elif not profile.contains_rankings:
        raise ValueError("Profile must contain rankings.")
    pairwise_dict = {
        tuple(sorted((c1, c2))): [0.0, 0.0]
        for c1, c2 in combinations(profile.candidates, 2)
    }

    candidate_set = set(profile.candidates)

    mentioned_pairs_dict: dict[str, list[tuple[str, str]]] = {
        c: [] for c in profile.candidates
    }
    for c1, c2 in pairwise_dict.keys():
        mentioned_pairs_dict[c1].append((c1, c2))
        mentioned_pairs_dict[c2].append((c1, c2))

    sets_mentioned_pairs_dict = {
        c: set(mentioned_pairs_dict[c]) for c in mentioned_pairs_dict
    }

    for _, row in profile.df.iterrows():
        ranking = _convert_ranking_cols_to_ranking(
            row, max_ranking_length=profile.max_ranking_length
        )

        mentioned_so_far = set()
        if ranking is not None:
            for i, cand_set in enumerate(ranking):
                if len(cand_set) > 1:
                    raise ValueError(
                        "Pairwise dict does not support profiles with ties in ballots."
                    )
                cand = next(iter(cand_set))
                if cand in mentioned_so_far:
                    continue

                mentioned_so_far.add(cand)

                remaining_candidates = candidate_set - mentioned_so_far
                for c1, c2 in sets_mentioned_pairs_dict[cand]:
                    if c1 in remaining_candidates or c2 in remaining_candidates:
                        if c1 == cand:
                            pairwise_dict[(c1, c2)][0] += row["Weight"]
                        else:
                            pairwise_dict[(c1, c2)][1] += row["Weight"]
    return pairwise_dict


def restrict_pairwise_dict_to_subset(
    cand_subset: list[str] | tuple[str] | set[str],
    pairwise_dict: dict[tuple[str, ...], list[float]],
) -> dict[tuple[str, ...], list[float]]:
    """
    Restricts the full pairwise dictionary to a subset of candidates. The pairwise dictionary is a
    dictionary whose keys are candidate pairs (A,B) and whose values are lists [a,b]
    where 'a' denotes the number of times A beats B head to head, and 'b' is the reverse.

    Args:
        cands (list[str] | tuple[str] | set[str]): Candidate subset to restrict to.
        pairwise_dict (dict[tuple[str, ...], tuple[int,int]]): Full pairwise comparison dictionary.

    Returns:
        dict[tuple[str, ...], list[float]]: Pairwise dict restricted to the provided
            candidates.

    Raises:
        ValueError: cand_subset must be at least length 2.
        ValueError: cand_subset must be a subset of the candidates in the dictionary.
    """
    if len(cand_subset) < 2:
        raise ValueError(
            f"Must be at least two candidates in cand_subset: {cand_subset}"
        )

    candidates = [c for s in pairwise_dict.keys() for c in s]

    extra_cands = set(cand_subset).difference(candidates)
    if extra_cands != set():
        raise ValueError(
            (
                f"{extra_cands} are found in cand_subset but "
                f"not in the list of candidates found in the dictionary: {candidates}"
            )
        )

    new_pairwise_dict = {}
    for c1, c2 in combinations(cand_subset, 2):
        tup = tuple(sorted([c1, c2]))
        new_pairwise_dict[tup] = pairwise_dict[tup]
    return new_pairwise_dict


class PairwiseComparisonGraph(nx.DiGraph):
    """
    Class to construct the pairwise comparison graph where nodes are candidates
    and edges are pairwise preferences.

    Args:
        profile (PreferenceProfile): ``PreferenceProfile`` to construct graph from.

    Attributes:
        profile (PreferenceProfile): ``PreferenceProfile`` to construct graph from.
        candidates (list): List of candidates.
        pairwise_dict (dict[tuple[str, str], list[float]]): Dictionary constructed from
            ``pairwise_dict``. The pairwise dictionary is a
            dictionary whose keys are candidate pairs (A,B) and whose values are lists [a,b]
            where 'a' denotes the number of times A beats B head to head, and 'b' is the reverse.
        pairwise_graph (networkx.DiGraph): Underlying graph.
    """

    def __init__(self, profile: PreferenceProfile):
        self.profile = profile
        self.pairwise_dict = pairwise_dict(profile)
        self.pairwise_graph = self._build_graph()

    def _build_graph(self) -> nx.DiGraph:
        """
        Constructs the pairwise comparison graph from the pairwise comparison dictionary.
        """
        G = nx.DiGraph()
        decoupled = [item for tup in self.pairwise_dict.keys() for item in tup]
        G.add_nodes_from(decoupled)

        for (c1, c2), (v1, v2) in self.pairwise_dict.items():
            if v1 > v2:
                if (c1, c2) not in G.edges():
                    G.add_edge(c1, c2, weight=v1 - v2)
            elif v2 > v1:
                if (c2, c1) not in G.edges():
                    G.add_edge(c2, c1, weight=v2 - v1)
            else:
                # Add in the tie edges so `ties_or_beats` works
                if (c1, c2) not in G.edges():
                    G.add_edge(c1, c2, weight=0)
                    G.add_edge(c2, c1, weight=0)

        return G

    def ties_or_beats(self, candidate: str) -> set[str]:
        """
        Returns the predecessors of  x, which are all of the nodes m that have a directed
        path from m to x. In the pairwise comparison graph, these are the candidates that tie or
        beat the given candidate.

        Args:
            candidate (str): Candidate.

        Returns:
            set[str]: Candidates that beat or tie given candidate head to head.

        """
        return set(self.pairwise_graph.predecessors(candidate))

    @cache
    def get_dominating_tiers(self) -> list[set[str]]:
        """
        Compute the dominating tiers of the pairwise comparison graph.
        Candidates in a tier beat all other candidates in lower tiers in head to head comparisons.

        Returns:
            list[set[str]]: Dominating tiers, where the first entry of the list is the highest tier.

        """
        dominating_tiers = []

        G_left = self.pairwise_graph.copy()

        while len(G_left.nodes()) > 0:
            start_dom = set({min(map(lambda x: (x[1], x[0]), G_left.in_degree()))[1]})
            new_dom = start_dom.copy()
            while True:
                for node in start_dom:
                    new_dom = new_dom.union(set(G_left.predecessors(node)))

                if new_dom == start_dom:
                    break

                start_dom = new_dom

            dominating_tiers.append(new_dom)
            G_left.remove_nodes_from(new_dom)

        return dominating_tiers

    def has_condorcet_winner(self) -> bool:
        """
        Checks if graph has a condorcet winner.

        Returns:
            bool: True if condorcet winner exists, False otherwise.
        """
        dominating_tiers = self.get_dominating_tiers()
        return len(dominating_tiers[0]) == 1

    def get_condorcet_winner(self) -> str:
        """
        Returns the condorcet winner. Raises a ValueError if no condorcet winner.

        Returns:
            str: The condorcet winner.

        Raises:
            ValueError: There is no condorcet winner.
        """

        if self.has_condorcet_winner():
            return list(self.get_dominating_tiers()[0])[0]

        else:
            raise ValueError("There is no condorcet winner.")

    @cache
    def get_condorcet_cycles(self) -> list[set[str]]:
        """
        Returns a list of condorcet cycles in the graph, which we define as any cycle of length
        greater than 2.

        Returns:
            list[set[str]]: List of condorcet cycles sorted by length.
        """

        list_of_cycles = nx.recursive_simple_cycles(self.pairwise_graph)
        return [set(x) for x in sorted(list_of_cycles, key=lambda x: len(x))]

    def has_condorcet_cycles(self) -> bool:
        """
        Checks if graph has any condorcet cycles, which we define as any cycle of length
        greater than 2 in the graph.

        Returns:
            bool: True if condorcet cycles exists, False otherwise.
        """

        return len(self.get_condorcet_cycles()) > 0

    def draw(
        self,
        ax: Optional[Axes] = None,
        candidate_list: Optional[list[str]] = None,
    ) -> Axes:
        """
        Draws pairwise comparison graph.

        Args:
            ax (Axes, optional): Matplotlib axes to plot on. Defaults to None, in which case
                an axes is generated.
            candidate_list (list[str], optional): List of candidates to plot. Defaults to None,
                in which case all candidates are used.

        Returns:
            Axes: Matplotlib axes of pairwise comparison graph.
        """
        G = self.pairwise_graph

        known_candidate_list = list(G.nodes)
        if candidate_list is None:
            candidate_list = known_candidate_list

        assert set(candidate_list).issubset(set(G.nodes)), (
            f"Invalid candidates found: {set(candidate_list) - set(G.nodes)} does not appear as "
            f"a subset of known candidates {set(G.nodes)}"
        )

        G = G.subgraph(candidate_list).copy()

        ranking_labels = {c: i for i, c in enumerate(candidate_list)}

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))

        pos = nx.circular_layout(G)

        nx.draw(G, pos, ax=ax, with_labels=False, node_color="lightblue", node_size=300)
        nx.draw_networkx_labels(
            G, pos, ax=ax, labels=ranking_labels, font_size=10, font_color="black"
        )

        edge_weights = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(
            G, pos, ax=ax, edge_labels=edge_weights, font_size=10, label_pos=0.15
        )
        nx.draw_networkx_edge_labels(
            G, pos, ax=ax, edge_labels=edge_weights, font_size=10, label_pos=0.85
        )

        artists = []
        for label, number in ranking_labels.items():
            patch = mpatches.Patch(color="white", label=f"{number}: {label}")
            artists.append(patch)

        leg = ax.legend(
            handles=artists,
            labels=[f"{i}: {label}" for i, label in enumerate(candidate_list)],
            loc="center left",
            bbox_to_anchor=(1.03, 0.5),
            fontsize=10,
            frameon=True,
            borderaxespad=0.0,
            handlelength=0,
            handletextpad=0,
            fancybox=True,
        )

        for item in leg.legend_handles:
            if item:
                item.set_visible(False)

        return ax
