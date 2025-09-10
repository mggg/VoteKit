from itertools import combinations
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.axes import Axes
import matplotlib.patches as mpatches
import networkx as nx  # type: ignore
from functools import cache
from typing import Optional
from votekit.pref_profile import RankProfile
from numpy.typing import NDArray
import numpy as np  # type: ignore
from numba import njit, float64, int32  # type: ignore


def __rows_to_indices(
    profile: RankProfile, cand_name_to_idx: dict[str, int]
) -> NDArray:
    """
    Converts the ranking columns of a RankProfile to integer indices.
    Each singleton candidate set is converted to an index based on the provided candidates list.
    A special value of -2 is used for the short ballot candidate set {"~"} and -1 for undefined
    entries.

    Args:
        profile (RankProfile): The preference profile containing rankings.
        cand_name_to_idx (dict[str, int]): A mapping from candidate names to their integer index
            representations.

    Returns:
        NDArray: A tuple containing: An NDArray of integer indices representing the rankings.
    """
    fs_to_idx = {frozenset({c}): cand_name_to_idx[c] for c in cand_name_to_idx.keys()}
    fs_to_idx[frozenset({"~"})] = -2  # Make a special value for short ballots
    assert profile.max_ranking_length is not None
    ranking_cols = [f"Ranking_{i}" for i in range(1, profile.max_ranking_length + 1)]
    mat_obj = profile.df[ranking_cols].to_numpy(object)
    flat = mat_obj.ravel()
    flat_idx = np.empty(flat.shape[0], dtype=np.int32)
    for k, x in enumerate(flat):
        flat_idx[k] = fs_to_idx.get(x, -1)  # -1 for any undefined entry
    return flat_idx.reshape(mat_obj.shape).astype(np.int32)


# NOTE: There are a couple more optimizations that could be made here, such as avoiding the
# boolean indexing and using a visited epoch array, but this is already a significant
# speedup and is further optimization just makes this hard to read.
@njit((float64[:, :], int32[:, :], float64[:], int32), cache=True, fastmath=True)
def __tally_and_mutate_head_to_head(
    mutated_head_to_head_matrix: NDArray,
    integer_rankings_mat: NDArray,
    wt_vec: NDArray,
    n_cands: int,
):
    """
    Tallies the head-to-head matrix and mutates it in place.

    Args:
        mutated_head_to_head_matrix (NDArray): The head to head matrix to mutate.
        integer_rankings_mat (NDArray): The integer rankings matrix.
        wt_vec (NDArray): The weight vector.
        n_cands (int): The number of candidates.


    Returns:
        NDArray: The mutated head to head matrix.
    """
    for row_idx, row in enumerate(integer_rankings_mat):
        seen_vec = np.zeros(n_cands, dtype=np.bool_)
        for candset_idx in row:
            # -1 is for undefined entries
            if candset_idx == -1:
                continue
            # Special value for short ballots "~"
            if candset_idx == -2:
                break

            seen_vec[candset_idx] = True

            # Candidates always beat everyone that has not been seen yet in head to head
            mutated_head_to_head_matrix[candset_idx, ~seen_vec] += wt_vec[row_idx]

    return mutated_head_to_head_matrix


def pairwise_dict(
    profile: RankProfile, *, sort_candidate_pairs: bool = True
) -> dict[tuple[str, str], tuple[float, float]]:
    """
    Computes a dictionary whose keys are candidate pairs (A,B) and whose values are lists [a,b]
    where 'a' denotes the number of times A beats B head to head, and 'b' is the reverse.

    Args:
        profile (RankProfile): Profile to compute dict on.
        sort_candidate_pairs (bool): If True, candidate pairs in the pairwise comparison dictionary
            will be sorted lexicographically. Defaults to True.

    Returns:
        dict[tuple[str, str], tuple[float, float]]: Pairwise comparison dictionary.
    """
    if not isinstance(profile, RankProfile):
        raise ValueError("Profile must be of type RankProfile.")
    elif not all(b.ranking is not None for b in profile.ballots):
        raise ValueError("All ballots must have rankings.")

    candidates_lst = list(profile.candidates)

    if sort_candidate_pairs:
        candidates_lst.sort()

    n_cands = len(candidates_lst)

    cand_to_idx = {c: i for i, c in enumerate(candidates_lst)}
    integer_rankings_mat = __rows_to_indices(profile, cand_to_idx)
    wt_vec = profile.df["Weight"].to_numpy().astype(np.float64)

    head_to_head_matrix = np.zeros((n_cands, n_cands), dtype=np.float64)

    head_to_head_matrix = __tally_and_mutate_head_to_head(
        head_to_head_matrix, integer_rankings_mat, wt_vec, n_cands
    )

    pairwise = {
        (a, b): (
            head_to_head_matrix[cand_to_idx[a], cand_to_idx[b]],
            head_to_head_matrix[cand_to_idx[b], cand_to_idx[a]],
        )
        for a, b in combinations(sorted(candidates_lst), 2)
    }
    return pairwise


def get_dominating_tiers_digraph(graph: nx.DiGraph) -> list[set[str]]:
    """
    Compute the dominating tiers of the pairwise comparison graph.
    Candidates in a tier beat all other candidates in lower tiers in head to head comparisons.
    In other words, every candidate in a given tier must have a path to every candidate in
    the lower tier in the head-to-head graph.

    Args:
        graph (nx.DiGraph): A directed graph representing pairwise comparisons.

    Returns:
        list[set[str]]: Dominating tiers, where the first entry of the list is the highest tier.
    """
    # Condense the head-to-head cycles so we have a directed acyclic graph (DAG)
    condensed_acyclic_graph = nx.condensation(graph)
    quotient_generations = list(nx.topological_generations(condensed_acyclic_graph))

    node_to_descendants = {
        n: set(nx.descendants(condensed_acyclic_graph, n))
        for n in condensed_acyclic_graph.nodes
    }

    # Deal with unequal legs by checking checking the pairwise node sets for paths.
    required_merge = True
    while required_merge:
        merged_generations = []

        required_merge = False
        seen_nodes: set[int] = set()
        for src_nlist_idx, source_nlist in enumerate(quotient_generations):
            if set(source_nlist).issubset(seen_nodes):
                continue

            generation_nlist = source_nlist.copy()

            # Include the source nodes in the common set to guarantee termination
            common_descendant_set = set.intersection(
                *[node_to_descendants[n] for n in source_nlist]
            ) | set(generation_nlist)

            for target_nlist_idx in range(src_nlist_idx + 1, len(quotient_generations)):
                target_nlist = quotient_generations[target_nlist_idx]
                if set(target_nlist).issubset(common_descendant_set):
                    continue

                required_merge = True
                generation_nlist.extend(target_nlist)
                common_descendant_set = common_descendant_set.intersection(
                    *[node_to_descendants[n] for n in target_nlist],
                ) | set(generation_nlist)

            merged_generations.append(generation_nlist)
            seen_nodes.update(generation_nlist)

        quotient_generations = merged_generations

    # Now we need to unpack the quotient generations back into our dominating tiers.
    return [
        set().union(*[condensed_acyclic_graph.nodes[n]["members"] for n in nlist])
        for nlist in quotient_generations
    ]


def restrict_pairwise_dict_to_subset(
    cand_subset: list[str] | tuple[str] | set[str],
    pairwise_dict: dict[tuple[str, str], tuple[float, float]],
) -> dict[tuple[str, str], tuple[float, float]]:
    """
    Restricts the full pairwise dictionary to a subset of candidates. The pairwise dictionary is a
    dictionary whose keys are candidate pairs (A,B) and whose values are lists [a,b]
    where 'a' denotes the number of times A beats B head to head, and 'b' is the reverse.

    Args:
        cands (list[str] | tuple[str] | set[str]): Candidate subset to restrict to.
        pairwise_dict (dict[tuple[str, str], tuple[float, float]): Full pairwise comparison
            dictionary.

    Returns:
        dict[tuple[str, str], tuple[float, float]]: Pairwise dict restricted to the provided
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
    for tup in combinations(cand_subset, 2):
        if tup in pairwise_dict:
            new_pairwise_dict[tup] = pairwise_dict[tup]
        rev_tup = (tup[1], tup[0])
        if rev_tup in pairwise_dict:
            new_pairwise_dict[rev_tup] = pairwise_dict[rev_tup]

    return new_pairwise_dict


class PairwiseComparisonGraph(nx.DiGraph):
    """
    Class to construct the pairwise comparison graph where nodes are candidates
    and edges are pairwise preferences.

    Args:
        profile (RankProfile): ``RankProfile`` to construct graph from.
        sort_candidate_pairs (bool): If True, candidate pairs in the pairwise
            comparison dictionary will be sorted lexicographically. Defaults to True.

    Attributes:
        profile (RankProfile): ``RankProfile`` to construct graph from.
        candidates (list): List of candidates.
        pairwise_dict (dict[tuple[str, str], tuple[float, float]]): Dictionary constructed from
            ``pairwise_dict``. The pairwise dictionary is a
            dictionary whose keys are candidate pairs (A,B) and whose values are lists [a,b]
            where 'a' denotes the number of times A beats B head to head, and 'b' is
            the number of times B beats A head to head.
        pairwise_graph (networkx.DiGraph): Underlying graph.
    """

    def __init__(self, profile: RankProfile, *, sort_candidate_pairs: bool = True):
        self.profile = profile
        self.pairwise_dict = pairwise_dict(
            profile, sort_candidate_pairs=sort_candidate_pairs
        )
        self.pairwise_graph: nx.DiGraph = self.__build_graph()

    def __build_graph(self) -> nx.DiGraph:
        """
        Constructs the pairwise comparison graph from the pairwise comparison dictionary.
        """
        G: nx.DiGraph = nx.DiGraph()
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
        return get_dominating_tiers_digraph(self.pairwise_graph)

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
        G: nx.DiGraph = self.pairwise_graph

        known_candidate_list = list(G.nodes)
        if candidate_list is None:
            candidate_list = known_candidate_list

        assert set(candidate_list).issubset(set(G.nodes)), (
            f"Invalid candidates found: {set(candidate_list) - set(G.nodes)} does not appear as "
            f"a subset of known candidates {set(G.nodes)}"
        )

        G = nx.DiGraph(G.subgraph(candidate_list))

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
