from .base_graph import Graph
from ..pref_profile import PreferenceProfile
from ..utils import COLOR_LIST
from typing import Optional, Union
import networkx as nx  # type: ignore
from functools import cache


class BallotGraph(Graph):
    """
    Class to build ballot graphs.

    **Attributes**

    `source`
    :   data to create graph from, either PreferenceProfile object, number of
            candidates, or list of candidates

    `allow_partial`
    :   if True, builds graph using all possible ballots,
        if False, only uses total linear ordered ballots
        if building from a PreferenceProfile, defaults to True

    **Methods**
    """

    def __init__(
        self,
        source: Union[PreferenceProfile, int, list],
        allow_partial: Optional[bool] = True,
    ):
        super().__init__()

        self.profile = None
        self.candidates = None
        self.allow_partial = allow_partial

        if isinstance(source, int):
            self.num_cands = source
            self.graph = self.build_graph(source)

        if isinstance(source, list):
            self.num_cands = len(source)
            self.graph = self.build_graph(len(source))
            self.candidates = source

        if isinstance(source, PreferenceProfile):
            self.profile = source
            self.num_voters = source.num_ballots()
            self.num_cands = len(source.get_candidates())
            self.allow_partial = True
            if not self.graph:
                self.graph = self.build_graph(len(source.get_candidates()))
            self.graph = self.from_profile(source)

        self.num_voters = sum(self.node_weights.values())

        # if no partial ballots allowed, create induced subgraph
        if not self.allow_partial:
            total_ballots = [n for n in self.graph.nodes() if len(n) == self.num_cands]
            self.graph = self.graph.subgraph(total_ballots)

        if not self.node_weights:
            self.node_weights = {ballot: 0 for ballot in self.graph.nodes}

    def _relabel(self, gr: nx.Graph, new_label: int, num_cands: int) -> nx.Graph:
        """
        Relabels nodes in gr based on new_label
        """
        node_map = {}
        graph_nodes = list(gr.nodes)

        for k in graph_nodes:
            # add the value of new_label to every entry in every ballot
            tmp = [new_label + y for y in k]

            # reduce everything mod new_label
            for i in range(len(tmp)):
                if tmp[i] > num_cands:
                    tmp[i] = tmp[i] - num_cands
            node_map[k] = tuple([new_label] + tmp)

        return nx.relabel_nodes(gr, node_map)

    def build_graph(self, n: int) -> nx.Graph:  # ask Gabe about optimizing?
        """
        Builds graph of all possible ballots given a number of candiates

        Args:
            n: number of candidates per an election
        """
        Gc = nx.Graph()
        # base cases
        if n == 1:
            Gc.add_nodes_from([(1)], weight=0, cast=False)

        elif n == 2:
            Gc.add_nodes_from([(1, 2), (2, 1)], weight=0, cast=False)
            Gc.add_edges_from([((1, 2), (2, 1))])

        elif n > 2:
            G_prev = self.build_graph(n - 1)
            for i in range(1, n + 1):
                # add the node for the bullet vote i
                Gc.add_node((i,), weight=0, cast=False)

                # make the subgraph for the ballots where i is ranked first
                G_corner = self._relabel(G_prev, i, n)

                # add the components from that graph to the larger graph
                Gc.add_nodes_from(G_corner.nodes, weight=0, cast=False)
                Gc.add_edges_from(G_corner.edges)

                # connect the bullet vote node to the appropriate vertices
                if n == 3:
                    Gc.add_edges_from([(k, (i,)) for k in G_corner.nodes])
                else:
                    Gc.add_edges_from(
                        [(k, (i,)) for k in G_corner.nodes if len(k) == 2]
                    )

            nodes = Gc.nodes

            new_edges = [
                (bal, (bal[1], bal[0]) + bal[2:]) for bal in nodes if len(bal) >= 2
            ]
            Gc.add_edges_from(new_edges)

        return Gc

    def from_profile(
        self, profile: PreferenceProfile
    ) -> nx.Graph:
        """
        Updates existing graph based on cast ballots from a PreferenceProfile,
        or creates graph based on PreferenceProfile

        Args:
            profile: PreferenceProfile assigned to graph


        Returns:
            Graph based on PreferenceProfile, 'cast' node attribute indicates
                    ballots cast in PreferenceProfile
        """
        if not self.profile:
            self.profile = profile

        if not self.num_voters:
            self.num_voters = profile.num_ballots()

        self.candidates = profile.get_candidates()
        ballots = profile.get_ballots()
        self.cand_num = self._number_cands(tuple(self.candidates))
        self.node_weights = {ballot: 0 for ballot in self.graph.nodes}

        for ballot in ballots:
            ballot_node = []
            for position in ballot.ranking:
                if len(position) > 1:
                    raise ValueError(
                        "ballots must be cleaned to resolve ties"
                    )  # still unsure about ties
                for cand in position:
                    ballot_node.append(self.cand_num[cand])
            if len(ballot_node) == len(self.candidates) - 1:
                ballot_node = self.fix_short_ballot(
                    ballot_node, list(self.cand_num.values())
                )

            if tuple(ballot_node) in self.graph.nodes:
                self.graph.nodes[tuple(ballot_node)]["weight"] += ballot.weight
                self.graph.nodes[tuple(ballot_node)]["cast"] = True
                self.node_weights[tuple(ballot_node)] += ballot.weight

        return self.graph

    def fix_short_ballot(self, ballot: list, candidates: list) -> list:
        """
        Appends short ballots of n-1 length to add to BallotGraph
        """
        missing = set(candidates).difference(set(ballot))

        return ballot + list(missing)

    def label_cands(self, candidates,
                    show_cast: Optional[bool] = False):
        """
        Assigns candidate labels to ballot graph for plotting

        Args:
            show_cast: if True, only gives labels for ballots cast in PrefProfile
        """

        candidate_numbers = self._number_cands(tuple(candidates))

        cand_dict = {value: key for key, value in candidate_numbers.items()}

        cand_labels = {}
        for node in self.graph.nodes:
            if (show_cast and self.graph.nodes[node]['cast']) or not show_cast:
                    ballot = []
                    for num in node:
                        ballot.append(cand_dict[num])
                    cand_labels[node] = tuple(ballot)

        return cand_labels

    @cache
    def _number_cands(self, cands: tuple) -> dict:
        """
        Assigns numerical marker to candidates
        """
        legend = {}
        for idx, cand in enumerate(cands):
            legend[cand] = idx + 1

        return legend

    def draw(self, neighborhoods: Optional[dict] = {},
             labels: Optional[bool] = False,
             show_cast: Optional[bool] = False):
        """
        Visualize the whole election or select neighborhoods in the election.

        Args:
            neighborhoods: Section of graph to draw
            labels: If True, labels nodes with candidate names
            show_cast: If True, show only nodes with "cast" attribute = True
                        If False, show all nodes
        """
        # TODO: change this so that neighborhoods can have any neighborhood
        # not just heavy balls, also there's something wrong with the shades
        Gc = self.graph
        GREY = (0.44, 0.5, 0.56)
        node_cols: list = []
        node_labels = None

        k = len(neighborhoods) if neighborhoods else self.num_cands
        if k > len(COLOR_LIST):
            if neighborhoods:
                raise ValueError("Number of neighborhoods exceeds colors for plotting")
            else:
                raise ValueError("Number of candidates exceeds colors for plotting")
        cols = COLOR_LIST[:k]

        # self._clean()
        if show_cast:
            ballots = [n for n, data in Gc.nodes(data=True) if data["cast"]]
        else:
            ballots = Gc.nodes

        for ballot in ballots:
            i = -1
            color: tuple = GREY

            if neighborhoods:
                for center, neighborhood in neighborhoods.items():
                    neighbors, _ = neighborhood
                    if ballot in neighbors:
                        i = (list(neighborhoods.keys())).index(center)
                        break

            elif self.node_weights[ballot] != 0 and self.profile:
                # print(ballot)
                i = (list(self.cand_num.values())).index(ballot[0])

            if "weight" in ballot:
                color = tuple(ballot.weight * x for x in cols[i])

            node_cols.append(color)

        if labels:
            if not self.candidates:
                raise ValueError("no candidate names assigned")
            if self.candidates:
                node_labels = self.label_cands(self.candidates, show_cast)
            elif self.profile:
                node_labels = self.label_cands(self.profile.get_candidates(), show_cast)

        if show_cast:
            subgraph = Gc.subgraph(ballots)
            nx.draw_networkx(subgraph, with_labels=True, node_color=node_cols, labels=node_labels)

        else:
            nx.draw_networkx(Gc, with_labels=True, node_color=node_cols, labels=node_labels)

    # what are these functions supposed to do?
    # def compare(self, new_pref: PreferenceProfile, dist_type: Callable):
    #     """compares the ballots of current and new profile"""
    #     raise NotImplementedError("Not yet built")

    # def compare_rcv_results(self, new_pref: PreferenceProfile):
    #     """compares election results of current and new profle"""
    #     raise NotImplementedError("Not yet built")
