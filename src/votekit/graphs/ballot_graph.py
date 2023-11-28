from .base_graph import Graph
from ..pref_profile import PreferenceProfile
from typing import Optional, Union
import networkx as nx  # type: ignore
from functools import cache
from typing import Callable
import matplotlib.pyplot as plt

def all_nodes(graph, node):
    return True

class BallotGraph(Graph):
    """
    Class to build ballot graphs.

    **Attributes**

    `source`
    :   data to create graph from, either PreferenceProfile object, number of
            candidates, or list of candidates.

    `allow_partial`
    :   if True, builds graph using all possible ballots,
        if False, only uses total linear ordered ballots.
        If building from a PreferenceProfile, defaults to True.

    `fix_short`
    : if True, auto completes ballots of length n-1 to n.

    **Methods**
    """

    def __init__(
        self,
        source: Union[PreferenceProfile, int, list],
        allow_partial: Optional[bool] = True,
        fix_short: Optional[bool] = True
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
            self.graph = self.from_profile(source, fix_short = fix_short)

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
        Builds graph of all possible ballots given a number of candiates.

        Args:
            n: Number of candidates in an election.

        Returns:
            A networkx graph.
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
        self, profile: PreferenceProfile,
        fix_short: Optional[bool] = True
    ) -> nx.Graph:
        """
        Updates existing graph based on cast ballots from a PreferenceProfile,
        or creates graph based on PreferenceProfile.

        Args:
            profile: PreferenceProfile assigned to graph.


        Returns:
            Graph based on PreferenceProfile, 'cast' node attribute indicates
                    ballots cast in PreferenceProfile.
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
            if len(ballot_node) == len(self.candidates) - 1 and fix_short:
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
        Adds missing candidates to a short ballot.

        Args:
            ballot: A list of candidates on the ballot.
            candidates: A list of all candidates.

        Returns:
            A new list with the missing candidates added to the end of the ballot.

        """
        missing = set(candidates).difference(set(ballot))

        return ballot + list(missing)

    def label_cands(self, candidates,
                    to_display: Callable = all_nodes):
        """
        Assigns candidate labels to ballot graph for plotting.

        Args:
            candidates (list): A list of candidates. 
            to_display: A Boolean callable that takes in a graph and node, 
                        returns True if node should be displayed.
        """

        candidate_numbers = self._number_cands(tuple(candidates))

        cand_dict = {value: key for key, value in candidate_numbers.items()}

        cand_labels = {}
        for node in self.graph.nodes:
            if to_display(self.graph, node):
                ballot = []
                for num in node:
                    ballot.append(cand_dict[num])
                
                # label the ballot and give the number of votes
                cand_labels[node] = str(tuple(ballot))+": " + \
                                str(self.graph.nodes[node]["weight"])

        return cand_labels

    def label_weights(self, to_display: Callable = all_nodes):
        """
        Assigns weight labels to ballot graph for plotting.
        Only shows weight if non-zero.

        Args:
            to_display: A Boolean callable that takes in a graph and node, 
                        returns True if node should be displayed.
        """
        node_labels = {}
        for node in self.graph.nodes:
            if to_display(self.graph, node):
                # label the ballot and give the number of votes
                if self.graph.nodes[node]["weight"] >0:
                    node_labels[node] = str(node)+": " + \
                                str(self.graph.nodes[node]["weight"])
                else:
                    node_labels[node] = str(node)

        return node_labels

    @cache
    def _number_cands(self, cands: tuple) -> dict:
        """
        Assigns numerical marker to candidates
        """
        legend = {}
        for idx, cand in enumerate(cands):
            legend[cand] = idx + 1

        return legend

    def draw(self, to_display: Callable = all_nodes,
             neighborhoods: Optional[list[tuple]] = [],
             show_cast: Optional[bool] = False,
             labels: Optional[bool] = False):
        """
        Visualize the graph.

        Args:
            to_display: A boolean function that takes the graph and a node as input, 
                returns True if you want that node displayed. Defaults to showing all nodes.
            neighborhoods: A list of neighborhoods to display, given as tuple (node, radius).
                            (ex. (n,1) gives all nodes within one step of n).
            show_cast: If True, show only nodes with "cast" attribute = True.
                        If False, show all nodes.
            labels: If True, labels nodes with candidate names and vote totals.
        """
        

        def cast_nodes(graph, node):
            return graph.nodes[node]["cast"]
        
        def in_neighborhoods(graph, node):
            centers = [node for node, radius in neighborhoods]
            radii = [radius for node, radius in neighborhoods]

            distances = [nx.shortest_path_length(graph, node, x) for x in centers]

            return (True in [d <= r for d,r in zip(distances, radii)])
        
        if show_cast:
            to_display = cast_nodes

        if neighborhoods:
            to_display = in_neighborhoods

        ballots = [n for n in self.graph.nodes if to_display(self.graph, n)]


        if labels:
            if not self.candidates:
                raise ValueError("no candidate names assigned")
            node_labels = self.label_cands(self.candidates, to_display)

        else:
            node_labels = self.label_weights(to_display)
            
        subgraph = self.graph.subgraph(ballots)

        pos = nx.spring_layout(subgraph)
        nx.draw_networkx(subgraph, pos = pos, 
                         with_labels=True, labels=node_labels)

        # handles labels overlapping with margins
        x_values, y_values = zip(*pos.values())
        x_max, y_max = max(x_values), max(y_values)
        x_min, y_min = min(x_values), min(y_values)
        x_margin = (x_max - x_min) * 0.25
        y_margin = (y_max - y_min) * 0.25
        plt.xlim(x_min - x_margin, x_max + x_margin)
        plt.ylim(y_min - y_margin, y_max + y_margin)
        plt.show()

       



    # what are these functions supposed to do?
    # def compare(self, new_pref: PreferenceProfile, dist_type: Callable):
    #     """compares the ballots of current and new profile"""
    #     raise NotImplementedError("Not yet built")

    # def compare_rcv_results(self, new_pref: PreferenceProfile):
    #     """compares election results of current and new profle"""
    #     raise NotImplementedError("Not yet built")
