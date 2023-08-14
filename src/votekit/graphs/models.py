from ..profile import PreferenceProfile
from typing import Callable, Optional, Union, Any
from abc import ABC, abstractmethod
from distinctipy import get_colors  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import networkx as nx  # type: ignore


class Graph(ABC):
    """
    Base class for graph models, contains general graph algorithmics
    applicable to any implementation
    """

    def __init__(self, graph: nx.Graph = None):
        self.graph = graph
        self.ballot_dict: dict = {}

    @abstractmethod
    def build_graph(self, *args: Any, **kwargs: Any) -> nx.Graph:
        pass

    def distance_between_subsets(self, A: nx.Graph, B: nx.Graph):
        """Returns distance between A,B"""
        Gc = self.graph
        shortest_paths: list = []
        for a in A.nodes:
            shortest_paths.extend(nx.shortest_path_length(Gc, a, b) for b in B.nodes)
        return min(shortest_paths)

    def subgraph_neighborhood(self, center, radius: int = 2) -> nx.Graph:
        """Returns a ball around center of given radius in the graph of all ballots"""
        return nx.ego_graph(self.graph, center, radius)

    def k_heaviest_neighborhoods(self, k: int = 2, radius: int = 2):
        """Returns dict of k ball neighborhoods of
        given radius with their centers and weights
        """
        cast_ballots = {x for x in self.ballot_dict.keys() if self.ballot_dict[x] > 0}

        max_balls = {}

        for _ in range(k):
            weight = 0
            if len(cast_ballots) == 0:
                break
            for center in cast_ballots:
                ball = self.subgraph_neighborhood(center, radius)
                relevant = cast_ballots.intersection(
                    set(ball.nodes)
                )  ##cast ballots inside the ball
                tmp = sum(self.ballot_dict[node] for node in relevant)
                if tmp > weight:
                    weight = tmp
                    max_center = center
                    max_ball = ball

            not_cast_in_max_ball = set(max_ball.nodes).difference(cast_ballots)
            max_ball.remove_nodes_from(not_cast_in_max_ball)
            max_balls[max_center] = (max_ball, weight)

            cast_ballots = cast_ballots.difference(set(max_ball.nodes))

        return max_balls


class BallotGraph(Graph):
    """
    Class to build graphs for elections with possible incomplete ballots

    Inputs:
        source: (PreferenceProfile, number of candidates, or list of candiates),
                data to create graph from
        complete: (Optional[bool]) build complete graph or incomplete

    """

    def __init__(
        self,
        source: Union[PreferenceProfile, int, list],
        complete: Optional[bool] = True,
    ):
        if isinstance(source, int):
            self.graph = self.build_graph(source)
            self.num_cands = source

        if isinstance(source, PreferenceProfile):
            self.profile = source
            self.num_cands = len(source.get_candidates())
            self.ballot_dict: dict = source.to_dict()
            self.graph = self.from_profile(source, complete)

        if isinstance(source, list):
            self.num_cands = len(source)
            self.graph = self.build_graph(len(source))

        all_ballots = self.graph.nodes
        self.ballot_dict = {ballot: 0 for ballot in all_ballots}
        # self._clean()
        self.num_voters: int = sum(self.ballot_dict.values())

    def _clean(self):
        """deletes empty ballots, changes n-1 length ballots
        to n length ballots and updates counts
        """
        di = self.ballot_dict.copy()

        for ballot in di.keys():
            if len(ballot) == 0:
                self.ballot_dict.pop(ballot)
            elif len(ballot) == self.num_cands - 1:
                for i in self.profile.get_candidates():
                    if i not in ballot:
                        self.ballot_dict[ballot + (i,)] += di[ballot]
                        self.ballot_dict.pop(ballot)
                        break

    def _relabel(self, gr: nx.Graph, new_label: int, num_cands: int) -> nx.Graph:
        """Relabels nodes in gr based on new_label"""
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

    def build_graph(self, n: int) -> nx.Graph:
        """
        Builds graph of all possible ballots given a number of candiates
        """
        Gc = nx.Graph()
        # base cases
        if n == 1:
            Gc.add_nodes_from([(1)])

        elif n == 2:
            Gc.add_nodes_from([(1, 2), (2, 1)])
            Gc.add_edges_from([((1, 2), (2, 1))])

        elif n > 2:
            G_prev = self.build_graph(n - 1)
            for i in range(1, n + 1):
                # add the node for the bullet vote i
                Gc.add_node((i,))

                # make the subgraph for the ballots where i is ranked first
                G_corner = self._relabel(G_prev, i, n)

                # add the components from that graph to the larger graph
                Gc.add_nodes_from(G_corner.nodes)
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

    def from_profile(self, profile: PreferenceProfile, complete: Optional[bool] = True):
        """
        Updates existing graph based on cast ballots from a PreferenceProfile,
        or creates graph based on PreferenceProfile
        """
        if not self.profile:
            self.profile = profile

        if not self.graph:
            num_cands = len(profile.get_candidates())
            self.graph = self.build_graph(num_cands)

        cands = profile.get_candidates()
        ballots = profile.get_ballots()
        cand_num = self.number_cands(cands)

        for ballot in ballots:
            ballot_node = []
            for position in ballot.ranking:
                if len(position) > 1:
                    raise ValueError(
                        "ballots must be cleaned to resolve ties"
                    )  # still unsure about ties
                for cand in position:
                    ballot_node.append(cand_num[cand])
            if ballot_node in self.graph.nodes:
                self.graph.nodes[tuple(ballot_node)]["weight"] = ballot.weight
                self.graph.nodes[tuple(ballot_node)]["cast"] = True

        # removes uncast nodes from graph
        if not complete:
            for node in self.graph.nodes:
                if "cast" not in node:
                    self.graph.remove_node(node)

    @staticmethod
    def number_cands(cands: list) -> dict:
        """
        Assigns numerical marker to candidates
        """
        legend = {}
        for idx, cand in enumerate(cands):
            legend[cand] = idx

        return legend

    def visualize(self, neighborhoods: Optional[dict] = {}):  ##
        """visualize the whole election or select neighborhoods in the election."""
        # TODO: change this so that neighborhoods can have any neighborhood
        # not just heavy balls, also there's something wrong with the shades
        Gc = self.graph
        WHITE = (1, 1, 1)
        BLACK = (0, 0, 0)
        node_cols: list = []

        k = len(neighborhoods) if neighborhoods else self.num_cands
        cols = get_colors(
            k, [WHITE, BLACK], pastel_factor=0.7
        )  # redo the colors to match MGGG lab

        self._clean()
        for bal in Gc.nodes:
            i = -1
            color: tuple = WHITE
            weight = self.num_voters

            if neighborhoods:
                for c in neighborhoods.keys():
                    if bal in (neighborhoods[c])[0].nodes:
                        weight = (neighborhoods[c])[1]
                        i = (list(neighborhoods.keys())).index(c)
                        break
            elif self.ballot_dict[bal] != 0:
                i = (self.profile.get_candidates()).index(bal[0])

            if i != -1:
                color = tuple((1 - self.ballot_dict[bal] / weight) * x for x in cols[i])
            node_cols.append(color)

        nx.draw_networkx(Gc, with_labels=True, node_color=node_cols)
        plt.show()

    def show_all_ballot_types(self):
        """Draws graph with all possible ballot types, nodes all have same color"""
        Gc = self.graph
        nx.draw(Gc, with_labels=True)
        plt.show()

    # what are these functions supposed to do?
    def compare(self, new_pref: PreferenceProfile, dist_type: Callable):
        """compares the ballots of current and new profile"""
        raise NotImplementedError("Not yet built")

    def compare_rcv_results(self, new_pref: PreferenceProfile):
        """compares election results of current and new profle"""
        raise NotImplementedError("Not yet built")


## TODO


class PairwiseGraph(Graph):
    """
    Add what Brenda's been working on here
    """

    def build_graph(self) -> nx.Graph:
        pass
