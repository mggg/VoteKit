from abc import ABC, abstractmethod
import networkx as nx  # type: ignore
from typing import Any


class Graph(ABC):
    """
    Base class for graph models, contains general graph algorithmics
    applicable to any implementation
    """

    def __init__(self, graph: nx.Graph = None):
        self.graph = graph
        self.node_weights: dict = {}  # store node weights to avoid acessing Nx.graph

    @abstractmethod
    def build_graph(self, *args: Any, **kwargs: Any) -> nx.Graph:
        pass

    @abstractmethod
    def draw(self, *args: Any, **kwags: Any):
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
        if not self.node_weights or sum(self.node_weights.values()) == 0:
            raise TypeError("no weights assigned to graph")

        cast_ballots = {x for x in self.node_weights.keys() if self.node_weights[x] > 0}

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
                tmp = sum(self.node_weights[node] for node in relevant)
                if tmp > weight:
                    weight = tmp
                    max_center = center
                    max_ball = ball

            not_cast_in_max_ball = set(max_ball.nodes).difference(cast_ballots)
            max_ball.remove_nodes_from(not_cast_in_max_ball)
            max_balls[max_center] = (max_ball, weight)

            cast_ballots = cast_ballots.difference(set(max_ball.nodes))

        return max_balls
