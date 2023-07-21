import matplotlib.pyplot as plt
import networkx as nx
from profile import PreferenceProfile
from distinctipy import get_colors
from functools import cache
from typing import Callable


@cache
def build_graph(n: int):
    """Builds graph of all possible ballots"""
    Gc = nx.Graph()

    # base cases

    if n == 1:
        Gc.add_nodes_from([(1,)])

    elif n == 2:
        Gc.add_nodes_from([(1, 2), (2, 1)])
        Gc.add_edges_from([((1, 2), (2, 1))])

    elif n > 2:

        G_prev = build_graph(n - 1)
        for i in range(1, n + 1):
            # add the node for the bullet vote i
            Gc.add_node(tuple([i]))

            # make the subgraph for the ballots where i is ranked first
            G_corner = relabel(G_prev, i, n)

            # add the components from that graph to the larger graph
            Gc.add_nodes_from(G_corner.nodes)
            Gc.add_edges_from(G_corner.edges)

            # connect the bullet vote node to the appropriate verticies
            if n == 3:
                Gc.add_edges_from([(k, tuple([i])) for k in G_corner.nodes])
            else:
                Gc.add_edges_from(
                    [(k, tuple([i])) for k in G_corner.nodes if len(k) == 2]
                )

        nodes = Gc.nodes

        # add the additional edges corresponding to swapping the order of the
        # first two candidates
        new_edges = []
        for k in nodes:
            if len(k) == 2:
                new_edges.append(((k[0], k[1]), (k[1], k[0])))
            elif len(k) > 2:
                bal_as_list = list(k)
                a = bal_as_list[0]
                b = bal_as_list[1]
                new_edges.append(
                    (
                        tuple([a] + [b] + bal_as_list[2:]),
                        tuple([b] + [a] + bal_as_list[2:]),
                    )
                )

        Gc.add_edges_from(new_edges)

    return Gc


def relabel(gr: nx.Graph, new_label: int, num_cands: int):
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


class BallotGraph:
    """Class to build graphs for elections with possible incomplete ballots"""

    def __init__(self, profile: PreferenceProfile):
        self.num_cands = len(profile.get_candidates())
        self.profile = profile
        self.ballot_dict = profile.to_dict()
        Gc = build_graph(self.num_cands)
        all_ballots = Gc.nodes
        di = {}
        for ballot in all_ballots:
            di[ballot] = 0

        self.ballot_dict = di | self.ballot_dict
        self.clean()
        self.num_voters = sum(self.ballot_dict.values())

    def clean(self):
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
                        self.ballot_dict[ballot + tuple([i])] += di[ballot]
                        self.ballot_dict.pop(ballot)
                        break

    def visualize(self, neighborhoods: dict = None):  ##
        """visualize the whole election or select neighborhoods in the election."""
        # TODO: change this so that neighborhoods can have any neighborhood
        # not just heavy balls, also there's something wrong with the shades
        Gc = build_graph(self.num_cands)
        WHITE = (1, 1, 1)
        BLACK = (0, 0, 0)
        node_cols = []

        if not neighborhoods:
            k = self.num_cands
        else:
            k = len(neighborhoods)

        cols = get_colors(k, [WHITE, BLACK])

        self.clean()
        for bal in Gc.nodes:
            i = -1
            color = WHITE
            weight = self.num_voters

            if not neighborhoods:
                if self.ballot_dict[bal] != 0:
                    i = (self.profile.get_candidates()).index(bal[0])

            else:
                for c in neighborhoods.keys():
                    if bal in (neighborhoods[c])[0].nodes:
                        weight = (neighborhoods[c])[1]
                        i = (list(neighborhoods.keys())).index(c)
                        break
            if i != -1:
                color = tuple(
                    [(1 - self.ballot_dict[bal] / weight) * x for x in cols[i]]
                )

            node_cols.append(color)

        nx.draw_networkx(Gc, with_labels=True, node_color=node_cols)
        plt.show()

    def distance_between_subsets(self, A: nx.Graph, B: nx.Graph):
        """Returns distance between A,B"""
        Gc = build_graph(self.num_cands)
        shortest_paths = []
        for a in A.nodes:
            for b in B.nodes:
                shortest_paths.append(nx.shortest_path_length(Gc, a, b))
        return min(shortest_paths)

    @staticmethod
    def show_all_ballot_types(n: int):
        """Draws graph with all possible ballot types, nodes all have same color"""
        Gc = build_graph(n)
        nx.draw(Gc, with_labels=True)
        plt.show()

    def subgraph_neighborhood(self, center, radius: int = 2):
        """Returns a ball around center of given radius in the graph of all ballots"""
        return nx.ego_graph(build_graph(self.num_cands), center, radius)

    def k_heaviest_neighborhoods(self, k: int = 2, radius: int = 2):
        """Returns dict of k ball neighborhoods of
        given radius with their centers and weights
        """
        cast_ballots = set(
            [x for x in self.ballot_dict.keys() if self.ballot_dict[x] > 0]
        )  ##has

        max_balls = {}

        for i in range(k):
            weight = 0
            if len(cast_ballots) == 0:
                break
            for center in cast_ballots:
                tmp = 0
                ball = self.subgraph_neighborhood(center, radius)
                relevant = cast_ballots.intersection(
                    set(ball.nodes)
                )  ##cast ballots inside the ball
                for node in relevant:
                    tmp += self.ballot_dict[node]

                if tmp > weight:
                    weight = tmp
                    max_center = center
                    max_ball = ball

            not_cast_in_max_ball = set(max_ball.nodes).difference(cast_ballots)
            max_ball.remove_nodes_from(not_cast_in_max_ball)
            max_balls[max_center] = (max_ball, weight)

            cast_ballots = cast_ballots.difference(set(max_ball.nodes))

        return max_balls

    def compare(self, new_pref: PreferenceProfile, dist_type: Callable = None):
        """compares the ballots of current and new profile"""
        raise NotImplementedError("Not yet built")

    def compare_rcv_results(self, new_pref: PreferenceProfile):
        """compares election results of current and new profile"""
        raise NotImplementedError("Not yet built")
