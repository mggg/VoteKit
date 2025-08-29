from votekit.graphs.pairwise_comparison_graph import get_dominating_tiers_digraph
import networkx as nx


def add_cycle(mutated_graph, cycle_nodes):
    """
    Helper function to add a cycle to the graph.
    """
    for i in range(len(cycle_nodes)):
        mutated_graph.add_edge(cycle_nodes[i], cycle_nodes[(i + 1) % len(cycle_nodes)])

    return mutated_graph


# A
# |
# v
# B
# |
# v
# C
def test_simple_dominating_tiers():
    graph = nx.DiGraph()
    graph.add_nodes_from(["A", "B", "C"])
    graph.add_weighted_edges_from([("A", "B", 10), ("B", "C", 10)])
    assert get_dominating_tiers_digraph(graph) == [{"A"}, {"B"}, {"C"}]


# A   C
#  \ /
#   v
#   B
def test_multimember_top_tier():
    graph = nx.DiGraph()
    graph.add_edges_from([("A", "B"), ("C", "B")])
    assert get_dominating_tiers_digraph(graph) == [{"A", "C"}, {"B"}]


#    A
#  /   \
# v     v
# B     C
#   \ /
#    v
#    D
def test_multimember_middle_tiers():
    graph = nx.DiGraph()
    graph.add_edges_from([("A", "B"), ("B", "D"), ("A", "C"), ("C", "D")])
    assert get_dominating_tiers_digraph(graph) == [{"A"}, {"B", "C"}, {"D"}]


#   A
#  / \
# v   v
# B   C
def test_multimember_bottom_tiers():
    graph = nx.DiGraph()
    graph.add_edges_from([("A", "B"), ("A", "C")])
    assert get_dominating_tiers_digraph(graph) == [{"A"}, {"B", "C"}]


# ABC
#  |
#  v
#  D
def test_cycle_in_top_tier():
    graph = nx.DiGraph()
    graph = add_cycle(graph, ["A", "B", "C"])
    graph.add_edges_from([("C", "D")])
    assert get_dominating_tiers_digraph(graph) == [{"A", "B", "C"}, {"D"}]


#  A
#  |
#  v
# BCD
#  |
#  v
#  E
def test_cycle_in_middle_tier():
    graph = nx.DiGraph()
    graph.add_edges_from([("A", "B")])
    graph = add_cycle(graph, ["B", "C", "D"])
    graph.add_edges_from([("D", "E")])
    assert get_dominating_tiers_digraph(graph) == [{"A"}, {"B", "C", "D"}, {"E"}]


#  A
#  |
#  v
# BCD
def test_cycle_in_bottom_tier():
    graph = nx.DiGraph()
    graph.add_edges_from([("A", "B")])
    graph = add_cycle(graph, ["B", "C", "D"])
    assert get_dominating_tiers_digraph(graph) == [{"A"}, {"B", "C", "D"}]


# ABC  D
#   \ /
#    v
#    E
def test_cycles_and_isolated_nodes_in_top_tier():
    graph = nx.DiGraph()
    graph = add_cycle(graph, ["A", "B", "C"])
    graph.add_edges_from([("C", "E"), ("D", "E")])
    assert get_dominating_tiers_digraph(graph) == [{"A", "B", "C", "D"}, {"E"}]


#     A
#   /   \
#  v     v
# BCD    E
#   \   /
#     v
#     F
def test_cycles_and_isolated_nodes_in_middle_tier():
    graph = nx.DiGraph()
    graph.add_edges_from([("A", "B"), ("A", "E")])
    graph = add_cycle(graph, ["B", "C", "D"])
    graph.add_edges_from([("D", "F"), ("E", "F")])
    assert get_dominating_tiers_digraph(graph) == [{"A"}, {"B", "C", "D", "E"}, {"F"}]


#     A
#   /   \
#  v     v
# BDE    C
def test_cycles_and_isolated_nodes_in_bottom_tier():
    graph = nx.DiGraph()
    graph.add_edges_from([("A", "B"), ("A", "C")])
    graph = add_cycle(graph, ["B", "D", "E"])
    assert get_dominating_tiers_digraph(graph) == [{"A"}, {"B", "C", "D", "E"}]


#     A
#   /   \
#  v     v
# BCD    E
#  |     |
#  |     v
#  |    FGH
#   \   /
#     v
#     I
def test_diamond_graph_with_uneven_legs():
    graph = nx.DiGraph()

    graph.add_edges_from([("A", "B"), ("A", "C")])
    graph = add_cycle(graph, ["B", "D", "E"])
    graph.add_edges_from([("C", "F")])
    graph = add_cycle(graph, ["F", "G", "H"])
    graph.add_edges_from([("E", "I"), ("H", "I")])
    assert get_dominating_tiers_digraph(graph) == [
        {"A"},
        {"B", "C", "D", "E", "F", "G", "H"},
        {"I"},
    ]
