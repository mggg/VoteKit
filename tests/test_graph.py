from fractions import Fraction
import networkx as nx
import pytest

from votekit.ballot import Ballot
from votekit.graphs.ballot_graph import BallotGraph
from votekit.pref_profile import PreferenceProfile


three_cand = PreferenceProfile(
    ballots=[
        Ballot(ranking=[{"A"}, {"B"}, {"C"}], weight=Fraction(4)),
        Ballot(ranking=[{"C"}, {"B"}, {"A"}], weight=Fraction(3)),
        Ballot(ranking=[{"C"}, {"B"}], weight=Fraction(2)),
    ]
)


def sum_weights(graph):
    total = 0
    for node in graph.nodes:
        if "weight" in graph.nodes[node]:
            total += graph.nodes[node]["weight"]

    return total


def test_from_n_cands():
    three = BallotGraph(3)
    num_nodes = len(three.graph.nodes)

    assert num_nodes == 9


def test_add_profile():
    three = BallotGraph(3)
    wprofile = three.from_profile(three_cand)
    assert isinstance(wprofile, nx.Graph)
    num_ballots = sum_weights(wprofile)
    assert num_ballots == 9


def test_allow_partial():
    three = BallotGraph(3, allow_partial=False)
    assert len(three.graph.nodes) == 6


def test_graph_labels():
    test = BallotGraph(three_cand)
    labels = test.label_cands(three_cand.get_candidates())
    assert len(labels) == 9


def test_k_neighbors_no_weights():
    test = BallotGraph(3)
    with pytest.raises(TypeError):
        test.k_heaviest_neighborhoods(k=2, int=2)


def test_k_neighborhoods():
    test = BallotGraph(3)
    test.node_weights = {
        (1,): 4,  # fix weights
        (1, 2, 3): 3,
        (1, 3, 2): 1,
        (2,): 0,
        (2, 3, 1): 2,
        (2, 1, 3): 3,
        (3,): 3,
        (3, 1, 2): 1,
        (3, 2, 1): 2,
    }

    centers = [(1, 3, 2), (2, 3, 1)]
    k_neighbs = test.k_heaviest_neighborhoods(k=2, radius=2)
    assert centers == list(k_neighbs.keys())


def test_labels_no_cands():
    test = BallotGraph(3)
    with pytest.raises(ValueError):
        test.draw(labels=True)


def test_fix_short_ballots_strings():
    graph = BallotGraph(2)
    cands = ["A", "B", "C"]
    ballot = ["A", "B"]
    fixed = graph.fix_short_ballot(ballot, cands)
    assert fixed == cands
