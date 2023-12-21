from fractions import Fraction
import itertools as it
import networkx as nx
import networkx.algorithms.isomorphism as iso

from votekit.ballot import Ballot
from votekit.graphs.pairwise_comparison_graph import PairwiseComparisonGraph
from votekit.pref_profile import PreferenceProfile


ballot_list = [
    Ballot(
        id=None, ranking=[{"A"}, {"C"}, {"D"}, {"B"}, {"E"}], weight=Fraction(10, 1)
    ),
    Ballot(
        id=None, ranking=[{"A"}, {"B"}, {"C"}, {"D"}, {"E"}], weight=Fraction(10, 1)
    ),
    Ballot(
        id=None, ranking=[{"D"}, {"A"}, {"E"}, {"B"}, {"C"}], weight=Fraction(10, 1)
    ),
    Ballot(id=None, ranking=[{"A"}], weight=Fraction(24, 1)),
]
TEST_PROFILE = PreferenceProfile(ballots=ballot_list)

simple_ballot_list = [
    Ballot(id=None, ranking=[{"C"}, {"B"}, {"A"}], weight=Fraction(10, 1)),
    Ballot(id=None, ranking=[{"A"}, {"C"}, {"B"}], weight=Fraction(10, 1)),
    Ballot(id=None, ranking=[{"B"}, {"A"}, {"C"}], weight=Fraction(10, 1)),
]
SIMPLE_TEST_PROFILE = PreferenceProfile(ballots=simple_ballot_list)


def test_constructor_sequence():
    pwcg = PairwiseComparisonGraph(TEST_PROFILE)
    assert isinstance(pwcg, PairwiseComparisonGraph)


def test_pwcg_ballot_fill():
    pwcg = PairwiseComparisonGraph(TEST_PROFILE)
    filled_ballots = pwcg.ballot_fill(TEST_PROFILE, 5)

    target_ballot_list = [
        Ballot(
            id=None, ranking=[{"A"}, {"C"}, {"D"}, {"B"}, {"E"}], weight=Fraction(10, 1)
        ),
        Ballot(
            id=None, ranking=[{"A"}, {"B"}, {"C"}, {"D"}, {"E"}], weight=Fraction(10, 1)
        ),
        Ballot(
            id=None, ranking=[{"D"}, {"A"}, {"E"}, {"B"}, {"C"}], weight=Fraction(10, 1)
        ),
    ]

    for perm in list(it.permutations([{"B"}, {"C"}, {"D"}, {"E"}])):
        perm_ballot = Ballot(id=None, ranking=[{"A"}] + list(perm), weight=Fraction(1))
        target_ballot_list.append(perm_ballot)
    target_filled_ballots = PreferenceProfile(ballots=target_ballot_list)

    assert filled_ballots == target_filled_ballots


def test_pwcg_dominating_tiers():
    pwcg = PairwiseComparisonGraph(TEST_PROFILE)
    dominating_tiers = pwcg.dominating_tiers()

    target_dominating_tiers = [{"A"}, {"B", "C", "D"}, {"E"}]

    assert dominating_tiers == target_dominating_tiers


def test_pwcg_has_condorcet_winner():
    pwcg = PairwiseComparisonGraph(TEST_PROFILE)
    pwcg_has_condorcet = pwcg.has_condorcet_winner()

    target_has_condorcet = True

    assert target_has_condorcet == pwcg_has_condorcet


def test_h2h_count():
    pwcg = PairwiseComparisonGraph(TEST_PROFILE)
    pwcg_h2h_count = pwcg.head2head_count("A", "D")

    target_h2h_count = Fraction(44)

    assert pwcg_h2h_count == target_h2h_count


def test_pairwise_dict():
    pwcg = PairwiseComparisonGraph(SIMPLE_TEST_PROFILE)
    pwcg_pairwise_dict = pwcg.compute_pairwise_dict()

    target_pairwise_dict = {
        ("A", "C"): Fraction(10),
        ("B", "A"): Fraction(10),
        ("C", "B"): Fraction(10),
    }

    assert pwcg_pairwise_dict == target_pairwise_dict


def test_graph_build():
    pwcg = PairwiseComparisonGraph(SIMPLE_TEST_PROFILE)
    pwcg_graph = pwcg.build_graph()

    target_graph = nx.DiGraph()
    target_graph.add_nodes_from(["A", "B", "C"])
    target_graph.add_weighted_edges_from(
        [("A", "C", 10), ("B", "A", 10), ("C", "B", 10)]
    )

    edge_match = iso.numerical_edge_match("weight", 1)
    assert nx.is_isomorphic(pwcg_graph, target_graph, edge_match=edge_match)
