from fractions import Fraction
import networkx as nx
import networkx.algorithms.isomorphism as iso

from votekit.ballot import Ballot
from votekit.graphs.pairwise_comparison_graph import PairwiseComparisonGraph
from votekit.pref_profile import PreferenceProfile
from votekit.cvr_loaders import load_csv
from votekit.cleaning import remove_and_condense

from matplotlib.axes import Axes
import pytest

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
CSV_DIR = BASE_DIR / "data/csv/"
portland_profile = remove_and_condense(
    "skipped",
    load_csv(CSV_DIR / "Portland_D3_Condensed.csv", rank_cols=[1, 2, 3, 4, 5, 6]),
)


ballot_list = [
    Ballot(ranking=[{"A"}, {"C"}, {"D"}, {"B"}, {"E"}], weight=Fraction(10, 1)),
    Ballot(ranking=[{"A"}, {"B"}, {"C"}, {"D"}, {"E"}], weight=Fraction(10, 1)),
    Ballot(ranking=[{"D"}, {"A"}, {"E"}, {"B"}, {"C"}], weight=Fraction(10, 1)),
    Ballot(ranking=[{"A"}], weight=Fraction(24, 1)),
]
TEST_PROFILE = PreferenceProfile(ballots=ballot_list)

simple_ballot_list = [
    Ballot(ranking=[{"C"}, {"B"}, {"A"}], weight=Fraction(10, 1)),
    Ballot(ranking=[{"A"}, {"C"}, {"B"}], weight=Fraction(10, 1)),
    Ballot(ranking=[{"B"}, {"A"}, {"C"}], weight=Fraction(10, 1)),
]
SIMPLE_TEST_PROFILE = PreferenceProfile(ballots=simple_ballot_list)

EDGE_WEIGHT_0_PROFILE = PreferenceProfile(
    ballots=[
        Ballot(ranking=({"A"}, {"B"}, {"C"})),
        Ballot(ranking=({"A"}, {"C"}, {"B"})),
        Ballot(ranking=({"B"}, {"A"}, {"C"}), weight=2),
    ]
)


def test_constructor_sequence():
    pwcg = PairwiseComparisonGraph(TEST_PROFILE)
    assert isinstance(pwcg, PairwiseComparisonGraph)


def test_graph_build():
    pwcg = PairwiseComparisonGraph(SIMPLE_TEST_PROFILE)
    pwcg_graph = pwcg.pairwise_graph

    target_graph = nx.DiGraph()
    target_graph.add_nodes_from(["A", "B", "C"])
    target_graph.add_weighted_edges_from(
        [("A", "C", 10), ("B", "A", 10), ("C", "B", 10)]
    )

    edge_match = iso.numerical_edge_match("weight", 1)
    assert nx.is_isomorphic(pwcg_graph, target_graph, edge_match=edge_match)


def test_ties_or_beats():
    pwcg = PairwiseComparisonGraph(SIMPLE_TEST_PROFILE)

    assert pwcg.ties_or_beats("A") == {"B"}


def test_pwcg_dominating_tiers():
    pwcg = PairwiseComparisonGraph(TEST_PROFILE)
    dominating_tiers = pwcg.get_dominating_tiers()

    assert dominating_tiers == [{"A"}, {"B", "C", "D"}, {"E"}]

    pwcg = PairwiseComparisonGraph(EDGE_WEIGHT_0_PROFILE)
    dominating_tiers = pwcg.get_dominating_tiers()

    assert dominating_tiers == [{"A", "B"}, {"C"}]


def test_pwcg_has_condorcet_winner():
    pwcg = PairwiseComparisonGraph(TEST_PROFILE)
    assert pwcg.has_condorcet_winner()


def test_pwcg_get_condorcet_winner():
    pwcg = PairwiseComparisonGraph(TEST_PROFILE)
    assert pwcg.get_condorcet_winner() == "A"


def test_get_condorcet_cycles():
    pwcg = PairwiseComparisonGraph(portland_profile)

    cycles = pwcg.get_condorcet_cycles()

    assert cycles[0] == {"Cristal Azul Otero", "Kezia Wanner", "Philippe Knab"}


def test_has_condorcet_cycles():
    pwcg = PairwiseComparisonGraph(portland_profile)
    assert pwcg.has_condorcet_cycles()


def test_draw_pwcg():
    pwcg = PairwiseComparisonGraph(TEST_PROFILE)
    ax = pwcg.draw()

    assert isinstance(ax, Axes)

    pwcg = PairwiseComparisonGraph(TEST_PROFILE)
    ax = pwcg.draw(candidate_list=["A", "B", "C"])

    assert isinstance(ax, Axes)


def test_draw_pwcg_non_candidate_error():
    pwcg = PairwiseComparisonGraph(TEST_PROFILE)
    with pytest.raises(KeyError, match="'Chris'"):
        pwcg.draw(candidate_list=["A", "B", "Chris"])
