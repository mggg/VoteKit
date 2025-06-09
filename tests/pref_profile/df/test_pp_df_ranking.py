from votekit.ballot import Ballot
from votekit.pref_profile import PreferenceProfile
import pandas as pd
from fractions import Fraction

ballots_rankings = [
    Ballot(ranking=({"A"}, {"B"}, {"C"}), weight=2),
    Ballot(ranking=({"A", "B"}, frozenset(), {"D"}), voter_set={"Chris"}),
    Ballot(),
    Ballot(weight=0),
]


def test_pp_df_rankings():
    pp = PreferenceProfile(ballots=ballots_rankings)
    data = {
        "Ranking_1": [
            frozenset({"A"}),
            frozenset({"A", "B"}),
            frozenset(),
            frozenset(),
        ],
        "Ranking_2": [frozenset({"B"}), frozenset(), frozenset(), frozenset()],
        "Ranking_3": [frozenset({"C"}), frozenset({"D"}), frozenset(), frozenset()],
        "Voter Set": [set(), {"Chris"}, set(), set()],
        "Weight": [Fraction(2), Fraction(1), Fraction(1), Fraction(0)],
    }
    true_df = pd.DataFrame(data)
    true_df.index.name = "Ballot Index"
    assert pp.df.equals(true_df)


def test_pp_df_rankings_args():
    pp = PreferenceProfile(
        ballots=ballots_rankings,
        contains_rankings=True,
        contains_scores=False,
        candidates=["A", "B", "C", "D", "E"],
        max_ranking_length=4,
    )
    data = {
        "Ranking_1": [
            frozenset({"A"}),
            frozenset({"A", "B"}),
            frozenset(),
            frozenset(),
        ],
        "Ranking_2": [frozenset({"B"}), frozenset(), frozenset(), frozenset()],
        "Ranking_3": [frozenset({"C"}), frozenset({"D"}), frozenset(), frozenset()],
        "Ranking_4": [frozenset(), frozenset(), frozenset(), frozenset()],
        "Voter Set": [set(), {"Chris"}, set(), set()],
        "Weight": [Fraction(2), Fraction(1), Fraction(1), Fraction(0)],
    }
    true_df = pd.DataFrame(data)
    true_df.index.name = "Ballot Index"
    assert pp.df.equals(true_df)
