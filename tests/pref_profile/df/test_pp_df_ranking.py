from votekit.ballot import Ballot
from votekit.pref_profile import PreferenceProfile
import pandas as pd
import numpy as np
from fractions import Fraction

ballots_rankings = [
    Ballot(ranking=({"A"}, {"B"}, {"C"}), weight=2),
    Ballot(ranking=({"A", "B"}, frozenset(), {"D"}), id="X29", voter_set={"Chris"}),
    Ballot(),
    Ballot(weight=0),
]


def test_pp_df_rankings():
    pp = PreferenceProfile(ballots=ballots_rankings)
    data = {
        "Ranking_1": [frozenset({"A"}), frozenset({"A", "B"}), np.nan, np.nan],
        "Ranking_2": [frozenset({"B"}), frozenset(), np.nan, np.nan],
        "Ranking_3": [frozenset({"C"}), frozenset({"D"}), np.nan, np.nan],
        "Weight": [Fraction(2), Fraction(1), Fraction(1), Fraction(0)],
        "ID": [np.nan, "X29", np.nan, np.nan],
        "Voter Set": [set(), {"Chris"}, set(), set()],
    }
    true_df = pd.DataFrame(data)
    true_df.index.name = "Ballot Index"
    assert pp.df.equals(true_df)


def test_pp_df_rankings_args():
    pp = PreferenceProfile(
        ballots=ballots_rankings,
        contains_rankings=True,
        contains_scores=False,
        contains_rankings_and_scores=False,
        candidates=["A", "B", "C", "D", "E"],
        max_ballot_length=4,
    )
    data = {
        "Ranking_1": [frozenset({"A"}), frozenset({"A", "B"}), np.nan, np.nan],
        "Ranking_2": [frozenset({"B"}), frozenset(), np.nan, np.nan],
        "Ranking_3": [frozenset({"C"}), frozenset({"D"}), np.nan, np.nan],
        "Ranking_4": [np.nan, np.nan, np.nan, np.nan],
        "Weight": [Fraction(2), Fraction(1), Fraction(1), Fraction(0)],
        "ID": [np.nan, "X29", np.nan, np.nan],
        "Voter Set": [set(), {"Chris"}, set(), set()],
    }
    true_df = pd.DataFrame(data)
    true_df.index.name = "Ballot Index"
    assert pp.df.equals(true_df)
