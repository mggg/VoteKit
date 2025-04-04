from votekit.ballot import Ballot
from votekit.pref_profile import PreferenceProfile
import pandas as pd
import numpy as np
from fractions import Fraction

ballots_rankings_and_scores = [
    Ballot(
        ranking=({"A"}, {"B"}, {"C"}),
        weight=2,
        scores={
            "A": 1,
            "B": 2,
        },
    ),
    Ballot(
        ranking=({"A", "B"}, {"D"}),
        scores={"D": 2, "E": 1},
        id="X29",
        voter_set={"Chris"},
    ),
    Ballot(),
    Ballot(weight=0),
]


def test_pp_df_ranking_and_score():
    pp = PreferenceProfile(ballots=ballots_rankings_and_scores)

    data = {
        "A": [
            Fraction(1),
            np.nan,
            np.nan,
            np.nan,
        ],
        "B": [
            Fraction(2),
            np.nan,
            np.nan,
            np.nan,
        ],
        "D": [np.nan, Fraction(2), np.nan, np.nan],
        "E": [np.nan, Fraction(1), np.nan, np.nan],
        "Ranking_1": [frozenset({"A"}), frozenset({"A", "B"}), np.nan, np.nan],
        "Ranking_2": [frozenset({"B"}), frozenset({"D"}), np.nan, np.nan],
        "Ranking_3": [frozenset({"C"}), np.nan, np.nan, np.nan],
        "Weight": [Fraction(2), Fraction(1), Fraction(1), Fraction(0)],
        "ID": [np.nan, "X29", np.nan, np.nan],
        "Voter Set": [set(), {"Chris"}, set(), set()],
    }
    true_df = pd.DataFrame(data)
    true_df.index.name = "Ballot Index"
    assert pp.df.equals(true_df)


def test_pp_df_ranking_and_score_args():
    pp = PreferenceProfile(
        ballots=ballots_rankings_and_scores,
        contains_rankings=True,
        contains_scores=True,
        candidates=["A", "B", "C", "D", "E"],
        max_ranking_length=3,
    )
    data = {
        "A": [
            Fraction(1),
            np.nan,
            np.nan,
            np.nan,
        ],
        "B": [
            Fraction(2),
            np.nan,
            np.nan,
            np.nan,
        ],
        "C": [
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        ],
        "D": [np.nan, Fraction(2), np.nan, np.nan],
        "E": [np.nan, Fraction(1), np.nan, np.nan],
        "Ranking_1": [frozenset({"A"}), frozenset({"A", "B"}), np.nan, np.nan],
        "Ranking_2": [frozenset({"B"}), frozenset({"D"}), np.nan, np.nan],
        "Ranking_3": [frozenset({"C"}), np.nan, np.nan, np.nan],
        "Weight": [Fraction(2), Fraction(1), Fraction(1), Fraction(0)],
        "ID": [np.nan, "X29", np.nan, np.nan],
        "Voter Set": [set(), {"Chris"}, set(), set()],
    }
    true_df = pd.DataFrame(data)
    true_df.index.name = "Ballot Index"
    assert pp.df.equals(true_df)
