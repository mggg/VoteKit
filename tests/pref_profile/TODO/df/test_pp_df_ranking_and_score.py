from votekit.ballot import Ballot
from votekit.pref_profile import PreferenceProfile
import pandas as pd
import numpy as np

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
        voter_set={"Chris"},
    ),
    Ballot(),
    Ballot(weight=0),
]


def test_pp_df_ranking_and_score():
    pp = PreferenceProfile(ballots=ballots_rankings_and_scores)

    data = {
        "A": [
            1,
            np.nan,
            np.nan,
            np.nan,
        ],
        "B": [
            2,
            np.nan,
            np.nan,
            np.nan,
        ],
        "C": [np.nan, np.nan, np.nan, np.nan],
        "D": [np.nan, 2, np.nan, np.nan],
        "E": [np.nan, 1, np.nan, np.nan],
        "Ranking_1": [
            frozenset({"A"}),
            frozenset({"A", "B"}),
            frozenset("~"),
            frozenset("~"),
        ],
        "Ranking_2": [
            frozenset({"B"}),
            frozenset({"D"}),
            frozenset("~"),
            frozenset("~"),
        ],
        "Ranking_3": [frozenset({"C"}), frozenset("~"), frozenset("~"), frozenset("~")],
        "Voter Set": [set(), {"Chris"}, set(), set()],
        "Weight": [2.0, 1.0, 1.0, 0.0],
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
            1,
            np.nan,
            np.nan,
            np.nan,
        ],
        "B": [
            2,
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
        "D": [np.nan, 2, np.nan, np.nan],
        "E": [np.nan, 1, np.nan, np.nan],
        "Ranking_1": [
            frozenset({"A"}),
            frozenset({"A", "B"}),
            frozenset("~"),
            frozenset("~"),
        ],
        "Ranking_2": [
            frozenset({"B"}),
            frozenset({"D"}),
            frozenset("~"),
            frozenset("~"),
        ],
        "Ranking_3": [frozenset({"C"}), frozenset("~"), frozenset("~"), frozenset("~")],
        "Voter Set": [set(), {"Chris"}, set(), set()],
        "Weight": [2.0, 1.0, 1.0, 0.0],
    }
    true_df = pd.DataFrame(data)
    true_df.index.name = "Ballot Index"
    assert pp.df.equals(true_df)
