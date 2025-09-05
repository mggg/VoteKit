from votekit.ballot import Ballot
from votekit.pref_profile import PreferenceProfile, profile_df_head
import pandas as pd
import numpy as np


def test_pp_df_head_rankings():
    ballots_rankings = [
        Ballot(ranking=({"A", "B"}, frozenset(), {"D"}), voter_set={"Chris"}),
        Ballot(ranking=({"A"}, {"B"}, {"C"}), weight=2),
        Ballot(),
        Ballot(weight=0),
    ]
    pp = PreferenceProfile(ballots=ballots_rankings)
    data = {
        "Ranking_1": [
            frozenset({"A"}),
            frozenset({"A", "B"}),
            frozenset("~"),
            frozenset("~"),
        ],
        "Ranking_2": [frozenset({"B"}), frozenset(), frozenset("~"), frozenset("~")],
        "Ranking_3": [
            frozenset({"C"}),
            frozenset({"D"}),
            frozenset("~"),
            frozenset("~"),
        ],
        "Voter Set": [set(), {"Chris"}, set(), set()],
        "Weight": [2.0, 1.0, 1.0, 0.0],
        "Percent": [
            f"{.5:.1%}",
            f"{.25:.1%}",
            f"{.25:.1%}",
            f"{0:.1%}",
        ],
    }
    true_df = pd.DataFrame(data)
    true_df.index = [1, 0, 2, 3]
    true_df.index.name = "Ballot Index"
    true_df.loc["Total"] = [""] * 4 + [4, f"{1:.1%}"]

    assert profile_df_head(pp, n=4, percents=True, totals=True).equals(true_df)


def test_pp_df_head_scores():
    ballots_scores = [
        Ballot(scores={"D": 2, "E": 1}, voter_set={"Chris"}),
        Ballot(
            weight=2,
            scores={
                "A": 1,
                "B": 2,
            },
        ),
        Ballot(),
        Ballot(weight=0),
    ]
    pp = PreferenceProfile(ballots=ballots_scores)
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
        "D": [np.nan, 2, np.nan, np.nan],
        "E": [np.nan, 1, np.nan, np.nan],
        "Voter Set": [set(), {"Chris"}, set(), set()],
        "Weight": [2.0, 1.0, 1.0, 0.0],
        "Percent": [
            f"{.5:.2%}",
            f"{.25:.2%}",
            f"{.25:.2%}",
            f"{0:.2%}",
        ],
    }
    true_df = pd.DataFrame(data)
    true_df.index = [1, 0, 2, 3]
    true_df.index.name = "Ballot Index"
    true_df.loc["Total"] = [""] * 5 + [4, f"{1:.2%}"]

    assert profile_df_head(pp, n=4, percents=True, totals=True, n_decimals=2).equals(
        true_df
    )


def test_pp_df_head_mixed():
    ballots_mixed = [
        Ballot(
            weight=2,
            scores={
                "A": 1,
                "B": 2,
            },
        ),
        Ballot(ranking=({"A", "B"}, frozenset(), {"D"}), voter_set={"Chris"}),
        Ballot(
            ranking=({"A"}, {"B"}, {"C"}),
            weight=2,
            scores={
                "A": 1,
                "B": 2,
            },
        ),
    ]
    pp = PreferenceProfile(ballots=ballots_mixed)

    data = {
        "A": [
            1,
            1,
            np.nan,
        ],
        "B": [
            2,
            2,
            np.nan,
        ],
        "C": [
            np.nan,
            np.nan,
            np.nan,
        ],
        "D": [
            np.nan,
            np.nan,
            np.nan,
        ],
        "Ranking_1": [
            frozenset("~"),
            frozenset({"A"}),
            frozenset({"A", "B"}),
        ],
        "Ranking_2": [
            frozenset("~"),
            frozenset({"B"}),
            frozenset(),
        ],
        "Ranking_3": [
            frozenset("~"),
            frozenset({"C"}),
            frozenset({"D"}),
        ],
        "Voter Set": [
            set(),
            set(),
            {"Chris"},
        ],
        "Weight": [
            2.0,
            2.0,
            1.0,
        ],
        "Percent": [
            f"{.4:.1%}",
            f"{.4:.1%}",
            f"{.2:.1%}",
        ],
    }
    true_df = pd.DataFrame(data)
    true_df.index = [0, 2, 1]
    true_df.index.name = "Ballot Index"
    true_df.loc["Total"] = [""] * 8 + [5, f"{1:.1%}"]
    assert profile_df_head(pp, n=4, percents=True, totals=True).equals(true_df)
