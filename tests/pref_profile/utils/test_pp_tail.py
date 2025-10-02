from votekit.ballot import RankBallot, ScoreBallot
from votekit.pref_profile import PreferenceProfile, profile_df_tail
import pandas as pd
import numpy as np


def test_pp_df_tail_rankings():
    ballots_rankings = [
        RankBallot(ranking=({"A", "B"}, frozenset(), {"D"}), voter_set={"Chris"}),
        RankBallot(ranking=({"A"}, {"B"}, {"C"}), weight=2),
        RankBallot(),
        RankBallot(weight=0),
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

    assert profile_df_tail(pp, n=4, percents=True, totals=True).equals(true_df)


def test_pp_df_tail_scores():
    ballots_scores = [
        ScoreBallot(scores={"D": 2, "E": 1}, voter_set={"Chris"}),
        ScoreBallot(
            weight=2,
            scores={
                "A": 1,
                "B": 2,
            },
        ),
        ScoreBallot(),
        ScoreBallot(weight=0),
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

    assert profile_df_tail(pp, n=4, percents=True, totals=True, n_decimals=2).equals(
        true_df
    )
