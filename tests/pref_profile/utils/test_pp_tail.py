from votekit.ballot import Ballot
from votekit.pref_profile import PreferenceProfile, profile_df_tail
import pandas as pd
import numpy as np
from fractions import Fraction


def test_pp_df_tail_rankings():
    ballots_rankings = [
        Ballot(ranking=({"A", "B"}, frozenset(), {"D"}), voter_set={"Chris"}),
        Ballot(ranking=({"A"}, {"B"}, {"C"}), weight=2),
        Ballot(),
        Ballot(weight=0),
    ]
    pp = PreferenceProfile(ballots=ballots_rankings)
    data = {
        "Ranking_1": [frozenset({"A"}), frozenset({"A", "B"}), np.nan, np.nan],
        "Ranking_2": [frozenset({"B"}), frozenset(), np.nan, np.nan],
        "Ranking_3": [frozenset({"C"}), frozenset({"D"}), np.nan, np.nan],
        "Voter Set": [set(), {"Chris"}, set(), set()],
        "Weight": [Fraction(2), Fraction(1), Fraction(1), Fraction(0)],
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
        "Voter Set": [set(), {"Chris"}, set(), set()],
        "Weight": [Fraction(2), Fraction(1), Fraction(1), Fraction(0)],
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


def test_pp_df_tail_mixed():
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
            Fraction(1),
            Fraction(1),
            np.nan,
        ],
        "B": [
            Fraction(2),
            Fraction(2),
            np.nan,
        ],
        "Ranking_1": [
            np.nan,
            frozenset({"A"}),
            frozenset({"A", "B"}),
        ],
        "Ranking_2": [
            np.nan,
            frozenset({"B"}),
            frozenset(),
        ],
        "Ranking_3": [
            np.nan,
            frozenset({"C"}),
            frozenset({"D"}),
        ],
        "Voter Set": [
            set(),
            set(),
            {"Chris"},
        ],
        "Weight": [
            Fraction(2),
            Fraction(2),
            Fraction(1),
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
    true_df.loc["Total"] = [""] * 6 + [5, f"{1:.1%}"]
    assert profile_df_tail(pp, n=4, percents=True, totals=True).equals(true_df)
