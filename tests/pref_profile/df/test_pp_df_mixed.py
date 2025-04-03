from votekit.ballot import Ballot
from votekit.pref_profile import PreferenceProfile
import pandas as pd
import numpy as np
from fractions import Fraction

ballots_mixed = [
    Ballot(
        weight=2,
        scores={
            "A": 1,
            "B": 2,
        },
    ),
    Ballot(ranking=({"A", "B"}, frozenset(), {"D"}), id="X29", voter_set={"Chris"}),
    Ballot(
        ranking=({"A"}, {"B"}, {"C"}),
        weight=2,
        scores={
            "A": 1,
            "B": 2,
        },
    ),
]


def test_pp_df_mixed():
    pp = PreferenceProfile(ballots=ballots_mixed)

    data = {
        "A": [
            Fraction(1),
            np.nan,
            Fraction(1),
        ],
        "B": [
            Fraction(2),
            np.nan,
            Fraction(2),
        ],
        "Ranking_1": [
            np.nan,
            frozenset({"A", "B"}),
            frozenset({"A"}),
        ],
        "Ranking_2": [
            np.nan,
            frozenset(),
            frozenset({"B"}),
        ],
        "Ranking_3": [np.nan, frozenset({"D"}), frozenset({"C"})],
        "Weight": [
            Fraction(2),
            Fraction(1),
            Fraction(2),
        ],
        "ID": [
            np.nan,
            "X29",
            np.nan,
        ],
        "Voter Set": [
            set(),
            {"Chris"},
            set(),
        ],
    }
    true_df = pd.DataFrame(data)
    true_df.index.name = "Ballot Index"
    assert pp.df.equals(true_df)


def test_pp_df_mixed_args():
    pp = PreferenceProfile(
        ballots=ballots_mixed,
        contains_rankings=True,
        contains_scores=True,
        candidates=["A", "B", "C", "D", "E"],
        max_ballot_length=3,
    )
    data = {
        "A": [
            Fraction(1),
            np.nan,
            Fraction(1),
        ],
        "B": [
            Fraction(2),
            np.nan,
            Fraction(2),
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
        "E": [
            np.nan,
            np.nan,
            np.nan,
        ],
        "Ranking_1": [
            np.nan,
            frozenset({"A", "B"}),
            frozenset({"A"}),
        ],
        "Ranking_2": [
            np.nan,
            frozenset(),
            frozenset({"B"}),
        ],
        "Ranking_3": [np.nan, frozenset({"D"}), frozenset({"C"})],
        "Weight": [
            Fraction(2),
            Fraction(1),
            Fraction(2),
        ],
        "ID": [
            np.nan,
            "X29",
            np.nan,
        ],
        "Voter Set": [
            set(),
            {"Chris"},
            set(),
        ],
    }
    true_df = pd.DataFrame(data)
    true_df.index.name = "Ballot Index"
    assert pp.df.equals(true_df)
