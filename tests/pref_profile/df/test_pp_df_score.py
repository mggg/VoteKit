from votekit.ballot import Ballot
from votekit.pref_profile import PreferenceProfile
import pandas as pd
import numpy as np
from fractions import Fraction

ballots_scores = [
    Ballot(
        weight=2,
        scores={
            "A": 1,
            "B": 2,
        },
    ),
    Ballot(scores={"D": 2, "E": 1}, id="X29", voter_set={"Chris"}),
    Ballot(),
    Ballot(weight=0),
]


def test_pp_df_scoress():
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
        "ID": [np.nan, "X29", np.nan, np.nan],
        "Voter Set": [set(), {"Chris"}, set(), set()],
        "Weight": [Fraction(2), Fraction(1), Fraction(1), Fraction(0)],
    }
    true_df = pd.DataFrame(data)
    true_df.index.name = "Ballot Index"
    assert pp.df.equals(true_df)


def test_pp_df_rankings_args():
    pp = PreferenceProfile(
        ballots=ballots_scores,
        contains_rankings=False,
        contains_scores=True,
        contains_rankings_and_scores=False,
        candidates=["A", "B", "C", "D", "E"],
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
        "ID": [np.nan, "X29", np.nan, np.nan],
        "Voter Set": [set(), {"Chris"}, set(), set()],
        "Weight": [Fraction(2), Fraction(1), Fraction(1), Fraction(0)],
    }
    true_df = pd.DataFrame(data)
    true_df.index.name = "Ballot Index"
    assert pp.df.equals(true_df)
