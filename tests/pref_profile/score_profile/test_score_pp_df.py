from votekit.ballot import ScoreBallot
from votekit.pref_profile import ScoreProfile
import pandas as pd
import numpy as np

ballots_scores = [
    ScoreBallot(
        weight=2,
        scores={
            "A": 1,
            "B": 2,
        },
    ),
    ScoreBallot(scores={"D": 2, "E": 1}, voter_set={"Chris"}),
    ScoreBallot(),
    ScoreBallot(weight=0),
]


def test_pp_df_scores():
    pp = ScoreProfile(ballots=ballots_scores)
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
    }
    true_df = pd.DataFrame(data)
    true_df.index.name = "Ballot Index"
    assert pp.df.equals(true_df)


def test_pp_df_scores_args():
    pp = ScoreProfile(
        ballots=ballots_scores,
        candidates=["A", "B", "C", "D", "E"],
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
        "Voter Set": [set(), {"Chris"}, set(), set()],
        "Weight": [2.0, 1.0, 1.0, 0.0],
    }
    true_df = pd.DataFrame(data)
    true_df.index.name = "Ballot Index"
    assert pp.df.equals(true_df)
