import numpy as np
import pandas as pd

from votekit.ballot import ScoreBallot
from votekit.pref_profile import ScoreProfile


def test_where_matches_unscored_to_unscored_and_not_to_zero():
    profile = ScoreProfile(
        ballots=[ScoreBallot(scores={"A": 1}), ScoreBallot(scores={"A": 1, "B": 0})],
        candidates=["A", "B"],
    )

    assert profile.where(profile.to_df().iloc[0]).tolist() == [0]
    assert profile.where(profile.to_df().iloc[1]).tolist() == [1]


def test_where_returns_the_ballot_index_label():
    df = pd.DataFrame(
        {"A": [1.0, 2.0], "B": [np.nan, 3.0], "Voter Set": [set(), set()], "Weight": [1.0, 1.0]},
        index=pd.Index([10, 20], name="Ballot Index"),
    )
    profile = ScoreProfile(df=df, candidates=["A", "B"])

    assert profile.where(pd.Series({"A": 2.0, "B": 3.0})).tolist() == [20]


def test_where_finds_no_ballot_for_absent_scores():
    profile = ScoreProfile(ballots=[ScoreBallot(scores={"A": 1})], candidates=["A"])

    assert profile.where(pd.Series({"A": 5.0})).tolist() == []
