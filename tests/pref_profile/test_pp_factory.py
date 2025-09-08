from votekit.pref_profile import (
    PreferenceProfile,
    RankProfile,
    ScoreProfile,
    ProfileError,
)
from votekit.ballot import RankBallot, ScoreBallot
import pytest
import pandas as pd


def test_pp_factory_to_rank_profile():
    pp = PreferenceProfile(ballots=[RankBallot(ranking=[{"Chris"}])])

    assert isinstance(pp, RankProfile)


def test_pp_factory_to_score_profile():
    pp = PreferenceProfile(ballots=[ScoreBallot(scores={"Chris": 1})])

    assert isinstance(pp, ScoreProfile)


def test_pp_factory_errors():
    with pytest.raises(ProfileError, match="Cannot pass a dataframe and a ballot list"):
        PreferenceProfile(df=pd.DataFrame({"Col": [1]}), ballots=[RankBallot()])

    with pytest.raises(
        ProfileError,
        match="Profile cannot contain RankBallots and ScoreBallots. ScoreBallots"
        r" appear at indices \[1\], RankBallots appear at indices"
        r" \[0\].",
    ):
        PreferenceProfile(ballots=[RankBallot(), ScoreBallot()])
