from urllib.error import URLError

import pytest

from votekit.ballot import ScoreBallot
from votekit.pref_profile import ScoreProfile


def test_pkl_bijection_scores(tmp_path):
    profile_1 = ScoreProfile(
        ballots=[
            ScoreBallot(scores={"A": 2, "B": 2}, weight=2),
            ScoreBallot(scores={"A": 2, "C": 2}, voter_set={"Chris"}),
            ScoreBallot(),
            ScoreBallot(weight=0),
        ],
        candidates=["A", "B", "C", "D"],
    )

    out = str(tmp_path / "test_pkl_pp_scores.pkl")
    profile_1.to_pickle(out)
    read_profile = ScoreProfile.from_pickle(out)
    assert profile_1 == read_profile


def test_pkl_error():
    with pytest.raises(ValueError, match="File path must be provided."):
        ScoreProfile().to_pickle("")


def test_pkl_url():
    profile = ScoreProfile.from_pickle(
        "https://github.com/mggg/VoteKit/raw/refs/heads/main/examples/data/test_pkl_pp_scores.pkl"
    )

    assert isinstance(profile, ScoreProfile)

    with pytest.raises(URLError):
        ScoreProfile.from_pickle("https://www.fail.com")
