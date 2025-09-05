from votekit.ballot import ScoreBallot
from votekit.pref_profile import ScoreProfile
import pytest

filepath = "tests/pref_profile/data/pickle"


def test_pkl_bijection_scores():
    profile_1 = ScoreProfile(
        ballots=[
            ScoreBallot(scores={"A": 2, "B": 2}, weight=2),
            ScoreBallot(scores={"A": 2, "C": 2}, voter_set={"Chris"}),
            ScoreBallot(),
            ScoreBallot(weight=0),
        ],
        candidates=["A", "B", "C", "D"],
    )

    profile_1.to_pickle(f"{filepath}/test_pkl_pp_scores.pkl")
    read_profile = ScoreProfile.from_pickle(f"{filepath}/test_pkl_pp_scores.pkl")
    print(type(read_profile))
    assert profile_1 == read_profile


def test_pkl_error():
    with pytest.raises(ValueError, match="File path must be provided."):
        ScoreProfile().to_pickle("")
