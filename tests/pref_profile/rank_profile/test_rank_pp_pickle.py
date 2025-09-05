from votekit.ballot import RankBallot
from votekit.pref_profile import RankProfile
import pytest

filepath = "tests/pref_profile/data/pickle"


def test_pkl_bijection_rankings():
    profile_rankings = RankProfile(
        ballots=(
            RankBallot(
                ranking=({"A", "B"}, frozenset(), {"C"}),
                voter_set={"Chris", "Peter"},
                weight=3 / 2,
            ),
            RankBallot(
                ranking=({"A", "B"}, frozenset(), {"C"}),
                voter_set={"Moon"},
                weight=1 / 2,
            ),
            RankBallot(
                ranking=(
                    {"A"},
                    {"B"},
                ),
            ),
            RankBallot(
                ranking=(
                    {"A"},
                    {"B"},
                ),
            ),
            RankBallot(
                ranking=(
                    {"A"},
                    {"B"},
                ),
            ),
        )
        * 5,
        max_ranking_length=3,
        candidates=["A", "B", "C", "D", "E"],
    )

    profile_rankings.to_pickle(f"{filepath}/test_pkl_pp_rankings.pkl")
    read_profile = RankProfile.from_pickle(f"{filepath}/test_pkl_pp_rankings.pkl")
    assert profile_rankings == read_profile


def test_pkl_error():
    with pytest.raises(ValueError, match="File path must be provided."):
        RankProfile().to_pickle("")
