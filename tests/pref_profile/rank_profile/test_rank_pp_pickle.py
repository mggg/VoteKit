from urllib.error import URLError

import pytest

from votekit.ballot import RankBallot
from votekit.pref_profile import RankProfile


def test_pkl_bijection_rankings(tmp_path):
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

    out = str(tmp_path / "test_pkl_pp_rankings.pkl")
    profile_rankings.to_pickle(out)
    read_profile = RankProfile.from_pickle(out)
    assert profile_rankings == read_profile


def test_pkl_error():
    with pytest.raises(ValueError, match="File path must be provided."):
        RankProfile().to_pickle("")


def test_pkl_url():
    profile = RankProfile.from_pickle(
        "https://github.com/mggg/VoteKit/raw/refs/heads/main/examples/data/test_pkl_pp_rankings.pkl"
    )

    assert isinstance(profile, RankProfile)

    with pytest.raises(URLError):
        RankProfile.from_pickle("https://www.fail.com")
