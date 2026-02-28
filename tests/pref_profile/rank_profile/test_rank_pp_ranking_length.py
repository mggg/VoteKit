import pytest

from votekit.ballot import RankBallot
from votekit.pref_profile import RankProfile


def test_ranking_length_default():
    with pytest.raises(
        ValueError,
        match="number of candidates exceeds the length of the longest ranking",
    ):
        RankProfile(
            ballots=(
                RankBallot(ranking=({"A"}, {"B"}, {"C", "D"})),
                RankBallot(ranking=({"A"}, {"B"}), weight=3 / 2),
                RankBallot(ranking=({"C"}, {"B"}), weight=2),
            )
        )


def test_ranking_length_no_default():
    profile = RankProfile(
        ballots=(
            RankBallot(ranking=({"A"}, {"B"}, {"C", "D"})),
            RankBallot(ranking=({"A"}, {"B"}), weight=3 / 2),
            RankBallot(ranking=({"C"}, {"B"}), weight=2),
        ),
        max_ranking_length=4,
    )

    assert profile.max_ranking_length == 4
