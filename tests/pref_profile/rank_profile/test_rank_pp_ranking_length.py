from votekit.ballot import RankBallot
from votekit.pref_profile import RankProfile


def test_ranking_length_default():
    profile = RankProfile(
        ballots=(
            RankBallot(ranking=({"A"}, {"B"}, {"C", "D"})),
            RankBallot(ranking=({"A"}, {"B"}), weight=3 / 2),
            RankBallot(ranking=({"C"}, {"B"}), weight=2),
        )
    )

    assert profile.max_ranking_length == 3


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
