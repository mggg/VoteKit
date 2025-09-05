from votekit.ballot import RankBallot
from votekit.pref_profile import RankProfile


def test_profile_equals_rankings():
    profile1 = RankProfile(
        ballots=(
            RankBallot(ranking=({"A"}, {"B"}, {"C"}), weight=1),
            RankBallot(ranking=({"A"}, {"B"}, {"C"}), weight=2),
            RankBallot(ranking=({"E"}, {"C"}, {"B"}), weight=2),
        )
    )
    profile2 = RankProfile(
        ballots=(
            RankBallot(ranking=({"E"}, {"C"}, {"B"}), weight=2),
            RankBallot(ranking=({"A"}, {"B"}, {"C"}), weight=3),
        )
    )
    assert profile1 == profile2


def test_profile_not_equals_candidates():
    profile1 = RankProfile(
        ballots=(
            RankBallot(ranking=({"A"}, {"B"}, {"C"}), weight=1),
            RankBallot(ranking=({"A"}, {"B"}, {"C"}), weight=2),
            RankBallot(ranking=({"E"}, {"C"}, {"B"}), weight=2),
        )
    )
    profile2 = RankProfile(
        ballots=(
            RankBallot(ranking=({"E"}, {"C"}, {"B"}), weight=2),
            RankBallot(ranking=({"A"}, {"B"}, {"C"}), weight=3),
        ),
        candidates=["A", "B", "C", "D", "E"],
    )

    assert profile1 != profile2


def test_profile_not_equals_ballot_length():
    profile1 = RankProfile(
        ballots=(
            RankBallot(ranking=({"A"}, {"B"}, {"C"}), weight=1),
            RankBallot(ranking=({"A"}, {"B"}, {"C"}), weight=2),
            RankBallot(ranking=({"E"}, {"C"}, {"B"}), weight=2),
        )
    )
    profile2 = RankProfile(
        ballots=(
            RankBallot(ranking=({"E"}, {"C"}, {"B"}), weight=2),
            RankBallot(ranking=({"A"}, {"B"}, {"C"}), weight=3),
        ),
        max_ranking_length=4,
    )

    assert profile1 != profile2


def test_profile_not_equals_cand_cast():
    profile1 = RankProfile(
        ballots=(
            RankBallot(ranking=({"A"}, {"B"}, {"C"}), weight=1),
            RankBallot(ranking=({"A"}, {"B"}, {"C"}), weight=2),
            RankBallot(ranking=({"E"}, {"C"}, {"B"}), weight=2),
        )
    )
    profile2 = RankProfile(
        ballots=(
            RankBallot(ranking=({"F"}, {"C"}, {"B"}), weight=2),
            RankBallot(ranking=({"A"}, {"B"}, {"C"}), weight=3),
        ),
    )

    assert profile1 != profile2


def test_profile_not_equals_ballot_wt():
    profile1 = RankProfile(
        ballots=(
            RankBallot(ranking=({"A"}, {"B"}, {"C"}), weight=2),
            RankBallot(ranking=({"A"}, {"B"}, {"C"}), weight=2),
            RankBallot(ranking=({"E"}, {"C"}, {"B"}), weight=2),
        )
    )
    profile2 = RankProfile(
        ballots=(
            RankBallot(ranking=({"E"}, {"C"}, {"B"}), weight=2),
            RankBallot(ranking=({"A"}, {"B"}, {"C"}), weight=3),
        ),
    )

    assert profile1 != profile2
