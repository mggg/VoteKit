from votekit.ballot import RankBallot
from votekit.pref_profile import RankProfile


def test_add_profiles():
    profile_1 = RankProfile(
        ballots=[
            RankBallot(ranking=({"A"}, {"B"}, {"C"}), weight=2),
            RankBallot(ranking=({"A", "B"}, frozenset(), {"D"}), voter_set={"Chris"}),
            RankBallot(),
            RankBallot(weight=0),
        ],
        candidates=["A", "B", "C", "D"],
        max_ranking_length=3,
    )

    profile_2 = RankProfile(
        ballots=[
            RankBallot(ranking=({"E"}, {"D"}, {"F"}, {"E"}), weight=2),
            RankBallot(ranking=({"D"}, {"E"}, {"F"}), weight=2),
            RankBallot(),
            RankBallot(weight=0),
        ],
        candidates=["D", "E", "F"],
        max_ranking_length=0,
    )
    summed_profile = profile_1 + profile_2
    true_summed_profile = RankProfile(
        ballots=(
            RankBallot(ranking=({"A"}, {"B"}, {"C"}), weight=2),
            RankBallot(ranking=({"A", "B"}, frozenset(), {"D"}), voter_set={"Chris"}),
            RankBallot(ranking=({"E"}, {"D"}, {"F"}, {"E"}), weight=2),
            RankBallot(ranking=({"D"}, {"E"}, {"F"}), weight=2),
            RankBallot(),
            RankBallot(weight=0),
            RankBallot(),
            RankBallot(weight=0),
        ),
        candidates=["A", "B", "C", "D", "E", "F"],
        max_ranking_length=4,
    )

    assert set(summed_profile.candidates) == set(["A", "B", "C", "D", "E", "F"])
    assert summed_profile.max_ranking_length == 4
    assert isinstance(summed_profile, RankProfile)
    assert true_summed_profile == summed_profile
