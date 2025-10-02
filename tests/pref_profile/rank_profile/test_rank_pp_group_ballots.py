from votekit.ballot import RankBallot
from votekit.pref_profile import RankProfile


def test_pp_group_ballots_ranking():
    profile = RankProfile(
        ballots=(
            RankBallot(ranking=({"A"}, {"B"}, {"C"}), weight=2),
            RankBallot(ranking=({"A"}, {"B"}, {"C"}), weight=1, voter_set={"Chris"}),
            RankBallot(
                ranking=({"A"}, {"B"}, {"C"}),
                weight=2,
                voter_set={"Moon", "Peter"},
            ),
        ),
        candidates=("A", "B", "C", "D"),
    )

    pp = profile.group_ballots()
    assert pp.ballots == (
        RankBallot(
            ranking=({"A"}, {"B"}, {"C"}),
            weight=5,
            voter_set={"Chris", "Moon", "Peter"},
        ),
    )
    assert set(pp.candidates) == set(profile.candidates)
    assert profile == pp
