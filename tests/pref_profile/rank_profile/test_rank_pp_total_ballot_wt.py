from votekit.ballot import RankBallot
from votekit.pref_profile import RankProfile

ballots = [
    RankBallot(ranking=({"A"}, {"B"}, {"C"}), weight=2),
    RankBallot(ranking=({"A"}, {"B"}, {"C"}), weight=2),
    RankBallot(),
    RankBallot(weight=0),
    RankBallot(
        weight=2,
    ),
    RankBallot(ranking=({"A", "B"}, frozenset(), {"D"}), voter_set={"Chris"}),
    RankBallot(
        ranking=({"A"}, {"B"}, {"C"}),
        weight=2,
    ),
]


def test_pp_total_ballot_wt():
    pp = RankProfile(ballots=ballots)
    assert pp.total_ballot_wt == 10

    pp = pp.group_ballots()
    assert pp.total_ballot_wt == 10
