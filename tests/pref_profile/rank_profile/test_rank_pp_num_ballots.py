from votekit.ballot import RankBallot
from votekit.pref_profile import RankProfile

ballots = [
    RankBallot(ranking=({"A"}, {"B"}, {"C"}), weight=2),
    RankBallot(ranking=({"A"}, {"B"}, {"C"}), weight=2),
    RankBallot(),
    RankBallot(weight=0),
    RankBallot(ranking=({"A", "B"}, frozenset(), {"D"}), voter_set={"Chris"}),
]


def test_pp_num_ballots():
    pp = RankProfile(ballots=ballots)
    assert pp.num_ballots == 5

    pp = pp.group_ballots()
    assert pp.num_ballots == 3
