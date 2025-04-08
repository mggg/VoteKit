from votekit.ballot import Ballot
from votekit.pref_profile import PreferenceProfile

ballots = [
    Ballot(ranking=({"A"}, {"B"}, {"C"}), weight=2),
    Ballot(ranking=({"A"}, {"B"}, {"C"}), weight=2),
    Ballot(),
    Ballot(weight=0),
    Ballot(
        weight=2,
        scores={
            "A": 1,
            "B": 2,
        },
    ),
    Ballot(ranking=({"A", "B"}, frozenset(), {"D"}), voter_set={"Chris"}),
    Ballot(
        ranking=({"A"}, {"B"}, {"C"}),
        weight=2,
        scores={
            "A": 1,
            "B": 2,
        },
    ),
]


def test_pp_total_ballot_wt():
    pp = PreferenceProfile(ballots=ballots)
    assert pp.total_ballot_wt == 10

    pp = pp.group_ballots()
    assert pp.total_ballot_wt == 10
