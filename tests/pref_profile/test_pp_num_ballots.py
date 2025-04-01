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
    Ballot(ranking=({"A", "B"}, frozenset(), {"D"}), id="X29", voter_set={"Chris"}),
    Ballot(
        ranking=({"A"}, {"B"}, {"C"}),
        weight=2,
        scores={
            "A": 1,
            "B": 2,
        },
    ),
]


def test_pp_num_ballots():
    pp = PreferenceProfile(ballots=ballots)
    assert pp.num_ballots == 7
    for b in pp.ballots:
        print(b)
        print()

    pp = pp.group_ballots()
    print("------")
    for b in pp.ballots:
        print(b)
        print()
    assert pp.num_ballots == 5
