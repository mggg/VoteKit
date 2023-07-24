from votekit.election_state import ElectionState
from votekit.profile import PreferenceProfile
from votekit.ballot import Ballot
from fractions import Fraction

##TODO: use Scottish 3-cand ward_03 data,

b1 = Ballot(ranking=[{"A"}, {"B"}, {"C"}], weight=Fraction(250, 1))
b2 = Ballot(ranking=[{"B"}, {"A"}, {"C"}], weight=Fraction(200, 1))
b3 = Ballot(ranking=[{"C"}, {"B"}, {"A"}], weight=Fraction(100, 1))
ballots_2 = [b1, b2, b3]
pref_0 = PreferenceProfile(ballots=ballots_2)
pref_1 = PreferenceProfile(
    ballots=[
        Ballot(ranking=[{"A"}, {"B"}], weight=Fraction(250, 1)),
        Ballot(ranking=[{"B"}, {"A"}], weight=Fraction(300, 1)),
    ]
)
pref_2 = PreferenceProfile(ballots=[Ballot(ranking=[{"A"}], weight=Fraction(274, 1))])
round_0 = ElectionState(
    curr_round=0,
    elected=[],
    eliminated=[],
    remaining=["A", "B", "C"],
    profile=pref_0,
    previous=None,
)
round_1 = ElectionState(
    curr_round=1,
    elected=[],
    eliminated=["C"],
    remaining=["B", "A"],
    profile=pref_1,
    previous=round_0,
)
round_2 = ElectionState(
    curr_round=2,
    elected=["B"],
    eliminated=[],
    remaining=["A"],
    winner_votes={"B": [{"C"}, {"B"}, {"A"}]},
    profile=pref_2,
    previous=round_1,
)

rounds = [round_0, round_1, round_2]
rds = [0, 1, 2]
elects = [[], [], ["B"]]
elims = [[], ["C"], []]
remains = [["A", "B", "C"], ["B", "A"], ["A"]]
wins = [[], [], ["B"]]
los = [[], ["C"], ["C"]]
ranks = [["A", "B", "C"], ["B", "A", "C"], ["B", "A", "C"]]


def test_get_attributes():

    for i in range(3):
        assert rounds[i].curr_round == rds[i]
        assert rounds[i].elected == elects[i]
        assert rounds[i].eliminated == elims[i]
        assert rounds[i].remaining == remains[i]

    # orig = Outcome(remaining=["A", "B", "C"])
    # new = orig.add_winners_and_losers({"B"}, {"A"})
    # assert new.elected == {"B"}
    # assert new.eliminated == {"A"}
    # assert new.remaining == {"C"}


def test_lists():
    for i in range(3):
        assert rounds[i].get_all_winners() == wins[i]
        assert rounds[i].get_all_eliminated() == los[i]
        assert rounds[i].get_rankings() == ranks[i]


def test_changed_rankings():
    assert rounds[1].changed_rankings() == {"A": (0, 1), "B": (1, 0)}


##TODO: the rest of this
def test_round_outcome():
    return


def test_difference_remaining_candidates():
    return
