# import sys

# sys.path.insert(0, r"C:\Users\malav\OneDrive\Desktop\mggg\VoteKit-Mala\src\votekit")
# import pytest
from ballot import Ballot


def initialize():
    global ballot_1, ballot_2, ballot_3
    ballot_1 = Ballot(ranking=[{"A"}, {"B"}, {"C"}], weight=50)
    ballot_2 = Ballot(id="ballot_2", ranking=[{"A"}, {"B"}, {"C"}], weight=100)
    ballot_3 = Ballot(
        id="ballot_3",
        ranking=[{"A", "D"}, {"B"}, {"C"}],
        weight=200,
        voters={"voter1x150", "voter2x50"},
    )


def test_ballot_attributes():
    assert not ballot_1.id
    assert ballot_2.id == "ballot_2"
    assert ballot_3.id == "ballot_3"

    assert ballot_1.ranking == [{"A"}, {"B"}, {"C"}]
    assert ballot_2.ranking == [{"A"}, {"B"}, {"C"}]
    assert ballot_3.ranking == [{"D", "A"}, {"B"}, {"C"}]

    assert ballot_1.weight == 50
    assert ballot_2.weight == 100
    assert ballot_3.weight == 200

    assert not ballot_1.voters
    assert not ballot_2.voters
    assert ballot_3.voters == {"voter1x150", "voter2x50"}


initialize()
test_ballot_attributes()
