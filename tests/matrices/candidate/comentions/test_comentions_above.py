from votekit.matrices import comention_above
from votekit.ballot import Ballot
import pytest


def test_comention_above_error():
    b = Ballot(scores={"Chris": 1, "Peter": 4, "Jeanne": 0, "Moon": 2})
    with pytest.raises(TypeError, match="Ballot must be of type RankBallot"):
        comention_above("Chris", "Peter", b)


def test_comention_above():
    b = Ballot(ranking=({"Chris"}, {"Peter"}, {"Moon", "Jeanne"}))

    assert comention_above("Chris", "Peter", b)
    assert comention_above("Moon", "Jeanne", b)
    assert comention_above("Peter", "Jeanne", b)
    assert not comention_above("Chris", "David", b)
    assert not comention_above("Mala", "David", b)
    assert not comention_above("Moon", "Chris", b)
