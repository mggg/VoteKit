from votekit.ballot import Ballot
from votekit.matrices import comention


def test_comention_ranked():
    b = Ballot(ranking=({"Chris"}, {"Peter"}, {"Moon"}))

    assert comention(["Chris", "Peter"], b)
    assert comention(["Moon", "Peter"], b)
    assert comention(["Chris", "Moon", "Peter"], b)
    assert not comention(["Chris", "Jeanne"], b)
    assert not comention(["Jeanne", "David"], b)
