from votekit.matrices import comention
from votekit.ballot import Ballot


def test_comention_ranked():
    b = Ballot(ranking=({"Chris"}, {"Peter"}, {"Moon"}))

    assert comention(["Chris", "Peter"], b)
    assert comention(["Moon", "Peter"], b)
    assert comention(["Chris", "Moon", "Peter"], b)
    assert not comention(["Chris", "Jeanne"], b)
    assert not comention(["Jeanne", "David"], b)


def test_comentions_scored():
    b = Ballot(scores={"Chris": 1, "Peter": 4, "Jeanne": 0, "Moon": 2})
    assert comention(["Chris", "Peter"], b)
    assert comention(["Moon", "Peter"], b)
    assert not comention(["Jeanne", "Peter"], b)
    assert not comention(["David", "Mala"], b)


def test_comentions_rank_score():
    b = Ballot(
        ranking=({"Chris"}, {"Peter"}, {"Moon"}),
        scores={"Mala": 1, "Peter": 4, "Jeanne": 0, "David": 2},
    )

    assert comention(["Chris", "Peter"], b)
    assert comention(["Moon", "David"], b)
    assert comention(["David", "Mala"], b)
    assert not comention(["Jeanne", "Peter"], b)
    assert not comention(["Jeanne", "Tyler"], b)
