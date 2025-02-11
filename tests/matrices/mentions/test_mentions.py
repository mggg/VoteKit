from votekit.matrices import mention
from votekit.ballot import Ballot


def test_mentions_ranked():
    b = Ballot(ranking=({"Chris"}, {"Peter"}))
    assert mention("Chris", b)
    assert not mention("Moon", b)


def test_mentions_scored():
    b = Ballot(scores={"Chris": 1, "Peter": 4, "Jeanne": 0})
    assert mention("Chris", b)
    assert not mention("Jeanne", b)
    assert not mention("Moon", b)


def test_mentions_rank_score():
    b = Ballot(
        ranking=({"Chris"}, {"Nick"}), scores={"Chris": 1, "Peter": 4, "Jeanne": 0}
    )

    assert mention("Chris", b)
    assert mention("Nick", b)
    assert mention("Peter", b)
    assert not mention("Jeanne", b)
    assert not mention("Moon", b)
