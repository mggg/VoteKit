from votekit.ballot import Ballot
import pytest


def test_ballot_init():
    b = Ballot()
    assert isinstance(b, Ballot)
    assert b.weight == 1
    assert b.voter_set == frozenset()
    assert b._frozen


def test_ballot_is_frozen_set():
    b = Ballot()
    with pytest.raises(AttributeError, match="is frozen"):
        b.weight = 2
    with pytest.raises(AttributeError, match="is frozen"):
        b.voter_set = frozenset({"A"})
    with pytest.raises(AttributeError, match="is frozen"):
        b._frozen = False


def test_ballot_is_frozen_del():
    b = Ballot(weight=2, voter_set={"A"})
    with pytest.raises(AttributeError, match="is frozen"):
        del b.weight
    with pytest.raises(AttributeError, match="is frozen"):
        del b.voter_set
    with pytest.raises(AttributeError, match="is frozen"):
        del b._frozen


def test_ballot_hash():
    b1 = Ballot(weight=2, voter_set={"A"})
    b2 = Ballot(weight=2, voter_set={"A"})
    b3 = Ballot(weight=2, voter_set={"B"})

    assert b1 == b2 and hash(b1) == hash(b2)
    assert b1 != b3 and hash(b1) != hash(b3)

    assert b2 in {b1}


def test_ballot_coerce_wt_to_float():
    assert isinstance(Ballot(weight=3).weight, float)
    assert isinstance(Ballot(weight=3.2).weight, float)


def test_ballot_eq():
    b = Ballot(weight=3, voter_set={"Chris", "Peter"})

    assert b == Ballot(weight=3.0, voter_set={"Peter", "Chris"})
    assert b != "Hello"
    assert b != Ballot(weight=3.1, voter_set={"Peter", "Chris"})
    assert b != Ballot(weight=3.0, voter_set={"Chris"})


def test_ballot_str():
    b = Ballot(
        weight=3,
        voter_set={"Chris"},
    )

    assert str(b) == "Ballot\nWeight: 3.0\nVoter set: {'Chris'}"

    b = Ballot(
        weight=3,
    )

    assert str(b) == "Ballot\nWeight: 3.0"


def test_ballot_negative_weight():
    with pytest.raises(ValueError, match="Ballot weight cannot be negative."):
        Ballot(weight=-1.5)


def test_rank_and_score():
    with pytest.raises(
        TypeError, match="Only one of ranking or scores can be provided."
    ):
        Ballot(ranking=[{"A"}], scores={"A": 1})
