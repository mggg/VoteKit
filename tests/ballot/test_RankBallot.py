from votekit.ballot import RankBallot, Ballot
import pytest


def test_ballot_init():
    b = RankBallot()
    assert isinstance(b, RankBallot)
    assert b.ranking is None
    assert b.weight == 1
    assert b.voter_set == frozenset()


def test_init_from_parent_class():
    b = Ballot(ranking=[{"A"}, {"B"}], voter_set={"Chris"}, weight=2)
    assert isinstance(b, RankBallot)

    assert isinstance(b.ranking, tuple)
    assert isinstance(b.ranking[0], frozenset)
    assert b.ranking == (frozenset({"A"}), frozenset({"B"}))

    assert isinstance(b.weight, float)
    assert b.weight == 2.0

    assert isinstance(b.voter_set, frozenset)
    assert b.voter_set == frozenset({"Chris"})

    assert b == RankBallot(ranking=[{"A"}, {"B"}], voter_set={"Chris"}, weight=2)


def test_ballot_is_frozen():
    b = RankBallot()
    with pytest.raises(AttributeError, match="is frozen"):
        b.ranking = (frozenset({"A"}),)
    with pytest.raises(AttributeError, match="is frozen"):
        b.weight = 2
    with pytest.raises(AttributeError, match="is frozen"):
        b.voter_set = frozenset({"A"})
    with pytest.raises(AttributeError, match="is frozen"):
        b._frozen = False


def test_ballot_is_frozen_del():
    b = RankBallot(ranking=[{"A"}], weight=2, voter_set={"A"})
    with pytest.raises(AttributeError, match="is frozen"):
        del b.weight
    with pytest.raises(AttributeError, match="is frozen"):
        del b.voter_set
    with pytest.raises(AttributeError, match="is frozen"):
        del b._frozen
    with pytest.raises(AttributeError, match="is frozen"):
        del b.ranking


def test_ballot_hash():
    b1 = RankBallot(ranking=[{"A"}], weight=2, voter_set={"A"})
    b2 = RankBallot(ranking=[{"A"}], weight=2, voter_set={"A"})
    b3 = RankBallot(ranking=[{"A"}], weight=1, voter_set={"B"})

    assert b1 == b2 and hash(b1) == hash(b2)
    assert b1 != b3 and hash(b1) != hash(b3)

    assert b2 in {b1}


def test_ballot_coerce_wt_to_float():
    assert isinstance(RankBallot(weight=3).weight, float)
    assert isinstance(RankBallot(weight=3.2).weight, float)


def test_ballot_strip_whitespace():
    b = RankBallot(
        ranking=(frozenset({" Chris", "Peter "}), frozenset({" Moon "}), frozenset()),
    )

    assert b.ranking == (
        frozenset({"Chris", "Peter"}),
        frozenset({"Moon"}),
        frozenset(),
    )


def test_ballot_tilde_errors():
    with pytest.raises(
        ValueError,
        match="'~' is a reserved character and cannot be used for candidate names.",
    ):
        RankBallot(ranking=({"~"},))


def test_ballot_negative_weight():
    with pytest.raises(ValueError, match="Ballot weight cannot be negative."):
        RankBallot(weight=-1.5)


def test_ballot_eq():
    b = RankBallot(
        ranking=[{"A"}, {"B"}, {"C"}],
        weight=3,
        voter_set={"Chris", "peter"},
    )

    assert b == RankBallot(
        ranking=[{"A"}, {"B"}, {"C"}],
        weight=3.0,
        voter_set={"peter", "Chris"},
    )

    assert b != "Hello"

    assert b != RankBallot(
        weight=3,
        voter_set={"Chris", "peter"},
    )

    assert b != RankBallot(
        ranking=[{"A"}, {"B"}, {"C"}],
        voter_set={"Chris", "peter"},
    )

    assert b != RankBallot(
        ranking=[{"A"}, {"B"}, {"C"}],
        weight=3,
    )

    assert b != RankBallot(
        ranking=[{"B"}, {"A"}, {"C"}],
        weight=3,
        voter_set={"Chris", "peter"},
    )


def test_ballot_str():
    b = RankBallot(
        ranking=[{"A"}, {"B"}, {"C"}],
        weight=3,
        voter_set={"Chris"},
    )

    assert (
        str(b)
        == "RankBallot\n1.) A, \n2.) B, \n3.) C, \nWeight: 3.0\nVoter set: {'Chris'}"
    )


def test_rank_sub_ballot():
    assert isinstance(RankBallot(), Ballot)
    assert isinstance(RankBallot(), RankBallot)


def test_rank_and_score():
    with pytest.raises(
        TypeError, match="Only one of ranking or scores can be provided."
    ):
        RankBallot(ranking=[{"A"}], scores={"A": 1})
