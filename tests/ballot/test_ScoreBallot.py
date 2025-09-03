from votekit.ballot import ScoreBallot, Ballot
import pytest


# TODO add hash
def test_ballot_init():
    b = ScoreBallot()
    assert isinstance(b, ScoreBallot)
    assert b.scores is None
    assert b.weight == 1
    assert b.voter_set == frozenset()


def test_init_from_parent_class():
    b = Ballot(scores={"A": 3, "B": 0}, voter_set={"Chris"}, weight=2)
    assert isinstance(b, ScoreBallot)

    assert isinstance(b.scores, dict)
    assert isinstance(b.scores["A"], float)
    assert b.scores == {"A": 3.0}

    assert isinstance(b.weight, float)
    assert b.weight == 2.0

    assert isinstance(b.voter_set, frozenset)
    assert b.voter_set == frozenset({"Chris"})

    assert b == ScoreBallot(scores={"A": 3, "B": 0}, voter_set={"Chris"}, weight=2)


# TODO add delete tests
def test_ballot_is_frozen():
    b = ScoreBallot()
    with pytest.raises(AttributeError, match="is frozen"):
        b.scores = {"A": 2}
    with pytest.raises(AttributeError, match="is frozen"):
        b.weight = 2
    with pytest.raises(AttributeError, match="is frozen"):
        b.voter_set = frozenset({"A"})
    with pytest.raises(AttributeError, match="is frozen"):
        b._frozen = False


def test_ballot_coerce_wt_to_float():
    assert isinstance(ScoreBallot(weight=3).weight, float)
    assert isinstance(ScoreBallot(weight=3.2).weight, float)


def test_ballot_strip_whitespace():
    b = ScoreBallot(
        scores={"A ": 2, "C": 1, " B ": 2},
    )

    assert b.scores == {"A": 2, "B": 2, "C": 1}


def test_ballot_tilde_errors():
    with pytest.raises(
        ValueError,
        match="'~' is a reserved character and cannot be used for candidate names.",
    ):
        ScoreBallot(scores={"~": 1})


def test_ballot_negative_weight():
    with pytest.raises(ValueError, match="Ballot weight cannot be negative."):
        ScoreBallot(weight=-1.5)


def test_ballot_eq():
    b = ScoreBallot(
        scores={"A": 2, "C": 1},
        weight=3,
        voter_set={"Chris", "peter"},
    )

    assert b == ScoreBallot(
        scores={"A": 2, "C": 1},
        weight=3.0,
        voter_set={"peter", "Chris"},
    )

    assert b != "Hello"

    assert b != ScoreBallot(
        weight=3,
        voter_set={"Chris", "peter"},
    )

    assert b != ScoreBallot(
        scores={"A": 2, "C": 1},
        voter_set={"Chris", "peter"},
    )

    assert b != ScoreBallot(
        scores={"A": 2, "C": 1},
        weight=3,
    )

    assert b != ScoreBallot(
        scores={"A": 2, "C": 2, "B": 4},
        weight=3,
        voter_set={"Chris", "peter"},
    )


def test_ballot_str():
    b = ScoreBallot(
        scores={"A": 2, "C": 1},
        weight=3,
        voter_set={"Chris"},
    )

    assert str(b) == "ScoreBallot\nA: 2.00\nC: 1.00\nWeight: 3.0\nVoter set: {'Chris'}"


def test_rank_sub_ballot():
    assert isinstance(ScoreBallot(), Ballot)
    assert isinstance(ScoreBallot(), ScoreBallot)


def test_rank_and_score():
    with pytest.raises(
        TypeError, match="Only one of ranking or scores can be provided."
    ):
        ScoreBallot(ranking=[{"A"}], scores={"A": 1})
