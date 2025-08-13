from votekit import Ballot
import pytest


def test_ballot_init():
    b = Ballot()
    assert not b.ranking
    assert b.weight == 1
    assert not b.voter_set
    assert not b.scores


def test_ballot_strip_whitespace():
    b = Ballot(
        ranking=(frozenset({" Chris", "Peter "}), frozenset({" Moon "}), frozenset()),
        scores={" Chris": 2, "Peter ": 1, " Moon ": 3},
    )

    assert b.ranking == (
        frozenset({"Chris", "Peter"}),
        frozenset({"Moon"}),
        frozenset(),
    )
    assert b.scores == {"Chris": 2, "Peter": 1, "Moon": 3}


def test_ballot_post_init():
    assert isinstance(Ballot(weight=3).weight, float)
    assert isinstance(Ballot(weight=3.2).weight, float)
    assert Ballot(scores={"A": 1, "B": 0}).scores == {"A": float(1)}
    assert Ballot(scores={"A": 1, "B": -1}).scores == {
        "A": 1,
        "B": -1,
    }
    assert isinstance(Ballot(scores={"A": 2, "B": 2}).scores["A"], float)
    assert isinstance(Ballot(scores={"A": 2.0, "B": 2}).scores["A"], float)

    with pytest.raises(TypeError, match="Score values must be numeric."):
        Ballot(scores={"A": "a"})


def test_ballot_eq():
    b = Ballot(
        ranking=[{"A"}, {"B"}, {"C"}],
        weight=3,
        voter_set={"Chris"},
        scores={"A": 1, "B": 1 / 2, "C": 0},
    )

    assert b != Ballot(
        weight=3, voter_set={"Chris"}, scores={"A": 1, "B": 1 / 2, "C": 0}
    )

    assert b != Ballot(
        ranking=[{"A"}, {"B"}, {"C"}],
        voter_set={"Chris"},
        scores={"A": 1, "B": 1 / 2, "C": 0},
    )

    assert b != Ballot(
        ranking=[{"A"}, {"B"}, {"C"}],
        weight=3,
        scores={"A": 1, "B": 1 / 2, "C": 0},
    )

    assert b != Ballot(
        ranking=[{"A"}, {"B"}, {"C"}],
        weight=3,
        voter_set={"Chris"},
    )


def test_ballot_str():
    b = Ballot(
        ranking=[{"A"}, {"B"}, {"C"}],
        weight=3,
        voter_set={"Chris"},
        scores={"A": 1, "B": 1 / 2, "C": 0},
    )

    assert (
        str(b)
        == "Ranking\n1.) A, \n2.) B, \n3.) C, \nScores\nA: 1.00\nB: 0.50\nWeight: 3.0"
    )


def test_ballot_tilde_errors():
    with pytest.raises(
        ValueError,
        match="'~' is a reserved character and cannot be used for candidate names.",
    ):
        Ballot(ranking=({"~"},))

    with pytest.raises(
        ValueError,
        match="'~' is a reserved character and cannot be used for candidate names.",
    ):
        Ballot(scores={"~": 1})


def test_ballot_negative_weight():
    with pytest.raises(ValueError, match="Ballot weight cannot be negative."):
        Ballot(weight=-1.5)
