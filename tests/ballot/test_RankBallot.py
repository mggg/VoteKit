import pytest

from votekit.ballot import Ballot, RankBallot


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

    assert str(b) == "RankBallot\n1.) A, \n2.) B, \n3.) C, \nWeight: 3.0\nVoter set: {'Chris'}"


def test_rank_sub_ballot():
    assert isinstance(RankBallot(), Ballot)
    assert isinstance(RankBallot(), RankBallot)


def test_rank_and_score():
    with pytest.raises(TypeError, match="Only one of ranking or scores can be provided."):
        RankBallot(ranking=[{"A"}], scores={"A": 1})


def test_single_char_str_ranking_raises_type_error():
    with pytest.raises(
        TypeError, match="If you intended this to be a bullet vote, then wrap it in a list."
    ):
        RankBallot(ranking="A")


def test_mult_char_str_ranking_raises_type_error():
    """
    Regression test: Ties should be indicated by wrapping tied candidates in an iterable.
    Previously, a string ranking would be accepted and split str elements into a tie.
    This is not the intended behavior, and we want to raise a TypeError instead.
    """
    with pytest.raises(
        TypeError, match="If you intended this to be a bullet vote, then wrap it in a list."
    ):
        RankBallot(ranking="AB")


def test_str_singleton_ranking_elements():
    b = RankBallot(ranking=["A", "B", "C"], weight=1, voter_set={"A"})
    assert b.ranking == (frozenset({"A"}), frozenset({"B"}), frozenset({"C"}))


def test_mixed_str_and_iterable_ranking_elements():
    b = RankBallot(ranking=["A", {"B", "C"}, "D", {"E"}], weight=1, voter_set={"A"})
    assert b.ranking == (
        frozenset({"A"}),
        frozenset({"B", "C"}),
        frozenset({"D"}),
        frozenset({"E"}),
    )


def test_mixed_str_int_candidates_ballot():
    b = RankBallot(ranking=["A", {"B", 1}, "D", {2}, 3], weight=1, voter_set={"A"})
    assert b.ranking == (
        frozenset({"A"}),
        frozenset({"B", 1}),
        frozenset({"D"}),
        frozenset({2}),
        frozenset({3}),
    )


def test_equivalent_str_int_candidates_gives_warning():
    with pytest.warns(UserWarning, match="will be treated as separate candidates"):
        b = RankBallot(ranking=[1, "1"])
    assert b.ranking == (frozenset({1}), frozenset({"1"}))


def test_invalid_bare_candidate_type_ballot():
    with pytest.raises(
        TypeError, match="Ranking is a sequence of Iterables or bare str/int candidates."
    ):
        RankBallot(ranking=[1.5, {"B", 1}, "D", {2}, 3], weight=1, voter_set={"A"})  # type: ignore[arg-type]


def test_invalid_wrapped_candidate_type_ballot():
    with pytest.raises(
        TypeError, match=r"Non-string/integer candidate\(s\) found in RankBallot.ranking"
    ):
        RankBallot(ranking=[{1.5}, {"B", 1}, "D", {2}, 3], weight=1, voter_set={"A"})  # type: ignore[arg-type]


def test_negative_integer_candidate_ballot():
    with pytest.raises(
        ValueError, match=r"Negative integer candidate\(s\) found in RankBallot.ranking"
    ):
        RankBallot(ranking=["A", {"B", -1}, "D", {2}, 3], weight=1, voter_set={"A"})


def test_non_sequence_ranking_ballot_raises_error():
    with pytest.raises(TypeError, match="ranking must be a Sequence with a guaranteed order."):
        RankBallot(ranking={4, 3, 2, 1}, weight=1, voter_set={"A"})  # type: ignore[arg-type]


def test_bool_candidate_ballot_raises_error():
    with pytest.raises(TypeError, match=r"Boolean candidate\(s\) found in RankBallot.ranking"):
        RankBallot(ranking=[1, {True}], weight=1, voter_set={"A"})


def test_colon_char_in_candidate_ballot_raises_error():
    with pytest.raises(ValueError, match="':' found in RankBallot.ranking"):
        RankBallot(ranking=[{"A:B"}], weight=1, voter_set={"A"})
