from votekit.elections import Rating, ElectionState
from votekit.pref_profile import ScoreProfile
from votekit.ballot import ScoreBallot
import pytest
import pandas as pd

profile_no_tied_rating = ScoreProfile(
    ballots=[
        ScoreBallot(scores={"A": 2, "B": 1, "C": 1}, weight=2),
        ScoreBallot(scores={"A": 1, "B": 0, "C": 1}, weight=2),
        ScoreBallot(scores={"A": 2, "B": 1, "C": 1}),
    ]
)
# 8, 3,5


profile_no_tied_rating_round_1 = ScoreProfile(
    ballots=[
        ScoreBallot(scores={"B": 1, "C": 1}, weight=3),
        ScoreBallot(scores={"B": 0, "C": 1}, weight=2),
    ]
)

profile_tied_rating = ScoreProfile(
    ballots=[
        ScoreBallot(scores={"A": 1, "B": 1, "C": 1}),
        ScoreBallot(scores={"A": 3, "B": 3, "C": 3}),
    ]
)


states = [
    ElectionState(
        remaining=(frozenset({"A"}), frozenset({"C"}), frozenset({"B"})),
        scores={"A": 8, "B": 3, "C": 5},
    ),
    ElectionState(
        round_number=1,
        remaining=(frozenset({"C"}), frozenset({"B"})),
        elected=(frozenset({"A"}),),
        scores={"B": 3, "C": 5},
    ),
]


def test_init():
    e = Rating(profile_no_tied_rating, L=2)
    assert e.get_elected() == (frozenset({"A"}),)

    e = Rating(profile_no_tied_rating, m=2, L=2)
    assert e.get_elected() == (frozenset({"A"}), frozenset({"C"}))


def test_ties():
    e_random = Rating(profile_tied_rating, m=1, tiebreak="random", L=3)
    assert len([c for s in e_random.get_elected() for c in s]) == 1

    e_random = Rating(profile_tied_rating, m=2, tiebreak="random", L=3)
    assert len([c for s in e_random.get_elected() for c in s]) == 2

    e_random = Rating(profile_tied_rating, m=3, tiebreak="random", L=3)
    assert e_random.get_elected() == (frozenset({"A", "C", "B"}),)


def test_state_list():
    e = Rating(profile_no_tied_rating, L=2)
    assert e.election_states == states


def test_get_profile():
    e = Rating(profile_no_tied_rating, L=2)
    assert e.get_profile(0) == profile_no_tied_rating
    assert e.get_profile(1) == profile_no_tied_rating_round_1


def test_get_step():
    e = Rating(profile_no_tied_rating, L=2)
    assert e.get_step(1) == (profile_no_tied_rating_round_1, states[1])


def test_get_elected():
    e = Rating(profile_no_tied_rating, L=2)
    assert e.get_elected(0) == tuple()
    assert e.get_elected(1) == (frozenset({"A"}),)


def test_get_eliminated():
    e = Rating(profile_no_tied_rating, L=2)
    assert e.get_eliminated(0) == tuple()
    assert e.get_eliminated(1) == tuple()


def test_get_remaining():
    e = Rating(profile_no_tied_rating, L=2)
    assert e.get_remaining(0) == (frozenset({"A"}), frozenset({"C"}), frozenset({"B"}))
    assert e.get_remaining(1) == (frozenset({"C"}), frozenset({"B"}))


def test_get_ranking():
    e = Rating(profile_no_tied_rating, L=2)
    assert e.get_ranking(0) == (frozenset({"A"}), frozenset({"C"}), frozenset({"B"}))
    assert e.get_ranking(1) == (frozenset({"A"}), frozenset({"C"}), frozenset({"B"}))


def test_get_status_df():
    e = Rating(profile_no_tied_rating, L=2)

    df_0 = pd.DataFrame(
        {"Status": ["Remaining"] * 3, "Round": [0] * 3},
        index=["A", "C", "B"],
    )
    df_1 = pd.DataFrame(
        {"Status": ["Elected", "Remaining", "Remaining"], "Round": [1] * 3},
        index=["A", "C", "B"],
    )

    assert e.get_status_df(0).equals(df_0)
    assert e.get_status_df(1).equals(df_1)


def test_errors():
    with pytest.raises(ValueError, match="m must be positive."):
        Rating(profile_no_tied_rating, m=0, L=2)

    with pytest.raises(
        ValueError,
        match="Cannot elect correct number of candidates without breaking ties.",
    ):
        Rating(profile_tied_rating, m=2, L=3)


def test_validate_profile():
    with pytest.raises(TypeError, match="violates score limit"):
        profile = ScoreProfile(ballots=[ScoreBallot(scores={"A": 3})])
        Rating(profile, m=1, L=2)

    with pytest.raises(TypeError, match="must have non-negative scores."):
        profile = ScoreProfile(ballots=[ScoreBallot(scores={"A": -3})])
        Rating(profile, m=1, L=2)

    with pytest.raises(TypeError, match="All ballots must have score dictionary."):
        profile = ScoreProfile(ballots=[ScoreBallot(), ScoreBallot(scores={"A": 1})])
        Rating(profile, m=1, L=2)

    with pytest.raises(
        ValueError, match="Not enough candidates received votes to be elected."
    ):
        Rating(ScoreProfile(candidates=["A"]), m=1)
