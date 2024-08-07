from votekit.elections import Rating, ElectionState
from votekit import PreferenceProfile, Ballot
import pytest
import pandas as pd
from fractions import Fraction

profile_no_tied_rating = PreferenceProfile(
    ballots=[
        Ballot(scores={"A": 2, "B": 1, "C": 1}, weight=2),
        Ballot(scores={"A": 1, "B": 0, "C": 1}, weight=2),
        Ballot(scores={"A": 2, "B": 1, "C": 1}),
    ]
)
# 8, 3,5


profile_no_tied_rating_round_1 = PreferenceProfile(
    ballots=[
        Ballot(scores={"B": 1, "C": 1}, weight=3),
        Ballot(scores={"B": 0, "C": 1}, weight=2),
    ]
)

profile_tied_rating = PreferenceProfile(
    ballots=[
        Ballot(scores={"A": 1, "B": 1, "C": 1}),
        Ballot(scores={"A": 3, "B": 3, "C": 3}),
    ]
)


states = [
    ElectionState(
        remaining=(frozenset({"A"}), frozenset({"C"}), frozenset({"B"})),
        scores={"A": Fraction(8), "B": Fraction(3), "C": Fraction(5)},
    ),
    ElectionState(
        round_number=1,
        remaining=(frozenset({"C"}), frozenset({"B"})),
        elected=(frozenset({"A"}),),
        scores={"B": Fraction(3), "C": Fraction(5)},
    ),
]


def test_init():
    e = Rating(profile_no_tied_rating, L=2)
    assert e.get_elected() == (frozenset({"A"}),)

    e = Rating(profile_no_tied_rating, m=2, L=2)
    assert e.get_elected() == (frozenset({"A"}), frozenset({"C"}))


def test_ties():
    e_random = Rating(profile_tied_rating, m=1, tiebreak="random", L=3)  # noqa
    assert len([c for s in e_random.get_elected() for c in s]) == 1

    e_random = Rating(profile_tied_rating, m=2, tiebreak="random", L=3)  # noqa
    assert len([c for s in e_random.get_elected() for c in s]) == 2

    e_random = Rating(profile_tied_rating, m=3, tiebreak="random", L=3)  # noqa
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
    with pytest.raises(ValueError):  # m must be non negative
        Rating(profile_no_tied_rating, m=0, L=2)

    with pytest.raises(ValueError):  # m must be less than num cands
        Rating(profile_no_tied_rating, m=4, L=2)

    with pytest.raises(ValueError):  # needs tiebreak
        Rating(profile_tied_rating, m=2, L=3)


def test_validate_profile():
    with pytest.raises(TypeError):  # must be less than limit
        profile = PreferenceProfile(ballots=[Ballot(scores={"A": 3})])
        Rating(profile, m=2, L=2)

    with pytest.raises(TypeError):  # must be non-negative
        profile = PreferenceProfile(ballots=[Ballot(scores={"A": -3})])
        Rating(profile, m=2, L=2)

    with pytest.raises(TypeError):  # must have scores
        profile = PreferenceProfile(ballots=[Ballot()])
        Rating(profile, m=2, L=2)
