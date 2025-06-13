from votekit.elections import Cumulative, ElectionState
from votekit import PreferenceProfile, Ballot
import pytest
import pandas as pd

profile_no_tied_cumulative = PreferenceProfile(
    ballots=[
        Ballot(scores={"A": 1, "B": 1, "C": 0}, weight=2),
        Ballot(scores={"A": 2, "B": 0, "C": 0}),
        Ballot(scores={"A": 0, "B": 1, "C": 1}),
    ]
)
# 4,3,1


profile_no_tied_cumulative_round_1 = PreferenceProfile(
    ballots=[
        Ballot(scores={"C": 1}, weight=1),
    ]
)

profile_tied_cumulative = PreferenceProfile(
    ballots=[
        Ballot(scores={"A": 2, "B": 0, "C": 0}),
        Ballot(scores={"A": 0, "B": 2, "C": 0}),
        Ballot(scores={"A": 0, "B": 0, "C": 2}),
    ]
)


states = [
    ElectionState(
        remaining=(frozenset({"A"}), frozenset({"B"}), frozenset({"C"})),
        scores={"A": 4, "B": 3, "C": 1},
    ),
    ElectionState(
        round_number=1,
        remaining=(frozenset({"C"}),),
        elected=(frozenset({"A"}), frozenset({"B"})),
        scores={"C": 1},
    ),
]


def test_init():
    e = Cumulative(profile_no_tied_cumulative, m=2)
    assert e.get_elected() == (frozenset({"A"}), frozenset({"B"}))


def test_ties():
    e_random = Cumulative(profile_tied_cumulative, m=2, tiebreak="random")
    assert len([c for s in e_random.get_elected() for c in s]) == 2


def test_state_list():
    e = Cumulative(profile_no_tied_cumulative, m=2)
    assert e.election_states == states


def test_get_profile():
    e = Cumulative(profile_no_tied_cumulative, m=2)
    assert e.get_profile(0) == profile_no_tied_cumulative
    assert e.get_profile(1) == profile_no_tied_cumulative_round_1


def test_get_step():
    e = Cumulative(profile_no_tied_cumulative, m=2)
    assert e.get_step(1) == (profile_no_tied_cumulative_round_1, states[1])


def test_get_elected():
    e = Cumulative(profile_no_tied_cumulative, m=2)
    assert e.get_elected(0) == tuple()
    assert e.get_elected(1) == (frozenset({"A"}), frozenset({"B"}))


def test_get_eliminated():
    e = Cumulative(profile_no_tied_cumulative, m=2)
    assert e.get_eliminated(0) == tuple()
    assert e.get_eliminated(1) == tuple()


def test_get_remaining():
    e = Cumulative(profile_no_tied_cumulative, m=2)
    assert e.get_remaining(0) == (frozenset({"A"}), frozenset({"B"}), frozenset({"C"}))
    assert e.get_remaining(1) == (frozenset({"C"}),)


def test_get_ranking():
    e = Cumulative(profile_no_tied_cumulative, m=2)
    assert e.get_ranking(0) == (frozenset({"A"}), frozenset({"B"}), frozenset({"C"}))
    assert e.get_ranking(1) == (frozenset({"A"}), frozenset({"B"}), frozenset({"C"}))


def test_get_status_df():
    e = Cumulative(profile_no_tied_cumulative, m=2)

    df_0 = pd.DataFrame(
        {"Status": ["Remaining"] * 3, "Round": [0] * 3},
        index=["A", "B", "C"],
    )
    df_1 = pd.DataFrame(
        {"Status": ["Elected", "Elected", "Remaining"], "Round": [1] * 3},
        index=["A", "B", "C"],
    )

    assert e.get_status_df(0).equals(df_0)
    assert e.get_status_df(1).equals(df_1)


def test_errors():
    with pytest.raises(ValueError, match="m must be positive."):
        Cumulative(profile_no_tied_cumulative, m=0)

    with pytest.raises(
        ValueError, match="Not enough candidates received votes to be elected."
    ):
        Cumulative(profile_no_tied_cumulative, m=4)

    with pytest.raises(
        ValueError,
        match="Cannot elect correct number of candidates without breaking ties.",
    ):
        Cumulative(profile_tied_cumulative, m=2)


def test_validate_profile():
    with pytest.raises(TypeError, match="violates score limit"):
        profile = PreferenceProfile(ballots=[Ballot(scores={"A": 3, "B": 4})])
        Cumulative(profile, m=2)

    with pytest.raises(TypeError, match="violates total score budget"):
        profile = PreferenceProfile(ballots=[Ballot(scores={"A": 1, "B": 1, "C": 1})])
        Cumulative(profile, m=2)

    with pytest.raises(TypeError, match="must have non-negative scores."):
        profile = PreferenceProfile(ballots=[Ballot(scores={"A": -3})])
        Cumulative(profile, m=1)

    with pytest.raises(TypeError, match="All ballots must have score dictionary."):
        profile = PreferenceProfile(ballots=[Ballot(), Ballot(scores={"A": 1, "B": 1})])
        Cumulative(profile, m=2)
