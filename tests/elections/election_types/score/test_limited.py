from votekit.elections import Limited, ElectionState
from votekit import PreferenceProfile, Ballot
import pytest
import pandas as pd

profile_no_tied_limited = PreferenceProfile(
    ballots=[
        Ballot(scores={"A": 1, "B": 1, "C": 0}),
        Ballot(scores={"A": 2, "B": 0, "C": 0}),
        Ballot(scores={"A": 0, "B": 1, "C": 1}),
    ],
    candidates=["A", "B", "C", "D"],
)
# 3,2,1,0


profile_no_tied_limited_round_1 = PreferenceProfile(candidates=["D"])

profile_tied_limited = PreferenceProfile(
    ballots=[
        Ballot(scores={"A": 1, "B": 1, "C": 0}),
        Ballot(scores={"A": 0, "B": 1, "C": 1}),
        Ballot(scores={"A": 1, "B": 0, "C": 1}),
        Ballot(scores={"D": 2}),
    ],
)


states = [
    ElectionState(
        remaining=(
            frozenset({"A"}),
            frozenset({"B"}),
            frozenset({"C"}),
            frozenset({"D"}),
        ),
        scores={"A": 3, "B": 2, "C": 1, "D": 0},
    ),
    ElectionState(
        round_number=1,
        remaining=(frozenset({"D"}),),
        elected=(frozenset({"A"}), frozenset({"B"}), frozenset({"C"})),
        scores={"D": 0},
    ),
]


def test_init():
    e = Limited(profile_no_tied_limited, m=3, k=2)
    assert e.get_elected() == (frozenset({"A"}), frozenset({"B"}), frozenset({"C"}))


def test_ties():
    e_random = Limited(profile_tied_limited, m=3, k=2, tiebreak="random")
    assert len([c for s in e_random.get_elected() for c in s]) == 3


def test_state_list():
    e = Limited(profile_no_tied_limited, m=3, k=2)
    assert e.election_states == states


def test_get_profile():
    e = Limited(profile_no_tied_limited, m=3, k=2)
    assert e.get_profile(0) == profile_no_tied_limited
    assert e.get_profile(1) == profile_no_tied_limited_round_1


def test_get_step():
    e = Limited(profile_no_tied_limited, m=3, k=2)
    assert e.get_step(1) == (profile_no_tied_limited_round_1, states[1])


def test_get_elected():
    e = Limited(profile_no_tied_limited, m=3, k=2)
    assert e.get_elected(0) == tuple()
    assert e.get_elected(1) == (frozenset({"A"}), frozenset({"B"}), frozenset({"C"}))


def test_get_eliminated():
    e = Limited(profile_no_tied_limited, m=3, k=2)
    assert e.get_eliminated(0) == tuple()
    assert e.get_eliminated(1) == tuple()


def test_get_remaining():
    e = Limited(profile_no_tied_limited, m=3, k=2)
    assert e.get_remaining(0) == (
        frozenset({"A"}),
        frozenset({"B"}),
        frozenset({"C"}),
        frozenset({"D"}),
    )
    assert e.get_remaining(1) == (frozenset({"D"}),)


def test_get_ranking():
    e = Limited(profile_no_tied_limited, m=3, k=2)
    assert e.get_ranking(0) == (
        frozenset({"A"}),
        frozenset({"B"}),
        frozenset({"C"}),
        frozenset({"D"}),
    )
    assert e.get_ranking(1) == (
        frozenset({"A"}),
        frozenset({"B"}),
        frozenset({"C"}),
        frozenset({"D"}),
    )


def test_get_status_df():
    e = Limited(profile_no_tied_limited, m=3, k=2)

    df_0 = pd.DataFrame(
        {"Status": ["Remaining"] * 4, "Round": [0] * 4},
        index=["A", "B", "C", "D"],
    )
    df_1 = pd.DataFrame(
        {"Status": ["Elected", "Elected", "Elected", "Remaining"], "Round": [1] * 4},
        index=["A", "B", "C", "D"],
    )

    assert e.get_status_df(0).equals(df_0)
    assert e.get_status_df(1).equals(df_1)


def test_errors():
    with pytest.raises(ValueError, match="m must be positive."):
        Limited(profile_no_tied_limited, m=0, k=0)

    with pytest.raises(
        ValueError, match="Not enough candidates received votes to be elected."
    ):
        Limited(profile_no_tied_limited, m=5, k=2)

    with pytest.raises(
        ValueError,
        match="Cannot elect correct number of candidates without breaking ties.",
    ):
        Limited(profile_tied_limited, m=3, k=2)

    with pytest.raises(ValueError, match="k must be less than or equal to m."):
        Limited(profile_tied_limited, m=2, k=3)


def test_validate_profile():
    with pytest.raises(TypeError, match="violates score limit"):
        profile = PreferenceProfile(ballots=[Ballot(scores={"A": 3, "B": 2})])
        Limited(profile, m=2, k=2)

    with pytest.raises(TypeError, match="violates total score budget"):
        profile = PreferenceProfile(ballots=[Ballot(scores={"A": 1, "B": 1, "C": 1})])
        Limited(profile, m=2, k=2)

    with pytest.raises(TypeError, match="must have non-negative scores."):
        profile = PreferenceProfile(ballots=[Ballot(scores={"A": -3})])
        Limited(profile, m=1)

    with pytest.raises(TypeError, match="All ballots must have score dictionary."):
        profile = PreferenceProfile(ballots=[Ballot(), Ballot(scores={"A": 1, "B": 1})])
        Limited(profile, m=2)
