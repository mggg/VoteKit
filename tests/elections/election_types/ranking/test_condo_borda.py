from votekit.elections import CondoBorda, ElectionState
from votekit import PreferenceProfile, Ballot
import pytest
import pandas as pd
from fractions import Fraction

profile_tied_set = PreferenceProfile(
    ballots=[
        Ballot(ranking=({"A"}, {"B"}, {"C"})),
        Ballot(ranking=({"A"}, {"C"}, {"B"})),
        Ballot(ranking=({"B"}, {"A"}, {"C"}), weight=2),
    ]
)

profile_tied_set_round_1 = PreferenceProfile(
    ballots=[
        Ballot(
            ranking=(
                {"B"},
                {"C"},
            ),
            weight=3,
        ),
        Ballot(
            ranking=(
                {"C"},
                {"B"},
            )
        ),
    ]
)


profile_no_tied_set = PreferenceProfile(
    ballots=[
        Ballot(ranking=({"A"}, {"B"}, {"C"})),
        Ballot(ranking=({"A"}, {"C"}, {"B"})),
        Ballot(ranking=({"B"}, {"A"}, {"C"})),
    ]
)

profile_tied_borda = PreferenceProfile(
    ballots=[
        Ballot(ranking=({"A"}, {"B"}, {"C"})),
        Ballot(ranking=({"A"}, {"C"}, {"B"})),
        Ballot(ranking=({"B"}, {"A"}, {"C"})),
        Ballot(ranking=({"B"}, {"C"}, {"A"})),
        Ballot(ranking=({"C"}, {"A"}, {"B"})),
        Ballot(ranking=({"C"}, {"B"}, {"A"})),
    ]
)


states = [
    ElectionState(
        remaining=(frozenset({"A"}), frozenset({"B"}), frozenset({"C"})),
        scores={"A": Fraction(10), "B": Fraction(9), "C": Fraction(5)},
    ),
    ElectionState(
        round_number=1,
        remaining=(frozenset({"B"}), frozenset({"C"})),
        elected=(frozenset({"A"}),),
        scores={"B": Fraction(7), "C": Fraction(5)},
        tiebreaks={frozenset({"A", "B"}): (frozenset({"A"}), frozenset({"B"}))},
    ),
]


def test_init():
    e = CondoBorda(profile_no_tied_set)
    assert e.get_elected() == (frozenset({"A"}),)


def test_multiwinner():
    e = CondoBorda(profile_no_tied_set, m=2)

    assert e.get_ranking(0) == (frozenset({"A"}), frozenset({"B"}), frozenset({"C"}))

    assert e.get_elected(1) == (
        frozenset({"A"}),
        frozenset({"B"}),
    )

    e = CondoBorda(profile_no_tied_set, m=3)

    assert e.get_ranking(0) == (frozenset({"A"}), frozenset({"B"}), frozenset({"C"}))

    assert e.get_elected(1) == (frozenset({"A"}), frozenset({"B"}), frozenset({"C"}))


def test_tied_borda_scores():
    e = CondoBorda(profile_tied_borda)
    assert len([c for s in e.get_elected() for c in s]) == 1


def test_state_list():
    e = CondoBorda(profile_tied_set)
    assert e.election_states == states


def test_get_profile():
    e = CondoBorda(profile_tied_set)
    assert e.get_profile(0) == profile_tied_set
    assert e.get_profile(1) == profile_tied_set_round_1


def test_get_step():
    e = CondoBorda(profile_tied_set)
    assert e.get_step(1) == (profile_tied_set_round_1, states[1])


def test_get_elected():
    e = CondoBorda(profile_tied_set)
    assert e.get_elected(0) == tuple()
    assert e.get_elected(1) == (frozenset({"A"}),)


def test_get_eliminated():
    e = CondoBorda(profile_tied_set)
    assert e.get_eliminated(0) == tuple()
    assert e.get_eliminated(1) == tuple()


def test_get_remaining():
    e = CondoBorda(profile_tied_set)
    assert e.get_remaining(0) == (frozenset({"A"}), frozenset({"B"}), frozenset({"C"}))
    assert e.get_remaining(1) == (
        frozenset({"B"}),
        frozenset({"C"}),
    )


def test_get_ranking():
    e = CondoBorda(profile_tied_set)
    assert e.get_ranking(0) == (frozenset({"A"}), frozenset({"B"}), frozenset({"C"}))
    assert e.get_ranking(1) == (frozenset({"A"}), frozenset({"B"}), frozenset({"C"}))


def test_get_status_df():
    e = CondoBorda(profile_tied_set)

    df_0 = pd.DataFrame(
        {"Status": ["Remaining"] * 3, "Round": [0] * 3},
        index=["A", "B", "C"],
    )
    df_1 = pd.DataFrame(
        {"Status": ["Elected", "Remaining", "Remaining"], "Round": [1] * 3},
        index=["A", "B", "C"],
    )

    assert e.get_status_df(0).equals(df_0)
    assert e.get_status_df(1).equals(df_1)


def test_errors():
    with pytest.raises(ValueError, match="m must be strictly positive"):
        CondoBorda(profile_tied_set, m=0)

    with pytest.raises(
        ValueError, match="m must be no more than the number of candidates."
    ):
        CondoBorda(profile_tied_set, m=4)

    with pytest.raises(TypeError, match="has no ranking."):
        CondoBorda(PreferenceProfile(ballots=(Ballot(scores={"A": 4}),)))
