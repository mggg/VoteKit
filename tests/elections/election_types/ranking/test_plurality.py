from votekit.elections import Plurality, ElectionState, SNTV
from votekit.pref_profile import (
    PreferenceProfile,
    ProfileError,
)
from votekit.ballot import Ballot
import pytest
import pandas as pd

profile_no_tied_fpv = PreferenceProfile(
    ballots=[
        Ballot(ranking=({"A"}, {"B"}, {"C"})),
        Ballot(ranking=({"A"}, {"C"}, {"B"})),
        Ballot(ranking=({"B"}, {"A"}, {"C"})),
    ],
    max_ranking_length=3,
)

profile_no_tied_fpv_round_1 = PreferenceProfile(
    ballots=[Ballot(ranking=({"C"},), weight=3)],
    max_ranking_length=3,
)

profile_with_tied_fpv = PreferenceProfile(
    ballots=[
        Ballot(ranking=[{"A"}, {"B"}, {"C"}, {"D"}, {"E"}], weight=4),
        Ballot(ranking=[{"B"}, {"A"}, {"C"}, {"D"}, {"E"}], weight=3),
        Ballot(ranking=[{"C"}, {"B"}, {"A"}, {"D"}, {"E"}], weight=2),
        Ballot(ranking=[{"D"}, {"B"}, {"C"}, {"A"}, {"E"}], weight=2),
    ],
    max_ranking_length=5,
)

states = [
    ElectionState(
        remaining=(frozenset({"A"}), frozenset({"B"}), frozenset({"C"})),
        scores={"A": 2, "B": 1, "C": 0},
    ),
    ElectionState(
        round_number=1,
        remaining=(frozenset({"C"}),),
        elected=(frozenset({"A"}), frozenset({"B"})),
        scores={"C": 3},
    ),
]


def test_init():
    e = Plurality(profile_no_tied_fpv)
    assert e.get_elected() == (frozenset({"A"}),)


def test_multiwinner_ties():
    e_random = Plurality(profile_with_tied_fpv, m=3, tiebreak="random")
    e_borda = Plurality(profile_with_tied_fpv, m=3, tiebreak="borda")

    assert e_borda.election_states[1].tiebreaks == {
        frozenset({"C", "D"}): (frozenset({"C"}), frozenset({"D"}))
    }
    assert e_borda.get_ranking(0) == (
        frozenset({"A"}),
        frozenset({"B"}),
        frozenset({"C", "D"}),
        frozenset({"E"}),
    )

    assert e_borda.get_ranking(1) == (
        frozenset({"A"}),
        frozenset({"B"}),
        frozenset({"C"}),
        frozenset({"D"}),
        frozenset({"E"}),
    )

    assert len(e_random.get_elected()) == 3


def test_state_list():
    e = Plurality(profile_no_tied_fpv, m=2)
    assert e.election_states == states


def test_get_profile():
    e = Plurality(profile_no_tied_fpv, m=2)
    assert e.get_profile(0) == profile_no_tied_fpv
    assert e.get_profile(1) == profile_no_tied_fpv_round_1


def test_get_step():
    e = Plurality(profile_no_tied_fpv, m=2)
    assert e.get_step(1) == (profile_no_tied_fpv_round_1, states[1])


def test_get_elected():
    e = Plurality(profile_no_tied_fpv, m=2)
    assert e.get_elected(0) == tuple()
    assert e.get_elected(1) == (frozenset({"A"}), frozenset({"B"}))


def test_get_eliminated():
    e = Plurality(profile_no_tied_fpv, m=2)
    assert e.get_eliminated(0) == tuple()
    assert e.get_eliminated(1) == tuple()


def test_get_remaining():
    e = Plurality(profile_no_tied_fpv, m=2)
    assert e.get_remaining(0) == (frozenset({"A"}), frozenset({"B"}), frozenset({"C"}))
    assert e.get_remaining(1) == (frozenset({"C"}),)


def test_get_ranking():
    e = Plurality(profile_no_tied_fpv, m=2)
    assert e.get_ranking(0) == (frozenset({"A"}), frozenset({"B"}), frozenset({"C"}))
    assert e.get_ranking(1) == (frozenset({"A"}), frozenset({"B"}), frozenset({"C"}))


def test_get_status_df():
    e = Plurality(profile_no_tied_fpv, m=2)

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
    with pytest.raises(ValueError, match="m must be strictly positive"):
        Plurality(profile_no_tied_fpv, m=0)

    with pytest.raises(
        ValueError, match="Not enough candidates received votes to be elected."
    ):
        Plurality(profile_no_tied_fpv, m=4)

    with pytest.raises(
        ValueError,
        match="Cannot elect correct number of candidates without breaking ties.",
    ):
        Plurality(profile_with_tied_fpv, m=3)

    with pytest.raises(ProfileError, match="Profile must be of type RankProfile."):
        Plurality(PreferenceProfile(ballots=(Ballot(scores={"A": 4}),)))


def test_SNTV_Wrapper():
    e = SNTV(profile_no_tied_fpv)
    assert e.get_elected() == (frozenset({"A"}),)
