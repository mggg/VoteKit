from typing import cast

import pandas as pd
import pytest

from votekit.ballot import RankBallot, ScoreBallot
from votekit.elections import Borda, ElectionState
from votekit.pref_profile import (
    ProfileError,
    RankProfile,
    ScoreProfile,
)

profile_no_tied_borda = RankProfile(
    ballots=[
        RankBallot(ranking=({"A"}, {"B"}, {"C"})),
        RankBallot(ranking=({"A"}, {"C"}, {"B"})),
        RankBallot(ranking=({"B"}, {"A"}, {"C"})),
    ],
    max_ranking_length=3,
)
# 8, 6, 4
# 3, 2, 1

profile_no_tied_borda_round_1 = RankProfile(
    ballots=[RankBallot(ranking=({"C"},), weight=3)],
    max_ranking_length=3,
)

profile_with_tied_borda = RankProfile(
    ballots=[
        RankBallot(ranking=[{"A"}, {"B"}, {"C"}, {"D"}]),
        RankBallot(ranking=[{"A"}, {"C"}, {"B"}, {"D"}]),
        RankBallot(ranking=[{"B"}, {"A"}, {"C"}, {"D"}]),
        RankBallot(ranking=[{"A"}, {"C"}, {"D"}, {"B"}]),
    ],
    max_ranking_length=4,
)


states = [
    ElectionState(
        remaining=(frozenset({"A"}), frozenset({"B"}), frozenset({"C"})),
        scores={"A": 8, "B": 6, "C": 4},
    ),
    ElectionState(
        round_number=1,
        remaining=(frozenset({"C"}),),
        elected=(frozenset({"A"}), frozenset({"B"})),
        scores={"C": 9},
    ),
]


def test_init():
    e = Borda(profile_no_tied_borda)
    assert e.get_elected() == (frozenset({"A"}),)


def test_alt_score_vector():
    e = Borda(profile_no_tied_borda, n_seats=2, score_vector=(1, 1, 0))
    assert e.get_ranking() == (frozenset({"A"}), frozenset({"B"}), frozenset({"C"}))
    assert e.election_states[0].scores == {
        "A": 3,
        "B": 2,
        "C": 1,
    }


def test_multiwinner_ties():
    _ = Borda(profile_with_tied_borda, n_seats=2, tiebreak="random")
    e_borda = Borda(profile_with_tied_borda, n_seats=2, tiebreak="first_place")

    assert e_borda.election_states[1].tiebreaks == {
        frozenset({"B", "C"}): (frozenset({"B"}), frozenset({"C"}))
    }
    assert e_borda.get_ranking(0) == (
        frozenset({"A"}),
        frozenset({"B", "C"}),
        frozenset({"D"}),
    )

    assert e_borda.get_ranking(1) == (
        frozenset({"A"}),
        frozenset({"B"}),
        frozenset({"C"}),
        frozenset({"D"}),
    )


def test_state_list():
    e = Borda(profile_no_tied_borda, n_seats=2)
    assert e.election_states == states


def test_get_profile():
    e = Borda(profile_no_tied_borda, n_seats=2)
    assert e.get_profile(0) == profile_no_tied_borda
    assert e.get_profile(1) == profile_no_tied_borda_round_1


def test_get_step():
    e = Borda(profile_no_tied_borda, n_seats=2)
    assert e.get_step(1) == (profile_no_tied_borda_round_1, states[1])


def test_get_elected():
    e = Borda(profile_no_tied_borda, n_seats=2)
    assert e.get_elected(0) == tuple()
    assert e.get_elected(1) == (frozenset({"A"}), frozenset({"B"}))


def test_get_eliminated():
    e = Borda(profile_no_tied_borda, n_seats=2)
    assert e.get_eliminated(0) == tuple()
    assert e.get_eliminated(1) == tuple()


def test_get_remaining():
    e = Borda(profile_no_tied_borda, n_seats=2)
    assert e.get_remaining(0) == (frozenset({"A"}), frozenset({"B"}), frozenset({"C"}))
    assert e.get_remaining(1) == (frozenset({"C"}),)


def test_get_ranking():
    e = Borda(profile_no_tied_borda, n_seats=2)
    assert e.get_ranking(0) == (frozenset({"A"}), frozenset({"B"}), frozenset({"C"}))
    assert e.get_ranking(1) == (frozenset({"A"}), frozenset({"B"}), frozenset({"C"}))


def test_get_status_df():
    e = Borda(profile_no_tied_borda, n_seats=2)

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
    with pytest.raises(ValueError, match="n_seats must be strictly positive"):
        Borda(profile_no_tied_borda, n_seats=0)

    with pytest.raises(ValueError, match="Not enough candidates received votes to be elected."):
        Borda(profile_no_tied_borda, n_seats=4)

    with pytest.raises(
        ValueError,
        match="Cannot elect correct number of candidates without breaking ties.",
    ):
        Borda(profile_with_tied_borda, n_seats=2)

    with pytest.raises(ProfileError, match="Profile must be of type RankProfile."):
        Borda(cast(RankProfile, ScoreProfile(ballots=(ScoreBallot(scores={"A": 4}),))))
