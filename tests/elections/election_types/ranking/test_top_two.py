from votekit.elections import TopTwo, ElectionState
from votekit.pref_profile import (
    PreferenceProfile,
    ProfileError,
)
from votekit.ballot import Ballot
import pytest
import pandas as pd

profile_no_tied_pl_no_tied_top_two = PreferenceProfile(
    ballots=[
        Ballot(ranking=({"A"}, {"B"}, {"C"})),
        Ballot(ranking=({"A"}, {"C"}, {"B"})),
        Ballot(ranking=({"B"}, {"A"}, {"C"})),
    ],
    max_ranking_length=3,
)


profile_no_tied_pl_no_tied_top_two_round_1 = PreferenceProfile(
    ballots=[Ballot(ranking=({"A"}, {"B"}), weight=2), Ballot(ranking=({"B"}, {"A"}))],
    max_ranking_length=3,
)

profile_no_tied_pl_no_tied_top_two_round_2 = PreferenceProfile(
    ballots=[Ballot(ranking=({"B"},), weight=3)],
    max_ranking_length=3,
)

profile_with_tied_pl = PreferenceProfile(
    ballots=[
        Ballot(ranking=[{"A"}, {"B"}, {"C"}]),
        Ballot(ranking=[{"B"}, {"C"}, {"A"}]),
        Ballot(ranking=[{"C"}, {"B"}, {"A"}]),
    ],
    max_ranking_length=3,
)

profile_with_tied_top_two = PreferenceProfile(
    ballots=[
        Ballot(ranking=[{"A"}, {"B"}, {"C"}]),
        Ballot(ranking=[{"B"}, {"C"}, {"A"}]),
    ],
    max_ranking_length=3,
)


no_tied_pl_no_tied_top_two_states = [
    ElectionState(
        remaining=(frozenset({"A"}), frozenset({"B"}), frozenset({"C"})),
        scores={"A": 2, "B": 1, "C": 0},
    ),
    ElectionState(
        round_number=1,
        eliminated=(frozenset({"C"}),),
        remaining=(frozenset({"A"}), frozenset({"B"})),
        scores={"A": 2, "B": 1},
    ),
    ElectionState(
        round_number=2,
        remaining=(frozenset({"B"}),),
        elected=(frozenset({"A"}),),
        scores={"B": 3},
    ),
]


def test_init():
    e = TopTwo(profile_no_tied_pl_no_tied_top_two)
    assert e.get_elected() == (frozenset({"A"}),)


def test_ties():
    e = TopTwo(profile_with_tied_pl, tiebreak="borda")

    assert e.get_remaining(1) == (frozenset({"B"}), frozenset({"C"}))
    assert e.get_elected(2) == (frozenset({"B"}),)
    assert e.election_states[1].tiebreaks == {
        (frozenset({"B", "C", "A"})): (
            frozenset({"B"}),
            frozenset({"C"}),
            frozenset({"A"}),
        )
    }

    e = TopTwo(profile_with_tied_pl, tiebreak="random")
    assert len([c for s in e.get_elected(2) for c in s]) == 1

    e = TopTwo(
        profile_with_tied_top_two, tiebreak="borda"
    )  # will perform random tie on top 2
    assert len([c for s in e.get_elected(2) for c in s]) == 1

    e = TopTwo(profile_with_tied_top_two, tiebreak="random")
    assert len([c for s in e.get_elected(2) for c in s]) == 1


def test_state_list():
    e = TopTwo(profile_no_tied_pl_no_tied_top_two)
    assert e.election_states == no_tied_pl_no_tied_top_two_states


def test_get_profile():
    e = TopTwo(profile_no_tied_pl_no_tied_top_two)
    assert e.get_profile(0) == profile_no_tied_pl_no_tied_top_two
    assert e.get_profile(1) == profile_no_tied_pl_no_tied_top_two_round_1
    assert e.get_profile(2) == profile_no_tied_pl_no_tied_top_two_round_2


def test_get_step():
    e = TopTwo(profile_no_tied_pl_no_tied_top_two)
    assert e.get_step(1) == (
        profile_no_tied_pl_no_tied_top_two_round_1,
        no_tied_pl_no_tied_top_two_states[1],
    )


def test_get_elected():
    e = TopTwo(profile_no_tied_pl_no_tied_top_two)
    assert e.get_elected(0) == tuple()
    assert e.get_elected(1) == tuple()
    assert e.get_elected(2) == (frozenset({"A"}),)


def test_get_eliminated():
    e = TopTwo(profile_no_tied_pl_no_tied_top_two)
    assert e.get_eliminated(0) == tuple()
    assert e.get_eliminated(1) == (frozenset({"C"}),)
    assert e.get_eliminated(2) == (frozenset({"C"}),)


def test_get_remaining():
    e = TopTwo(profile_no_tied_pl_no_tied_top_two)
    assert e.get_remaining(0) == (frozenset({"A"}), frozenset({"B"}), frozenset({"C"}))
    assert e.get_remaining(1) == (frozenset({"A"}), frozenset({"B"}))
    assert e.get_remaining(2) == (frozenset({"B"}),)


def test_get_ranking():
    e = TopTwo(profile_no_tied_pl_no_tied_top_two)
    assert e.get_ranking(0) == (frozenset({"A"}), frozenset({"B"}), frozenset({"C"}))
    assert e.get_ranking(1) == (frozenset({"A"}), frozenset({"B"}), frozenset({"C"}))
    assert e.get_ranking(2) == (frozenset({"A"}), frozenset({"B"}), frozenset({"C"}))


def test_get_status_df():
    e = TopTwo(profile_no_tied_pl_no_tied_top_two)

    df_0 = pd.DataFrame(
        {"Status": ["Remaining"] * 3, "Round": [0] * 3},
        index=["A", "B", "C"],
    )
    df_1 = pd.DataFrame(
        {"Status": ["Remaining", "Remaining", "Eliminated"], "Round": [1] * 3},
        index=["A", "B", "C"],
    )
    df_2 = pd.DataFrame(
        {"Status": ["Elected", "Remaining", "Eliminated"], "Round": [2, 2, 1]},
        index=["A", "B", "C"],
    )

    assert e.get_status_df(0).equals(df_0)
    assert e.get_status_df(1).equals(df_1)
    assert e.get_status_df(2).equals(df_2)


def test_errors():
    with pytest.raises(
        ValueError,
        match="Cannot elect correct number of candidates without breaking ties.",
    ):
        TopTwo(profile_with_tied_top_two)

    with pytest.raises(ProfileError, match="Profile must be of type RankProfile."):
        TopTwo(PreferenceProfile(ballots=(Ballot(scores={"A": 4, "B": 3}),)))
