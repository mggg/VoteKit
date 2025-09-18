from votekit.elections import DominatingSets, ElectionState
from votekit.pref_profile import (
    PreferenceProfile,
    ProfileError,
)
from votekit.ballot import Ballot
import pytest

profile_no_tied_dominating_sets = PreferenceProfile(
    ballots=[
        Ballot(ranking=({"A"}, {"B"}, {"C"})),
        Ballot(ranking=({"A"}, {"C"}, {"B"})),
        Ballot(ranking=({"B"}, {"A"}, {"C"})),
    ],
    max_ranking_length=3,
)


profile_no_tied_dominating_sets_round_1 = PreferenceProfile(
    ballots=[
        Ballot(
            ranking=(
                {"B"},
                {"C"},
            ),
            weight=2,
        ),
        Ballot(
            ranking=(
                {"C"},
                {"B"},
            )
        ),
    ],
    max_ranking_length=3,
)

profile_multiwinner_dominating_sets = PreferenceProfile(
    ballots=[
        Ballot(ranking=({"A"}, {"B"}, {"C"})),
        Ballot(ranking=({"A"}, {"C"}, {"B"})),
        Ballot(ranking=({"B"}, {"A"}, {"C"}), weight=2),
    ],
    max_ranking_length=3,
)


states = [
    ElectionState(
        remaining=(frozenset({"A", "B", "C"}),),
    ),
    ElectionState(
        round_number=1,
        remaining=(
            frozenset({"B"}),
            frozenset({"C"}),
        ),
        elected=(frozenset({"A"}),),
    ),
]


def test_init():
    e = DominatingSets(profile_no_tied_dominating_sets)
    assert e.get_elected() == (frozenset({"A"}),)


def test_multiwinner():
    e = DominatingSets(profile_multiwinner_dominating_sets)

    assert e.get_ranking(0) == (frozenset({"A", "B", "C"}),)

    assert e.get_ranking(1) == (
        frozenset({"A", "B"}),
        frozenset({"C"}),
    )


def test_state_list():
    e = DominatingSets(profile_no_tied_dominating_sets)
    assert e.election_states == states


def test_get_profile():
    e = DominatingSets(profile_no_tied_dominating_sets)
    assert e.get_profile(0) == profile_no_tied_dominating_sets
    assert e.get_profile(1) == profile_no_tied_dominating_sets_round_1


def test_get_step():
    e = DominatingSets(profile_no_tied_dominating_sets)
    assert e.get_step(1) == (profile_no_tied_dominating_sets_round_1, states[1])


def test_get_elected():
    e = DominatingSets(profile_no_tied_dominating_sets)
    assert e.get_elected(0) == tuple()
    assert e.get_elected(1) == (frozenset({"A"}),)


def test_get_eliminated():
    e = DominatingSets(profile_no_tied_dominating_sets)
    assert e.get_eliminated(0) == tuple()
    assert e.get_eliminated(1) == tuple()


def test_get_remaining():
    e = DominatingSets(profile_no_tied_dominating_sets)
    assert e.get_remaining(0) == (frozenset({"A", "B", "C"}),)
    assert e.get_remaining(1) == (frozenset({"B"}), frozenset({"C"}))


def test_get_ranking():
    e = DominatingSets(profile_no_tied_dominating_sets)
    assert e.get_ranking(0) == (frozenset({"A", "B", "C"}),)
    assert e.get_ranking(1) == (frozenset({"A"}), frozenset({"B"}), frozenset({"C"}))


def test_errors():
    with pytest.raises(ProfileError, match="Profile must be of type RankProfile."):
        DominatingSets(PreferenceProfile(ballots=(Ballot(scores={"A": 4}),)))
