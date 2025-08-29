from votekit.elections import Alaska, ElectionState
from votekit import PreferenceProfile, Ballot
import pytest

test_profile = PreferenceProfile(
    ballots=(
        Ballot(ranking=({"A"}, {"B"}, {"C"}, {"D"})),
        Ballot(ranking=({"B"}, {"A"}, {"C"}, {"D"}), weight=2),
        Ballot(ranking=({"C"}, {"A"}, {"B"}, {"D"}), weight=2),
    ),
    max_ranking_length=4,
)


test_profile_ties = PreferenceProfile(
    ballots=(
        Ballot(ranking=({"A"}, {"B"}, {"C"}, {"D"}, {"E"})),
        Ballot(ranking=({"B"}, {"A"}, {"C"}, {"D"}, {"E"})),
        Ballot(ranking=({"C"}, {"B"}, {"A"}, {"D"}, {"E"})),
        Ballot(ranking=({"D"}, {"A"}, {"B"}, {"C"}, {"E"})),
    ),
    max_ranking_length=5,
)

profiles = [
    test_profile,
    PreferenceProfile(
        ballots=(
            Ballot(ranking=({"A"}, {"B"}, {"C"})),
            Ballot(ranking=({"B"}, {"A"}, {"C"}), weight=2),
            Ballot(ranking=({"C"}, {"A"}, {"B"}), weight=2),
        ),
        max_ranking_length=4,
    ),
    PreferenceProfile(
        ballots=(Ballot(ranking=({"A"},)),),
        max_ranking_length=4,
    ),
]

states = [
    ElectionState(
        round_number=0,
        remaining=(frozenset({"B", "C"}), frozenset({"A"}), frozenset({"D"})),
        scores={"A": 1, "B": 2, "C": 2, "D": 0},
    ),
    ElectionState(
        round_number=1,
        remaining=(frozenset({"B", "C"}), frozenset({"A"})),
        eliminated=(frozenset({"D"}),),
        scores={"A": 1, "B": 2, "C": 2},
    ),
    ElectionState(
        round_number=2,
        remaining=(frozenset({"A"}),),
        elected=(frozenset({"B", "C"}),),
        scores={"A": 1},
    ),
]


def test_init():
    e = Alaska(test_profile, m_1=3, m_2=2)
    assert e.get_elected() == (frozenset({"B", "C"}),)


def test_ties():
    e_random = Alaska(test_profile_ties, m_1=3, m_2=3, tiebreak="random")
    e_borda = Alaska(test_profile_ties, m_1=3, m_2=3, tiebreak="borda")

    assert len([c for s in e_borda.get_remaining(1) for c in s]) == 3
    assert len([c for s in e_random.get_remaining(1) for c in s]) == 3
    assert len(e_random.election_states[1].tiebreaks) > 0
    assert len(e_borda.election_states[1].tiebreaks) > 0


def test_state_list():
    e = Alaska(test_profile, m_1=3, m_2=2)
    assert e.election_states == states


def test_get_profile():
    e = Alaska(test_profile, m_1=3, m_2=2)
    for i in range(len(e.election_states)):
        if i == 2:
            print(e.get_profile(2).df.to_string())
            print(profiles[2].df.to_string())
        assert e.get_profile(i) == profiles[i]
        print(i)
    # assert [e.get_profile(i) for i in range(len(e.election_states))] == profiles


def test_get_step():
    e = Alaska(test_profile, m_1=3, m_2=2)
    assert e.get_step(1) == (profiles[1], states[1])


def test_get_elected():
    e = Alaska(test_profile, m_1=3, m_2=2)
    assert e.get_elected(0) == tuple()
    assert e.get_elected(-1) == (frozenset({"B", "C"}),)


def test_get_eliminated():
    e = Alaska(test_profile, m_1=3, m_2=2)
    assert e.get_eliminated(0) == tuple()
    assert e.get_eliminated(1) == ({"D"},)


def test_get_remaining():
    e = Alaska(test_profile, m_1=3, m_2=2)
    assert e.get_remaining(0) == (
        frozenset({"C", "B"}),
        frozenset({"A"}),
        frozenset({"D"}),
    )
    assert e.get_remaining(1) == (frozenset({"C", "B"}), {"A"})


def test_get_ranking():
    e = Alaska(test_profile, m_1=3, m_2=2)
    assert e.get_ranking(0) == (
        frozenset({"C", "B"}),
        frozenset({"A"}),
        frozenset({"D"}),
    )
    assert e.get_ranking(-1) == (
        frozenset({"C", "B"}),
        frozenset({"A"}),
        frozenset({"D"}),
    )


def test_errors():
    with pytest.raises(ValueError, match="m_1 must be positive."):
        Alaska(test_profile, m_1=0)

    with pytest.raises(ValueError, match="m_2 must be positive."):
        Alaska(test_profile, m_2=0)

    with pytest.raises(
        ValueError, match="Not enough candidates received votes to be elected."
    ):
        Alaska(test_profile, m_1=5)

    with pytest.raises(ValueError, match="m_1 must be greater than or equal to m_2."):
        Alaska(test_profile, m_2=5)

    with pytest.raises(
        ValueError,
        match="Cannot elect correct number of candidates without breaking ties.",
    ):
        Alaska(test_profile_ties, m_1=3, m_2=3)

    with pytest.raises(ValueError, match="Misspelled or unknown quota type."):
        Alaska(test_profile, quota="drip")

    with pytest.raises(TypeError, match="has no ranking."):
        Alaska(PreferenceProfile(ballots=(Ballot(scores={"A": 4}),)))
