from votekit.elections import Alaska, ElectionState
from votekit import PreferenceProfile, Ballot
import pytest

test_profile = PreferenceProfile(
    ballots=(
        Ballot(ranking=({"A"}, {"B"}, {"C"}, {"D"})),
        Ballot(ranking=({"B"}, {"A"}, {"C"}, {"D"}), weight=2),
        Ballot(ranking=({"C"}, {"A"}, {"B"}, {"D"}), weight=2),
    )
)


test_profile_ties = PreferenceProfile(
    ballots=(
        Ballot(ranking=({"A"}, {"B"}, {"C"}, {"D"}, {"E"})),
        Ballot(ranking=({"B"}, {"A"}, {"C"}, {"D"}, {"E"})),
        Ballot(ranking=({"C"}, {"B"}, {"A"}, {"D"}, {"E"})),
        Ballot(ranking=({"D"}, {"A"}, {"B"}, {"C"}, {"E"})),
    )
)

profiles = [
    test_profile,
    PreferenceProfile(
        ballots=(
            Ballot(ranking=({"A"}, {"B"}, {"C"})),
            Ballot(ranking=({"B"}, {"A"}, {"C"}), weight=2),
            Ballot(ranking=({"C"}, {"A"}, {"B"}), weight=2),
        )
    ),
    PreferenceProfile(ballots=(Ballot(ranking=({"A"},)),)),
]

states = [
    ElectionState(
        round_number=0,
        remaining=({"B", "C"}, {"A"}, {"D"}),
        scores={"A": 1, "B": 2, "C": 2, "D": 0},
    ),
    ElectionState(
        round_number=1,
        remaining=({"B", "C"}, {"A"}),
        eliminated=({"D"},),
        scores={"A": 1, "B": 2, "C": 2},
    ),
    ElectionState(
        round_number=2, remaining=({"A"},), elected=({"B", "C"},), scores={"A": 1}
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
    print(e.election_states[2])
    print(states[2])
    assert e.election_states == states


def test_get_profile():
    e = Alaska(test_profile, m_1=3, m_2=2)
    assert [e.get_profile(i) for i in range(len(e.election_states))] == profiles


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
    with pytest.raises(ValueError):  # m_1 must be non negative
        Alaska(test_profile, m_1=0)

    with pytest.raises(ValueError):  # m_2 must be non negative
        Alaska(test_profile, m_2=0)

    with pytest.raises(ValueError):  # m_1 must be less than num cands
        Alaska(test_profile, m_1=5)

    with pytest.raises(ValueError):  # m_2 must be less than num cands after round 1
        Alaska(test_profile, m_2=5)

    with pytest.raises(ValueError):  # needs tiebreak
        Alaska(test_profile_ties, m_1=3, m_2=3)

    with pytest.raises(ValueError):  # quota str
        Alaska(test_profile_ties, quota="drip")

    with pytest.raises(TypeError):  # need rankings
        Alaska(PreferenceProfile(ballots=(Ballot(scores={"A": 4}),)))
