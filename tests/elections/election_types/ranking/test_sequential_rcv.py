from votekit.elections import SequentialRCV, ElectionState
from votekit import Ballot, PreferenceProfile
import pytest

# taken from STV wiki
simult_same_as_one_by_one_profile = PreferenceProfile(
    ballots=[
        Ballot(ranking=({"Orange"}, {"Pear"}), weight=3),
        Ballot(ranking=({"Pear"}, {"Strawberry"}, {"Cake"}), weight=8),
        Ballot(ranking=({"Strawberry"}, {"Orange"}, {"Pear"}), weight=1),
        Ballot(ranking=({"Cake"}, {"Chocolate"}), weight=3),
        Ballot(ranking=({"Chocolate"}, {"Cake"}, {"Burger"}), weight=1),
        Ballot(ranking=({"Burger"}, {"Chicken"}), weight=4),
        Ballot(ranking=({"Chicken"}, {"Chocolate"}, {"Burger"}), weight=3),
    ]
)

profile_list = [
    PreferenceProfile(
        ballots=[
            Ballot(ranking=({"Orange"}, {"Pear"}), weight=3),
            Ballot(ranking=({"Pear"}, {"Strawberry"}, {"Cake"}), weight=8),
            Ballot(ranking=({"Strawberry"}, {"Orange"}, {"Pear"}), weight=1),
            Ballot(ranking=({"Cake"}, {"Chocolate"}), weight=3),
            Ballot(ranking=({"Chocolate"}, {"Cake"}, {"Burger"}), weight=1),
            Ballot(ranking=({"Burger"}, {"Chicken"}), weight=4),
            Ballot(ranking=({"Chicken"}, {"Chocolate"}, {"Burger"}), weight=3),
        ]
    ),
    PreferenceProfile(
        ballots=[
            Ballot(ranking=({"Orange"},), weight=3),
            Ballot(ranking=({"Strawberry"}, {"Cake"}), weight=8),
            Ballot(ranking=({"Strawberry"}, {"Orange"}), weight=1),
            Ballot(ranking=({"Cake"}, {"Chocolate"}), weight=3),
            Ballot(ranking=({"Chocolate"}, {"Cake"}, {"Burger"}), weight=1),
            Ballot(ranking=({"Burger"}, {"Chicken"}), weight=4),
            Ballot(ranking=({"Chicken"}, {"Chocolate"}, {"Burger"}), weight=3),
        ]
    ),
    PreferenceProfile(
        ballots=[
            Ballot(ranking=({"Orange"},), weight=4),
            Ballot(ranking=({"Cake"},), weight=8),
            Ballot(ranking=({"Cake"}, {"Chocolate"}), weight=3),
            Ballot(ranking=({"Chocolate"}, {"Cake"}, {"Burger"}), weight=1),
            Ballot(ranking=({"Burger"}, {"Chicken"}), weight=4),
            Ballot(ranking=({"Chicken"}, {"Chocolate"}, {"Burger"}), weight=3),
        ]
    ),
    PreferenceProfile(
        ballots=[
            Ballot(ranking=({"Orange"},), weight=4),
            Ballot(ranking=({"Chocolate"},), weight=3),
            Ballot(ranking=({"Chocolate"}, {"Burger"}), weight=1),
            Ballot(ranking=({"Burger"}, {"Chicken"}), weight=4),
            Ballot(ranking=({"Chicken"}, {"Chocolate"}, {"Burger"}), weight=3),
        ]
    ),
]

states = [
    ElectionState(
        round_number=0,
        remaining=(
            frozenset({"Pear"}),
            frozenset({"Burger"}),
            frozenset({"Orange", "Cake", "Chicken"}),
            frozenset({"Strawberry", "Chocolate"}),
        ),
        scores={
            "Pear": 8,
            "Burger": 4,
            "Orange": 3,
            "Cake": 3,
            "Chicken": 3,
            "Strawberry": 1,
            "Chocolate": 1,
        },
    ),
    ElectionState(
        round_number=1,
        remaining=(
            frozenset({"Strawberry"}),
            frozenset({"Burger"}),
            frozenset({"Orange", "Cake", "Chicken"}),
            frozenset({"Chocolate"}),
        ),
        elected=(frozenset({"Pear"}),),
        scores={
            "Burger": 4,
            "Orange": 3,
            "Cake": 3,
            "Chicken": 3,
            "Strawberry": 9,
            "Chocolate": 1,
        },
    ),
    ElectionState(
        round_number=2,
        remaining=({"Cake"}, {"Burger", "Orange"}, {"Chicken"}, {"Chocolate"}),
        elected=({"Strawberry"},),
        scores={"Burger": 4, "Orange": 4, "Cake": 11, "Chicken": 3, "Chocolate": 1},
    ),
    ElectionState(
        round_number=3,
        remaining=({"Burger", "Orange", "Chocolate"}, {"Chicken"}),
        elected=({"Cake"},),
        scores={"Burger": 4, "Orange": 4, "Chicken": 3, "Chocolate": 4},
    ),
]


def test_init():
    e = SequentialRCV(simult_same_as_one_by_one_profile, m=3)
    assert e.get_elected() == (
        frozenset({"Pear"}),
        frozenset({"Strawberry"}),
        frozenset({"Cake"}),
    )


def test_simul_match_1by1():
    e_simul = SequentialRCV(simult_same_as_one_by_one_profile, m=3, simultaneous=True)
    e_1by1 = SequentialRCV(simult_same_as_one_by_one_profile, m=3, simultaneous=False)

    assert e_simul.get_elected() == e_1by1.get_elected()


def test_quotas():
    e = SequentialRCV(simult_same_as_one_by_one_profile, m=3, quota="droop")
    assert e.threshold == 6

    e = SequentialRCV(simult_same_as_one_by_one_profile, m=3, quota="hare")
    assert e.threshold == 7


def test_profiles():
    e = SequentialRCV(simult_same_as_one_by_one_profile, m=3)
    assert [e.get_profile(i) for i in range(len(e.election_states))] == profile_list


def test_state_list():
    e = SequentialRCV(simult_same_as_one_by_one_profile, m=3)
    assert e.election_states == states


def test_get_profile():
    e = SequentialRCV(simult_same_as_one_by_one_profile, m=3)
    assert e.get_profile(0) == simult_same_as_one_by_one_profile
    assert e.get_profile(-1) == profile_list[-1]


def test_get_step():
    e = SequentialRCV(simult_same_as_one_by_one_profile, m=3)
    assert e.get_step(-1) == (profile_list[-1], states[-1])


def test_get_elected():
    e = SequentialRCV(simult_same_as_one_by_one_profile, m=3)
    assert e.get_elected(0) == tuple()
    assert e.get_elected(1) == (frozenset({"Pear"}),)


def test_get_eliminated():
    e = SequentialRCV(simult_same_as_one_by_one_profile, m=3)
    assert e.get_eliminated(0) == tuple()
    assert e.get_eliminated(1) == tuple()


def test_get_remaining():
    e = SequentialRCV(simult_same_as_one_by_one_profile, m=3)
    assert e.get_remaining(0) == (
        frozenset({"Pear"}),
        frozenset({"Burger"}),
        frozenset({"Orange", "Cake", "Chicken"}),
        frozenset({"Strawberry", "Chocolate"}),
    )
    assert e.get_remaining(-1) == ({"Burger", "Orange", "Chocolate"}, {"Chicken"})


def test_get_ranking():
    e = SequentialRCV(simult_same_as_one_by_one_profile, m=3)
    assert e.get_ranking(0) == (
        frozenset({"Pear"}),
        frozenset({"Burger"}),
        frozenset({"Orange", "Cake", "Chicken"}),
        frozenset({"Strawberry", "Chocolate"}),
    )

    assert e.get_ranking(-1) == (
        frozenset({"Pear"}),
        frozenset({"Strawberry"}),
        frozenset({"Cake"}),
        {"Burger", "Orange", "Chocolate"},
        {"Chicken"},
    )


def test_fpv_tie():
    profile = PreferenceProfile(
        ballots=(
            Ballot(ranking=(frozenset({"A"}),)),
            Ballot(ranking=(frozenset({"B"}),)),
        ),
        candidates=("A", "B", "C"),
    )

    # A and B are tied
    e = SequentialRCV(profile, m=2, simultaneous=False, tiebreak="random")
    assert len([c for s in e.get_elected() for c in s]) == 2


def test_simul_v_1by1_():
    profile = PreferenceProfile(
        ballots=(
            Ballot(ranking=(frozenset({"A"}),)),
            Ballot(ranking=(frozenset({"B"}),)),
        ),
        candidates=("A", "B", "C"),
    )

    e_simul = SequentialRCV(profile, m=2, simultaneous=True)
    e_1by1 = SequentialRCV(profile, m=2, simultaneous=False, tiebreak="random")

    assert e_simul.election_states != e_1by1.election_states
    assert e_simul.get_remaining(1) == (frozenset(),)
    assert len(e_1by1.get_remaining(1)) == 1


def test_errors():
    with pytest.raises(
        ValueError,
        match="m must be positive.",
    ):
        SequentialRCV(simult_same_as_one_by_one_profile, m=0)

    with pytest.raises(
        ValueError,
        match="Not enough candidates received votes to be elected.",
    ):
        SequentialRCV(simult_same_as_one_by_one_profile, m=8)

    with pytest.raises(
        ValueError,
        match="Cannot elect correct number of candidates without breaking ties.",
    ):
        profile = PreferenceProfile(
            ballots=(
                Ballot(ranking=(frozenset({"A"}),)),
                Ballot(ranking=(frozenset({"B"}),)),
            ),
            candidates=("A", "B", "C"),
        )

        # A and B are tied
        SequentialRCV(profile, m=2, simultaneous=False)

    with pytest.raises(ValueError, match="Misspelled or unknown quota type."):
        SequentialRCV(
            PreferenceProfile(ballots=(Ballot(ranking=({"a"},)),)), m=1, quota="Drip"
        )

    with pytest.raises(TypeError, match="Ballots must have rankings."):
        SequentialRCV(PreferenceProfile(ballots=(Ballot(scores={"A": 4}),)))
