from votekit.elections import STV, ElectionState
from votekit import Ballot, PreferenceProfile
import pandas as pd
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
    ],
    max_ranking_length=3,
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
        ],
        max_ranking_length=3,
    ),
    PreferenceProfile(
        ballots=[
            Ballot(ranking=({"Orange"},), weight=3),
            Ballot(ranking=({"Strawberry"}, {"Cake"}), weight=2),
            Ballot(ranking=({"Strawberry"}, {"Orange"}), weight=1),
            Ballot(ranking=({"Cake"}, {"Chocolate"}), weight=3),
            Ballot(ranking=({"Chocolate"}, {"Cake"}, {"Burger"}), weight=1),
            Ballot(ranking=({"Burger"}, {"Chicken"}), weight=4),
            Ballot(ranking=({"Chicken"}, {"Chocolate"}, {"Burger"}), weight=3),
        ],
        max_ranking_length=3,
    ),
    PreferenceProfile(
        ballots=[
            Ballot(ranking=({"Orange"},), weight=3),
            Ballot(ranking=({"Strawberry"}, {"Cake"}), weight=2),
            Ballot(ranking=({"Strawberry"}, {"Orange"}), weight=1),
            Ballot(ranking=({"Cake"},), weight=3),
            Ballot(ranking=({"Cake"}, {"Burger"}), weight=1),
            Ballot(ranking=({"Burger"}, {"Chicken"}), weight=4),
            Ballot(ranking=({"Chicken"}, {"Burger"}), weight=3),
        ],
        max_ranking_length=3,
    ),
    PreferenceProfile(
        ballots=[
            Ballot(ranking=({"Orange"},), weight=4),
            Ballot(ranking=({"Cake"},), weight=5),
            Ballot(ranking=({"Cake"}, {"Burger"}), weight=1),
            Ballot(ranking=({"Burger"}, {"Chicken"}), weight=4),
            Ballot(ranking=({"Chicken"}, {"Burger"}), weight=3),
        ],
        max_ranking_length=3,
    ),
    PreferenceProfile(
        ballots=[
            Ballot(ranking=({"Orange"},), weight=4),
            Ballot(ranking=({"Burger"}, {"Chicken"}), weight=4),
            Ballot(ranking=({"Chicken"}, {"Burger"}), weight=3),
        ],
        max_ranking_length=3,
    ),
    PreferenceProfile(
        ballots=[
            Ballot(ranking=({"Orange"},), weight=4),
            Ballot(ranking=({"Burger"},), weight=7),
        ],
        max_ranking_length=3,
    ),
    PreferenceProfile(
        ballots=[
            Ballot(ranking=({"Orange"},), weight=4),
        ],
        max_ranking_length=3,
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
            frozenset({"Burger"}),
            frozenset({"Orange", "Cake", "Chicken", "Strawberry"}),
            frozenset({"Chocolate"}),
        ),
        elected=(frozenset({"Pear"}),),
        scores={
            "Burger": 4,
            "Orange": 3,
            "Cake": 3,
            "Chicken": 3,
            "Strawberry": 3,
            "Chocolate": 1,
        },
    ),
    ElectionState(
        round_number=2,
        remaining=({"Burger", "Cake"}, {"Orange", "Chicken", "Strawberry"}),
        eliminated=({"Chocolate"},),
        scores={"Burger": 4, "Orange": 3, "Cake": 4, "Chicken": 3, "Strawberry": 3},
    ),
    ElectionState(
        round_number=3,
        remaining=({"Cake"}, {"Burger", "Orange"}, {"Chicken"}),
        eliminated=({"Strawberry"},),
        scores={"Burger": 4, "Orange": 4, "Cake": 6, "Chicken": 3},
        tiebreaks={
            frozenset({"Chicken", "Strawberry", "Orange"}): (
                {"Orange"},
                {"Chicken"},
                {"Strawberry"},
            )
        },
    ),
    ElectionState(
        round_number=4,
        remaining=({"Burger", "Orange"}, {"Chicken"}),
        elected=({"Cake"},),
        scores={"Burger": 4, "Orange": 4, "Chicken": 3},
    ),
    ElectionState(
        round_number=5,
        remaining=({"Burger"}, {"Orange"}),
        eliminated=({"Chicken"},),
        scores={"Burger": 7, "Orange": 4},
    ),
    ElectionState(
        round_number=6,
        remaining=({"Orange"},),
        elected=({"Burger"},),
        scores={"Orange": 4},
    ),
]


def test_init():
    e = STV(simult_same_as_one_by_one_profile, m=3)
    assert e.get_elected() == (
        frozenset({"Pear"}),
        frozenset({"Cake"}),
        frozenset({"Burger"}),
    )


def test_simul_match_1by1():
    e_simul = STV(simult_same_as_one_by_one_profile, m=3, simultaneous=True)
    e_1by1 = STV(simult_same_as_one_by_one_profile, m=3, simultaneous=False)

    assert e_simul.get_elected() == e_1by1.get_elected()


def test_quotas():
    # e = STV(simult_same_as_one_by_one_profile, m=3, quota="droop")
    # assert e.threshold == 6

    e = STV(simult_same_as_one_by_one_profile, m=3, quota="hare")
    assert e.threshold == 7


def test_profiles():
    e = STV(simult_same_as_one_by_one_profile, m=3)
    assert [e.get_profile(i) for i in range(len(e.election_states))] == profile_list


def test_state_list():
    e = STV(simult_same_as_one_by_one_profile, m=3)
    # third state has random tiebreak resolution
    assert all(e.election_states[i] == states[i] for i in [0, 1, 2, 4, 5, 6])


def test_get_profile():
    e = STV(simult_same_as_one_by_one_profile, m=3)
    assert e.get_profile(0) == simult_same_as_one_by_one_profile
    assert e.get_profile(-1) == profile_list[-1]


def test_get_step():
    e = STV(simult_same_as_one_by_one_profile, m=3)
    assert e.get_step(-1) == (profile_list[-1], states[-1])


def test_get_elected():
    e = STV(simult_same_as_one_by_one_profile, m=3)
    assert e.get_elected(0) == tuple()
    assert e.get_elected(1) == (frozenset({"Pear"}),)


def test_get_eliminated():
    e = STV(simult_same_as_one_by_one_profile, m=3)
    assert e.get_eliminated(0) == tuple()
    assert e.get_eliminated(1) == tuple()


def test_get_remaining():
    e = STV(simult_same_as_one_by_one_profile, m=3)
    assert e.get_remaining(0) == (
        frozenset({"Pear"}),
        frozenset({"Burger"}),
        frozenset({"Orange", "Cake", "Chicken"}),
        frozenset({"Strawberry", "Chocolate"}),
    )
    assert e.get_remaining(-1) == (frozenset({"Orange"}),)


def test_get_ranking():
    e = STV(simult_same_as_one_by_one_profile, m=3)
    assert e.get_ranking(0) == (
        frozenset({"Pear"}),
        frozenset({"Burger"}),
        frozenset({"Orange", "Cake", "Chicken"}),
        frozenset({"Strawberry", "Chocolate"}),
    )
    assert e.get_ranking(-1) == (
        frozenset({"Pear"}),
        frozenset({"Cake"}),
        frozenset({"Burger"}),
        frozenset({"Orange"}),
        frozenset({"Chicken"}),
        frozenset({"Strawberry"}),
        frozenset({"Chocolate"}),
    )


def test_get_status_df():
    e = STV(simult_same_as_one_by_one_profile, m=3)
    df_final = pd.DataFrame(
        {
            "Status": [
                "Elected",
                "Elected",
                "Elected",
                "Remaining",
                "Eliminated",
                "Eliminated",
                "Eliminated",
            ],
            "Round": [1, 4, 6, 6, 5, 3, 2],
        },
        index=[
            "Pear",
            "Cake",
            "Burger",
            "Orange",
            "Chicken",
            "Strawberry",
            "Chocolate",
        ],
    )

    assert e.get_status_df(-1).equals(df_final)


def test_fpv_tie():
    profile = PreferenceProfile(
        ballots=(
            Ballot(ranking=(frozenset({"A"}),)),
            Ballot(ranking=(frozenset({"B"}),)),
        ),
        candidates=("A", "B", "C"),
    )

    # A and B are tied
    e = STV(profile, m=2, simultaneous=False, tiebreak="random")
    assert len([c for s in e.get_elected() for c in s]) == 2


def test_simul_v_1by1_():
    profile = PreferenceProfile(
        ballots=(
            Ballot(ranking=(frozenset({"A"}),)),
            Ballot(ranking=(frozenset({"B"}),)),
        ),
        candidates=("A", "B", "C"),
    )

    e_simul = STV(profile, m=2, simultaneous=True)
    e_1by1 = STV(profile, m=2, simultaneous=False, tiebreak="random")

    assert e_simul.election_states != e_1by1.election_states
    assert e_simul.get_remaining(1) == (frozenset(),)
    assert len(e_1by1.get_remaining(1)) == 1


def test_errors():
    with pytest.raises(
        ValueError,
        match="m must be positive.",
    ):
        STV(simult_same_as_one_by_one_profile, m=0)

    with pytest.raises(
        ValueError,
        match="Not enough candidates received votes to be elected.",
    ):
        STV(simult_same_as_one_by_one_profile, m=8)

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
        STV(profile, m=2, simultaneous=False)

    with pytest.raises(ValueError, match="Misspelled or unknown quota type."):
        STV(PreferenceProfile(ballots=(Ballot(ranking=({"A"},)),)), m=1, quota="Drip")

    with pytest.raises(TypeError, match="Ballots must have rankings."):
        STV(PreferenceProfile(ballots=(Ballot(scores={"A": 4}),)))


def test_stv_cands_cast():
    profile = PreferenceProfile(
        ballots=(
            Ballot(ranking=({"A"},), weight=4),
            Ballot(ranking=({"B"},), weight=2),
            Ballot(ranking=({"C"},), weight=5),
        ),
        candidates=["A", "B", "C", "D", "E"],
    )

    assert STV(profile, m=3).get_elected() == ({"C"}, {"A"}, {"B"})
