from votekit.elections import ElectionState
from votekit.elections import FastSTV as STV
from votekit import Ballot, PreferenceProfile
import pandas as pd
import pytest

# taken from STV wiki
simult_same_as_one_by_one_profile = PreferenceProfile(
    ballots=tuple(
        [
            Ballot(ranking=tuple(map(frozenset, [{"Orange"}, {"Pear"}])), weight=3),
            Ballot(
                ranking=tuple(map(frozenset, [{"Pear"}, {"Strawberry"}, {"Cake"}])),
                weight=8,
            ),
            Ballot(
                ranking=tuple(map(frozenset, [{"Strawberry"}, {"Orange"}, {"Pear"}])),
                weight=1,
            ),
            Ballot(ranking=tuple(map(frozenset, [{"Cake"}, {"Chocolate"}])), weight=3),
            Ballot(
                ranking=tuple(map(frozenset, [{"Chocolate"}, {"Cake"}, {"Burger"}])),
                weight=1,
            ),
            Ballot(ranking=tuple(map(frozenset, [{"Burger"}, {"Chicken"}])), weight=4),
            Ballot(
                ranking=tuple(map(frozenset, [{"Chicken"}, {"Chocolate"}, {"Burger"}])),
                weight=3,
            ),
        ]
    ),
    max_ranking_length=3,
)

profile_list = [
    PreferenceProfile(
        ballots=tuple(
            [
                Ballot(ranking=tuple(map(frozenset, [{"Orange"}, {"Pear"}])), weight=3),
                Ballot(
                    ranking=tuple(map(frozenset, [{"Pear"}, {"Strawberry"}, {"Cake"}])),
                    weight=8,
                ),
                Ballot(
                    ranking=tuple(
                        map(frozenset, [{"Strawberry"}, {"Orange"}, {"Pear"}])
                    ),
                    weight=1,
                ),
                Ballot(
                    ranking=tuple(map(frozenset, [{"Cake"}, {"Chocolate"}])), weight=3
                ),
                Ballot(
                    ranking=tuple(
                        map(frozenset, [{"Chocolate"}, {"Cake"}, {"Burger"}])
                    ),
                    weight=1,
                ),
                Ballot(
                    ranking=tuple(map(frozenset, [{"Burger"}, {"Chicken"}])), weight=4
                ),
                Ballot(
                    ranking=tuple(
                        map(frozenset, [{"Chicken"}, {"Chocolate"}, {"Burger"}])
                    ),
                    weight=3,
                ),
            ]
        ),
        max_ranking_length=3,
    ),
    PreferenceProfile(
        ballots=tuple(
            [
                Ballot(ranking=tuple(map(frozenset, [{"Orange"}])), weight=3),
                Ballot(
                    ranking=tuple(map(frozenset, [{"Strawberry"}, {"Cake"}])), weight=2
                ),
                Ballot(
                    ranking=tuple(map(frozenset, [{"Strawberry"}, {"Orange"}])),
                    weight=1,
                ),
                Ballot(
                    ranking=tuple(map(frozenset, [{"Cake"}, {"Chocolate"}])), weight=3
                ),
                Ballot(
                    ranking=tuple(
                        map(frozenset, [{"Chocolate"}, {"Cake"}, {"Burger"}])
                    ),
                    weight=1,
                ),
                Ballot(
                    ranking=tuple(map(frozenset, [{"Burger"}, {"Chicken"}])), weight=4
                ),
                Ballot(
                    ranking=tuple(
                        map(frozenset, [{"Chicken"}, {"Chocolate"}, {"Burger"}])
                    ),
                    weight=3,
                ),
            ]
        ),
        max_ranking_length=3,
    ),
    PreferenceProfile(
        ballots=tuple(
            [
                Ballot(
                    ranking=tuple(map(frozenset, [{"Orange"}])),
                    weight=3,
                ),
                Ballot(
                    ranking=tuple(map(frozenset, [{"Strawberry"}, {"Cake"}])), weight=2
                ),
                Ballot(
                    ranking=tuple(map(frozenset, [{"Strawberry"}, {"Orange"}])),
                    weight=1,
                ),
                Ballot(ranking=tuple(map(frozenset, [{"Cake"}])), weight=3),
                Ballot(ranking=tuple(map(frozenset, [{"Cake"}, {"Burger"}])), weight=1),
                Ballot(
                    ranking=tuple(map(frozenset, [{"Burger"}, {"Chicken"}])), weight=4
                ),
                Ballot(
                    ranking=tuple(map(frozenset, [{"Chicken"}, {"Burger"}])), weight=3
                ),
            ]
        ),
        max_ranking_length=3,
    ),
    PreferenceProfile(
        ballots=tuple(
            [
                Ballot(ranking=tuple(map(frozenset, [{"Orange"}])), weight=4),
                Ballot(ranking=tuple(map(frozenset, [{"Cake"}])), weight=5),
                Ballot(ranking=tuple(map(frozenset, [{"Cake"}, {"Burger"}])), weight=1),
                Ballot(
                    ranking=tuple(map(frozenset, [{"Burger"}, {"Chicken"}])), weight=4
                ),
                Ballot(
                    ranking=tuple(map(frozenset, [{"Chicken"}, {"Burger"}])), weight=3
                ),
            ]
        ),
        max_ranking_length=3,
    ),
    PreferenceProfile(
        ballots=tuple(
            [
                Ballot(ranking=tuple(map(frozenset, [{"Orange"}])), weight=4),
                Ballot(
                    ranking=tuple(map(frozenset, [{"Burger"}, {"Chicken"}])), weight=4
                ),
                Ballot(
                    ranking=tuple(map(frozenset, [{"Chicken"}, {"Burger"}])), weight=3
                ),
            ]
        ),
        max_ranking_length=3,
    ),
    PreferenceProfile(
        ballots=tuple(
            [
                Ballot(ranking=tuple(map(frozenset, [{"Orange"}])), weight=4),
                Ballot(ranking=tuple(map(frozenset, [{"Burger"}])), weight=7),
            ]
        ),
        max_ranking_length=3,
    ),
    PreferenceProfile(
        ballots=tuple(
            [
                Ballot(ranking=tuple(map(frozenset, [{"Orange"}])), weight=4),
            ]
        ),
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
        remaining=(
            frozenset({"Burger", "Cake"}),
            frozenset({"Orange", "Chicken", "Strawberry"}),
        ),
        eliminated=(frozenset({"Chocolate"}),),
        scores={"Burger": 4, "Orange": 3, "Cake": 4, "Chicken": 3, "Strawberry": 3},
    ),
    ElectionState(
        round_number=3,
        remaining=(
            frozenset({"Cake"}),
            frozenset({"Burger", "Orange"}),
            frozenset({"Chicken"}),
        ),
        eliminated=(frozenset({"Strawberry"}),),
        scores={"Burger": 4, "Orange": 4, "Cake": 6, "Chicken": 3},
        tiebreaks={
            frozenset({"Chicken", "Strawberry", "Orange"}): (
                frozenset({"Orange"}),
                frozenset({"Chicken"}),
                frozenset({"Strawberry"}),
            )
        },
    ),
    ElectionState(
        round_number=4,
        remaining=(frozenset({"Burger", "Orange"}), frozenset({"Chicken"})),
        elected=(frozenset({"Cake"}),),
        scores={"Burger": 4, "Orange": 4, "Chicken": 3},
    ),
    ElectionState(
        round_number=5,
        remaining=(frozenset({"Burger"}), frozenset({"Orange"})),
        eliminated=(frozenset({"Chicken"}),),
        scores={"Burger": 7, "Orange": 4},
    ),
    ElectionState(
        round_number=6,
        remaining=(frozenset({"Orange"}),),
        elected=(frozenset({"Burger"}),),
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
        STV(
            PreferenceProfile(ballots=(Ballot(ranking=(frozenset({"A"}),)),)),
            m=1,
            quota="Drip",
        )

    with pytest.raises(TypeError, match="Ballots must have rankings."):
        STV(PreferenceProfile(ballots=(Ballot(scores={"A": 4}),)))


def test_stv_cands_cast():
    profile = PreferenceProfile(
        ballots=(
            Ballot(ranking=(frozenset({"A"}),), weight=4),
            Ballot(ranking=(frozenset({"B"}),), weight=2),
            Ballot(ranking=(frozenset({"C"}),), weight=5),
        ),
        candidates=("A", "B", "C", "D", "E"),
    )

    assert STV(profile, m=3).get_elected() == ({"C"}, {"A"}, {"B"})


@pytest.mark.slow
def test_stv_resolves_losing_tiebreaks_consistently_on_rerun():
    for _ in range(100):
        profile = PreferenceProfile(
            ballots=(
                Ballot(ranking=tuple(map(frozenset, [{"Orange"}, {"Pear"}])), weight=5),
                Ballot(
                    ranking=tuple(map(frozenset, [{"Pear"}, {"Strawberry"}, {"Cake"}])),
                    weight=8,
                ),
                Ballot(
                    ranking=tuple(
                        map(frozenset, [{"Strawberry"}, {"Orange"}, {"Pear"}])
                    ),
                    weight=1,
                ),
                Ballot(
                    ranking=tuple(map(frozenset, [{"Cake"}, {"Chocolate"}])), weight=3
                ),
                Ballot(
                    ranking=tuple(
                        map(frozenset, [{"Chocolate"}, {"Cake"}, {"Burger"}])
                    ),
                    weight=2,
                ),
                Ballot(
                    ranking=tuple(map(frozenset, [{"Burger"}, {"Chicken"}])), weight=4
                ),
                Ballot(
                    ranking=tuple(
                        map(frozenset, [{"Chicken"}, {"Chocolate"}, {"Burger"}])
                    ),
                    weight=4,
                ),
            ),
            max_ranking_length=3,
        )

        # There is a tiebreak between Burger and Chicken that must be resolved
        election = STV(profile, m=3)

        # The following line will error if the tiebreaks are not resolved consistently
        election.get_step(7)

@pytest.mark.slow
def test_stv_resolves_winning_tiebreaks_consistently_on_rerun():
    for _ in range(100):
        profile = PreferenceProfile(
            ballots=(
                Ballot(ranking=tuple(map(frozenset, [{"Orange"}, {"Pear"}])), weight=5),
                Ballot(
                    ranking=tuple(map(frozenset, [{"Pear"}, {"Strawberry"}, {"Cake"}])),
                    weight=10,
                ),
                Ballot(
                    ranking=tuple(
                        map(frozenset, [{"Strawberry"}, {"Orange"}, {"Pear"}])
                    ),
                    weight=1,
                ),
                Ballot(
                    ranking=tuple(map(frozenset, [{"Cake"}, {"Chocolate"}])), weight=3
                ),
                Ballot(
                    ranking=tuple(
                        map(frozenset, [{"Chocolate"}, {"Cake"}, {"Burger"}])
                    ),
                    weight=2,
                ),
                Ballot(
                    ranking=tuple(map(frozenset, [{"Burger"}, {"Chicken"}])), weight=4
                ),
                Ballot(
                    ranking=tuple(
                        map(frozenset, [{"Chicken"}, {"Chocolate"}, {"Burger"}])
                    ),
                    weight=4,
                ),
                Ballot(ranking=tuple(map(frozenset, [{"Cake"}])), weight=2),
                Ballot(ranking=tuple(map(frozenset, [{"Burger"}])), weight=6),
            ),
            max_ranking_length=3,
        )

        # There is a tiebreak between Burger and Chicken that must be resolved
        election = STV(profile, m=3, simultaneous=False, tiebreak="random")

        # The following line will error if the tiebreaks are not resolved consistently
        election_order = []
        for state in election.election_states:
            if state.elected != (frozenset(),):
                election_order.append(state.elected[0])
        new_election_order = []
        round_number = 0
        while len(new_election_order) < len(election_order):
            state = election.get_step(round_number)[1]
            if state.elected != (frozenset(),):
                new_election_order.append(state.elected[0])
            round_number += 1

        assert election_order == new_election_order


def test_random_transfers():
    # in the below profile, B always wins with Cambridge-styled random transfers,
    # but C would always win with fractional transfers, and wins with probability P > 1 - (1/2)**49 with the "fractional_random" method
    reducto_ad_absurdum = PreferenceProfile(
        ballots=(
            Ballot(ranking=(frozenset({"A"}), frozenset({"B"})), weight=50),
            Ballot(ranking=(frozenset({"A"}),), weight=150),
            Ballot(ranking=(frozenset({"B"}),), weight=24),
            Ballot(ranking=(frozenset({"C"}),), weight=73),
        ),
        candidates=("A", "B", "C"),
    )
    assert STV(
        reducto_ad_absurdum, m=2, transfer="cambridge_random"
    ).get_elected() == (
        frozenset({"A"}),
        frozenset({"B"}),
    )
    assert STV(
        reducto_ad_absurdum, m=2, transfer="fractional_random"
    ).get_elected() == (
        frozenset({"A"}),
        frozenset({"C"}),
    )
    assert STV(reducto_ad_absurdum, m=2, transfer="fractional").get_elected() == (
        frozenset({"A"}),
        frozenset({"C"}),
    )


def test_simult_not_same_as_1b1():
    profile = PreferenceProfile(
        ballots=(
            Ballot(ranking=(frozenset(["C"]),), weight=73),
            Ballot(ranking=(frozenset(["B"]), frozenset(["C"])), weight=100),
            Ballot(
                ranking=(frozenset(["A"]), frozenset(["B"]), frozenset(["D"])),
                weight=150,
            ),
            Ballot(ranking=(frozenset(["D"]),), weight=73),
        ),
        candidates=("A", "B", "C", "D"),
    )
    assert STV(profile, m=3, simultaneous=True).get_elected() == (frozenset({"A"}), frozenset({"B"}), frozenset({"D"}))
    assert STV(profile, m=3, simultaneous=False).get_elected() == (frozenset({"A"}), frozenset({"B"}), frozenset({"C"}))


def test_borda_tiebreak():
    profile_with_different_outcome_under_borda_tiebreak = PreferenceProfile(
        ballots=(
            Ballot(ranking=(frozenset(["C"]),), weight=48),
            Ballot(
                ranking=(frozenset(["B"]), frozenset(["A"]), frozenset(["D"])),
                weight=150,
            ),
            Ballot(
                ranking=(frozenset(["A"]), frozenset(["B"]), frozenset(["C"])),
                weight=150,
            ),
            Ballot(ranking=(frozenset(["D"]), frozenset(["B"])), weight=48),
        ),
        candidates=("A", "B", "C", "D"),
    )
    e = STV(profile_with_different_outcome_under_borda_tiebreak, m=3, simultaneous=False, tiebreak="borda")
    assert e.get_elected() == (
        frozenset({"B"}),
        frozenset({"A"}),
        frozenset({"C"}),
    )  # weirdge b/c C gets rewarded for listing fewer preferences
    winner_record = set()
    for _ in range(50):
        e = STV(
            profile_with_different_outcome_under_borda_tiebreak, m=3, simultaneous=False, tiebreak="random"
        )
        for c in e.get_elected():
            winner_record.add(list(c)[0])
    assert "D" in winner_record
