from votekit.elections import IRV, ElectionState
from votekit.pref_profile import RankProfile, ScoreProfile, ProfileError
from votekit.ballot import RankBallot, ScoreBallot
import pandas as pd
import pytest

# taken from STV wiki
test_profile = RankProfile(
    ballots=[
        RankBallot(ranking=({"Orange"}, {"Pear"}), weight=3),
        RankBallot(ranking=({"Pear"}, {"Strawberry"}, {"Cake"}), weight=8),
        RankBallot(ranking=({"Strawberry"}, {"Orange"}, {"Pear"}), weight=1 / 2),
        RankBallot(ranking=({"Cake"}, {"Chocolate"}), weight=3),
        RankBallot(ranking=({"Chocolate"}, {"Cake"}, {"Burger"}), weight=1),
        RankBallot(ranking=({"Burger"}, {"Chicken"}), weight=4),
        RankBallot(ranking=({"Chicken"}, {"Chocolate"}, {"Burger"}), weight=3),
    ],
    max_ranking_length=3,
)

profile_list = [
    RankProfile(
        ballots=[
            RankBallot(ranking=({"Orange"}, {"Pear"}), weight=3),
            RankBallot(ranking=({"Pear"}, {"Strawberry"}, {"Cake"}), weight=8),
            RankBallot(ranking=({"Strawberry"}, {"Orange"}, {"Pear"}), weight=1 / 2),
            RankBallot(ranking=({"Cake"}, {"Chocolate"}), weight=3),
            RankBallot(ranking=({"Chocolate"}, {"Cake"}, {"Burger"}), weight=1),
            RankBallot(ranking=({"Burger"}, {"Chicken"}), weight=4),
            RankBallot(ranking=({"Chicken"}, {"Chocolate"}, {"Burger"}), weight=3),
        ],
        max_ranking_length=3,
    ),
    RankProfile(
        ballots=[
            RankBallot(ranking=({"Orange"}, {"Pear"}), weight=3.5),
            RankBallot(ranking=({"Pear"}, {"Cake"}), weight=8),
            RankBallot(ranking=({"Cake"}, {"Chocolate"}), weight=3),
            RankBallot(ranking=({"Chocolate"}, {"Cake"}, {"Burger"}), weight=1),
            RankBallot(ranking=({"Burger"}, {"Chicken"}), weight=4),
            RankBallot(ranking=({"Chicken"}, {"Chocolate"}, {"Burger"}), weight=3),
        ],
        max_ranking_length=3,
    ),
    RankProfile(
        ballots=[
            RankBallot(ranking=({"Orange"}, {"Pear"}), weight=3.5),
            RankBallot(ranking=({"Pear"}, {"Cake"}), weight=8),
            RankBallot(ranking=({"Cake"},), weight=3),
            RankBallot(ranking=({"Cake"}, {"Burger"}), weight=1),
            RankBallot(ranking=({"Burger"}, {"Chicken"}), weight=4),
            RankBallot(ranking=({"Chicken"}, {"Burger"}), weight=3),
        ],
        max_ranking_length=3,
    ),
    RankProfile(
        ballots=[
            RankBallot(ranking=({"Orange"}, {"Pear"}), weight=3.5),
            RankBallot(ranking=({"Pear"}, {"Cake"}), weight=8),
            RankBallot(ranking=({"Cake"},), weight=3),
            RankBallot(ranking=({"Cake"}, {"Burger"}), weight=1),
            RankBallot(ranking=({"Burger"},), weight=7),
        ],
        max_ranking_length=3,
    ),
    RankProfile(
        ballots=[
            RankBallot(ranking=({"Pear"},), weight=3.5),
            RankBallot(ranking=({"Pear"}, {"Cake"}), weight=8),
            RankBallot(ranking=({"Cake"},), weight=3),
            RankBallot(ranking=({"Cake"}, {"Burger"}), weight=1),
            RankBallot(ranking=({"Burger"},), weight=7),
        ],
        max_ranking_length=3,
    ),
    RankProfile(
        ballots=[
            RankBallot(ranking=({"Pear"},), weight=11.5),
            RankBallot(ranking=({"Burger"},), weight=8),
        ],
        max_ranking_length=3,
    ),
    RankProfile(
        ballots=[
            RankBallot(ranking=({"Pear"},), weight=11.5),
        ],
        max_ranking_length=3,
    ),
    RankProfile(max_ranking_length=3),
]

states = [
    ElectionState(
        round_number=0,
        remaining=(
            frozenset({"Pear"}),
            frozenset({"Burger"}),
            frozenset({"Orange", "Cake", "Chicken"}),
            frozenset({"Chocolate"}),
            frozenset({"Strawberry"}),
        ),
        scores={
            "Pear": 8,
            "Burger": 4,
            "Orange": 3,
            "Cake": 3,
            "Chicken": 3,
            "Strawberry": 1 / 2,
            "Chocolate": 1,
        },
    ),
    ElectionState(
        round_number=1,
        remaining=(
            frozenset({"Pear"}),
            frozenset({"Burger"}),
            frozenset({"Orange"}),
            frozenset({"Cake", "Chicken"}),
            frozenset({"Chocolate"}),
        ),
        eliminated=(frozenset({"Strawberry"}),),
        scores={
            "Burger": 4,
            "Orange": 3.5,
            "Cake": 3,
            "Chicken": 3,
            "Chocolate": 1,
            "Pear": 8,
        },
    ),
    ElectionState(
        round_number=2,
        remaining=(
            frozenset({"Pear"}),
            frozenset({"Burger", "Cake"}),
            frozenset({"Orange"}),
            frozenset({"Chicken"}),
        ),
        eliminated=(frozenset({"Chocolate"}),),
        scores={"Burger": 4, "Orange": 3.5, "Cake": 4, "Chicken": 3, "Pear": 8},
    ),
    ElectionState(
        round_number=3,
        remaining=(
            frozenset({"Pear"}),
            frozenset({"Burger"}),
            frozenset({"Cake"}),
            frozenset({"Orange"}),
        ),
        eliminated=(frozenset({"Chicken"}),),
        scores={"Burger": 7, "Orange": 3.5, "Cake": 4, "Pear": 8},
    ),
    ElectionState(
        round_number=4,
        remaining=(
            frozenset({"Pear"}),
            frozenset({"Burger"}),
            frozenset({"Cake"}),
        ),
        eliminated=(frozenset({"Orange"}),),
        scores={"Burger": 7, "Cake": 4, "Pear": 11.5},
    ),
    ElectionState(
        round_number=5,
        remaining=(
            frozenset({"Pear"}),
            frozenset({"Burger"}),
        ),
        eliminated=(frozenset({"Cake"}),),
        scores={"Burger": 8, "Pear": 11.5},
    ),
    ElectionState(
        round_number=6,
        remaining=(frozenset({"Pear"}),),
        eliminated=(frozenset({"Burger"}),),
        scores={"Pear": 11.5},
    ),
    ElectionState(
        round_number=7,
        elected=(frozenset({"Pear"}),),
    ),
]


def test_init():
    e = IRV(test_profile)
    assert e.get_elected() == (frozenset({"Pear"}),)


def test_quotas():
    e = IRV(test_profile, quota="droop")
    assert e.threshold == 12

    e = IRV(test_profile, quota="hare")
    assert e.threshold == 22


def test_profiles():
    e = IRV(test_profile)
    for i in range(len(e.election_states)):
        if e.get_profile(i) != profile_list[i]:
            print(e.get_profile(i).df.to_string())
            print(profile_list[i].df.to_string())
            print()
    assert [e.get_profile(i) for i in range(len(e.election_states))] == profile_list


def test_state_list():
    e = IRV(test_profile)
    assert e.election_states == states


def test_get_profile():
    e = IRV(test_profile)
    assert e.get_profile(0) == test_profile
    assert e.get_profile(-1) == profile_list[-1]


def test_get_step():
    e = IRV(test_profile)
    assert e.get_step(-1) == (profile_list[-1], states[-1])


def test_get_elected():
    e = IRV(test_profile)
    assert e.get_elected(0) == tuple()
    assert e.get_elected(-1) == (frozenset({"Pear"}),)


def test_get_eliminated():
    e = IRV(test_profile)
    assert e.get_eliminated(0) == tuple()
    assert e.get_eliminated(1) == (frozenset({"Strawberry"}),)


def test_get_remaining():
    e = IRV(test_profile)
    assert e.get_remaining(0) == (
        frozenset({"Pear"}),
        frozenset({"Burger"}),
        frozenset({"Orange", "Cake", "Chicken"}),
        frozenset({"Chocolate"}),
        frozenset({"Strawberry"}),
    )
    assert e.get_remaining(-1) == (frozenset(),)


def test_get_ranking():
    e = IRV(test_profile)
    assert e.get_ranking(0) == (
        frozenset({"Pear"}),
        frozenset({"Burger"}),
        frozenset({"Orange", "Cake", "Chicken"}),
        frozenset({"Chocolate"}),
        frozenset({"Strawberry"}),
    )
    assert e.get_ranking(-1) == (
        frozenset({"Pear"}),
        frozenset({"Burger"}),
        frozenset({"Cake"}),
        frozenset({"Orange"}),
        frozenset({"Chicken"}),
        frozenset({"Chocolate"}),
        frozenset({"Strawberry"}),
    )


def test_get_status_df():
    e = IRV(test_profile)
    df_final = pd.DataFrame(
        {
            "Status": [
                "Elected",
                "Eliminated",
                "Eliminated",
                "Eliminated",
                "Eliminated",
                "Eliminated",
                "Eliminated",
            ],
            "Round": [7, 6, 5, 4, 3, 2, 1],
        },
        index=[
            "Pear",
            "Burger",
            "Cake",
            "Orange",
            "Chicken",
            "Chocolate",
            "Strawberry",
        ],
    )

    assert e.get_status_df(-1).equals(df_final)


def test_errors():
    with pytest.raises(ValueError, match="Misspelled or unknown quota type."):
        IRV(RankProfile(ballots=(RankBallot(ranking=({"A"},)),)), quota="Drip")

    with pytest.raises(ProfileError, match="Profile must be of type RankProfile."):
        IRV(ScoreProfile(ballots=(ScoreBallot(scores={"A": 4}),)))
