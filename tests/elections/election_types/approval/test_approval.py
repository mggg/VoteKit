from votekit.elections import Approval, ElectionState
from votekit.pref_profile import ScoreProfile, RankProfile
from votekit.ballot import ScoreBallot, RankBallot
import pytest
import pandas as pd

profile_no_tied_approval = ScoreProfile(
    ballots=[
        ScoreBallot(scores={"A": 1, "B": 1, "C": 1, "D": 1}),
        ScoreBallot(
            scores={
                "A": 1,
                "B": 1,
            },
            weight=2,
        ),
    ],
    candidates=("A", "B", "C", "D"),
)


profile_no_tied_approval_round_1 = ScoreProfile(
    ballots=[
        ScoreBallot(scores={"C": 1, "D": 1}),
    ],
    candidates=("C", "D"),
)

profile_tied_approval = ScoreProfile(
    ballots=[
        ScoreBallot(scores={"A": 1, "B": 1, "C": 1}),
        ScoreBallot(scores={"A": 1, "B": 1, "C": 0}),
    ],
    candidates=("A", "B", "C", "D"),
)


states = [
    ElectionState(
        remaining=(frozenset({"A", "B"}), frozenset({"C", "D"})),
        scores={"A": 3, "B": 3, "C": 1, "D": 1},
    ),
    ElectionState(
        round_number=1,
        remaining=(frozenset({"C", "D"}),),
        elected=(frozenset({"A", "B"}),),
        scores={"C": 1, "D": 1},
    ),
]


def test_init():
    e = Approval(profile_no_tied_approval, m=2)
    assert e.get_elected() == (frozenset({"A", "B"}),)


def test_ties():
    e_random = Approval(profile_tied_approval, m=1, tiebreak="random")
    assert len([c for s in e_random.get_elected() for c in s]) == 1


def test_state_list():
    e = Approval(profile_no_tied_approval, m=2)
    assert e.election_states == states


def test_get_profile():
    e = Approval(profile_no_tied_approval, m=2)
    assert e.get_profile(0) == profile_no_tied_approval
    assert e.get_profile(1) == profile_no_tied_approval_round_1


def test_get_step():
    e = Approval(profile_no_tied_approval, m=2)
    assert e.get_step(1) == (profile_no_tied_approval_round_1, states[1])


def test_get_elected():
    e = Approval(profile_no_tied_approval, m=2)
    assert e.get_elected(0) == tuple()
    assert e.get_elected(1) == (frozenset({"A", "B"}),)


def test_get_eliminated():
    e = Approval(profile_no_tied_approval, m=2)
    assert e.get_eliminated(0) == tuple()
    assert e.get_eliminated(1) == tuple()


def test_get_remaining():
    e = Approval(profile_no_tied_approval, m=2)
    assert e.get_remaining(0) == (frozenset({"A", "B"}), frozenset({"C", "D"}))
    assert e.get_remaining(1) == (frozenset({"C", "D"}),)


def test_get_ranking():
    e = Approval(profile_no_tied_approval, m=2)
    assert e.get_ranking(0) == (frozenset({"A", "B"}), frozenset({"C", "D"}))
    assert e.get_ranking(1) == (frozenset({"A", "B"}), frozenset({"C", "D"}))


def test_get_status_df():
    profile_no_ties = ScoreProfile(
        ballots=[
            ScoreBallot(scores={"A": 1}, weight=3),
            ScoreBallot(scores={"B": 1}, weight=2),
            ScoreBallot(scores={"C": 1}, weight=1),
        ],
        candidates=("A", "B", "C", "D"),
    )
    e = Approval(profile_no_ties, m=2)

    df_0 = pd.DataFrame(
        {"Status": ["Remaining"] * 4, "Round": [0] * 4},
        index=["A", "B", "C", "D"],
    )
    df_1 = pd.DataFrame(
        {"Status": ["Elected", "Elected", "Remaining", "Remaining"], "Round": [1] * 4},
        index=["A", "B", "C", "D"],
    )

    assert e.get_status_df(0).equals(df_0)
    assert e.get_status_df(1).equals(df_1)


def test_errors():
    with pytest.raises(ValueError, match="m must be positive."):
        Approval(profile_no_tied_approval, m=0)

    with pytest.raises(
        ValueError, match="Not enough candidates received votes to be elected."
    ):
        Approval(profile_no_tied_approval, m=5)

    with pytest.raises(
        ValueError,
        match="Cannot elect correct number of candidates without breaking ties.",
    ):
        Approval(profile_tied_approval, m=1)


def test_validate_profile():
    with pytest.raises(TypeError, match="violates score limit"):
        profile = ScoreProfile(ballots=[ScoreBallot(scores={"A": 3})])
        Approval(profile, m=1)

    with pytest.raises(TypeError, match="must have non-negative scores."):
        profile = ScoreProfile(ballots=[ScoreBallot(scores={"A": -3})])
        Approval(profile, m=1)

    with pytest.raises(TypeError, match="must be of type ScoreBallot"):
        profile = RankProfile(ballots=[RankBallot(ranking=({"A"},))])
        Approval(profile, m=1)

    with pytest.raises(TypeError, match="All ballots must have score dictionary."):
        profile = ScoreProfile(ballots=[ScoreBallot(), ScoreBallot(scores={"A": 1})])
        Approval(profile, m=1)
