import pandas as pd
import pytest

from votekit.ballot import RankBallot, ScoreBallot
from votekit.elections import BlockPlurality, ElectionState
from votekit.pref_profile import RankProfile, ScoreProfile

profile_no_tied_bloc_plurality = ScoreProfile(
    ballots=[
        ScoreBallot(scores={"A": 1, "B": 1}, weight=3),
        ScoreBallot(scores={"A": 1, "C": 1}, weight=2),
    ],
    candidates=("A", "B", "C", "D"),
)


profile_no_tied_bloc_plurality_round_1 = ScoreProfile(
    ballots=[
        ScoreBallot(scores={"C": 1}, weight=2),
    ],
    candidates=("C", "D"),
)

profile_tied_bloc_plurality = ScoreProfile(
    ballots=[
        ScoreBallot(scores={"A": 1}),
        ScoreBallot(scores={"B": 1}),
    ],
    candidates=("A", "B", "C"),
)


states = [
    ElectionState(
        remaining=(
            frozenset({"A"}),
            frozenset({"B"}),
            frozenset({"C"}),
            frozenset({"D"}),
        ),
        scores={"A": 5, "B": 3, "C": 2, "D": 0},
    ),
    ElectionState(
        round_number=1,
        remaining=(frozenset({"C"}), frozenset({"D"})),
        elected=(frozenset({"A"}), frozenset({"B"})),
        scores={"C": 2, "D": 0},
    ),
]


def test_init():
    e = BlockPlurality(profile_no_tied_bloc_plurality, n_seats=2)
    assert e.get_elected() == (frozenset({"A"}), frozenset({"B"}))


def test_ties():
    e_random = BlockPlurality(profile_tied_bloc_plurality, n_seats=1, tiebreak="random")
    assert len([c for s in e_random.get_elected() for c in s]) == 1


def test_state_list():
    e = BlockPlurality(profile_no_tied_bloc_plurality, n_seats=2)
    assert e.election_states == states


def test_get_profile():
    e = BlockPlurality(profile_no_tied_bloc_plurality, n_seats=2)
    assert e.get_profile(0) == profile_no_tied_bloc_plurality
    assert e.get_profile(1) == profile_no_tied_bloc_plurality_round_1


def test_get_step():
    e = BlockPlurality(profile_no_tied_bloc_plurality, n_seats=2)
    assert e.get_step(1) == (profile_no_tied_bloc_plurality_round_1, states[1])


def test_get_elected():
    e = BlockPlurality(profile_no_tied_bloc_plurality, n_seats=2)
    assert e.get_elected(0) == tuple()
    assert e.get_elected(1) == (frozenset({"A"}), frozenset({"B"}))


def test_get_eliminated():
    e = BlockPlurality(profile_no_tied_bloc_plurality, n_seats=2)
    assert e.get_eliminated(0) == tuple()
    assert e.get_eliminated(1) == tuple()


def test_get_remaining():
    e = BlockPlurality(profile_no_tied_bloc_plurality, n_seats=2)
    assert e.get_remaining(0) == (
        frozenset({"A"}),
        frozenset({"B"}),
        frozenset({"C"}),
        frozenset({"D"}),
    )
    assert e.get_remaining(1) == (frozenset({"C"}), frozenset({"D"}))


def test_get_ranking():
    e = BlockPlurality(profile_no_tied_bloc_plurality, n_seats=2)
    assert e.get_ranking(0) == (
        frozenset({"A"}),
        frozenset({"B"}),
        frozenset({"C"}),
        frozenset({"D"}),
    )
    assert e.get_ranking(1) == (
        frozenset({"A"}),
        frozenset({"B"}),
        frozenset({"C"}),
        frozenset({"D"}),
    )


def test_get_status_df():
    e = BlockPlurality(profile_no_tied_bloc_plurality, n_seats=2)

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
    with pytest.raises(ValueError, match="n_seats must be positive."):
        BlockPlurality(profile_no_tied_bloc_plurality, n_seats=0)

    with pytest.raises(ValueError, match="Not enough candidates received votes to be elected."):
        BlockPlurality(profile_no_tied_bloc_plurality, n_seats=5)

    with pytest.raises(
        ValueError,
        match="Cannot elect correct number of candidates without breaking ties.",
    ):
        BlockPlurality(profile_tied_bloc_plurality, n_seats=1)


def test_validate_profile():
    with pytest.raises(TypeError, match="violates score limit"):
        profile = ScoreProfile(ballots=[ScoreBallot(scores={"A": 3})])
        BlockPlurality(profile, n_seats=1)

    with pytest.raises(TypeError, match="must have non-negative scores."):
        profile = ScoreProfile(ballots=[ScoreBallot(scores={"A": -3})])
        BlockPlurality(profile, n_seats=1)

    with pytest.raises(TypeError, match="All ballots must have score dictionary."):
        profile = ScoreProfile(ballots=[ScoreBallot(), ScoreBallot(scores={"A": 1})])
        BlockPlurality(profile, n_seats=1)


# ============================================================
# Ranking-based BlockPlurality tests
# ============================================================

ranked_profile_no_tied = RankProfile(
    ballots=[
        RankBallot(ranking=({"A"}, {"B"}, {"C"}), weight=3),
        RankBallot(ranking=({"A"}, {"C"}, {"B"}), weight=2),
    ],
    max_ranking_length=3,
)

ranked_profile_tied = RankProfile(
    ballots=[
        RankBallot(ranking=({"A"}, {"B"})),
        RankBallot(ranking=({"B"}, {"A"})),
    ],
    max_ranking_length=2,
)

# With n_seats=2, budget=2 (default), score_vector=[1,1,0]:
#   Ballot 1 (w=3): A=1, B=1, C=0
#   Ballot 2 (w=2): A=1, C=1, B=0
#   Totals: A=5, B=3, C=2
ranked_states = [
    ElectionState(
        remaining=(frozenset({"A"}), frozenset({"B"}), frozenset({"C"})),
        scores={"A": 5, "B": 3, "C": 2},
    ),
    ElectionState(
        round_number=1,
        remaining=(frozenset({"C"}),),
        elected=(frozenset({"A"}), frozenset({"B"})),
        scores={"C": 5},
    ),
]


def test_ranked_init():
    e = BlockPlurality(ranked_profile_no_tied, n_seats=2)
    assert e.get_elected() == (frozenset({"A"}), frozenset({"B"}))


def test_ranked_ties():
    e_random = BlockPlurality(ranked_profile_tied, n_seats=1, tiebreak="random")
    assert len([c for s in e_random.get_elected() for c in s]) == 1


def test_ranked_state_list():
    e = BlockPlurality(ranked_profile_no_tied, n_seats=2)
    assert e.election_states == ranked_states


def test_ranked_get_elected():
    e = BlockPlurality(ranked_profile_no_tied, n_seats=2)
    assert e.get_elected(0) == tuple()
    assert e.get_elected(1) == (frozenset({"A"}), frozenset({"B"}))


def test_ranked_get_eliminated():
    e = BlockPlurality(ranked_profile_no_tied, n_seats=2)
    assert e.get_eliminated(0) == tuple()
    assert e.get_eliminated(1) == tuple()


def test_ranked_get_remaining():
    e = BlockPlurality(ranked_profile_no_tied, n_seats=2)
    assert e.get_remaining(0) == (
        frozenset({"A"}),
        frozenset({"B"}),
        frozenset({"C"}),
    )
    assert e.get_remaining(1) == (frozenset({"C"}),)


def test_ranked_get_ranking():
    e = BlockPlurality(ranked_profile_no_tied, n_seats=2)
    assert e.get_ranking(0) == (
        frozenset({"A"}),
        frozenset({"B"}),
        frozenset({"C"}),
    )
    assert e.get_ranking(1) == (
        frozenset({"A"}),
        frozenset({"B"}),
        frozenset({"C"}),
    )


def test_ranked_get_status_df():
    e = BlockPlurality(ranked_profile_no_tied, n_seats=2)

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


def test_ranked_budget():
    profile = RankProfile(
        ballots=[
            RankBallot(ranking=({"A"}, {"B"}, {"C"}), weight=4),
            RankBallot(ranking=({"B"}, {"C"}, {"A"}), weight=3),
            RankBallot(ranking=({"C"}, {"A"}, {"B"}), weight=2),
        ],
        max_ranking_length=3,
    )
    # budget=1: score_vector=[1,0,0], only 1st place counts
    #   A=4, B=3, C=2 → A elected
    e1 = BlockPlurality(profile, n_seats=1, budget=1)
    assert e1.get_elected() == (frozenset({"A"}),)

    # budget=2: score_vector=[1,1,0], top 2 positions count
    #   A=4+2=6, B=4+3=7, C=3+2=5 → B elected
    e2 = BlockPlurality(profile, n_seats=1, budget=2)
    assert e2.get_elected() == (frozenset({"B"}),)


def test_ranked_scoring_tie_convention():
    profile = RankProfile(
        ballots=[
            RankBallot(ranking=({"A", "B"}, {"C"})),
            RankBallot(ranking=({"C"}, {"A"}, {"B"})),
        ],
        max_ranking_length=3,
    )

    # budget=1, score_vector=[1, 0, 0]
    # Ballot 1: A,B tied for positions 1-2, C at position 3
    # Ballot 2: C at pos 1, A at pos 2, B at pos 3

    # "low": Ballot 1: A=min(1,0)=0, B=0. Ballot 2: C=1, A=0, B=0.
    #   Totals: A=0, B=0, C=1 → C elected
    e_low = BlockPlurality(profile, n_seats=1, budget=1, scoring_tie_convention="low")
    assert e_low.get_elected() == (frozenset({"C"}),)

    # "high": Ballot 1: A=max(1,0)=1, B=1. Ballot 2: C=1, A=0, B=0.
    #   Totals: A=1, B=1, C=1 → three-way tie, need tiebreak
    e_high = BlockPlurality(
        profile, n_seats=1, budget=1, scoring_tie_convention="high", tiebreak="random"
    )
    assert len([c for s in e_high.get_elected() for c in s]) == 1


def test_ranked_errors():
    with pytest.raises(ValueError, match="n_seats must be positive."):
        BlockPlurality(ranked_profile_no_tied, n_seats=0)

    with pytest.raises(ValueError, match="Not enough candidates received votes to be elected."):
        BlockPlurality(ranked_profile_no_tied, n_seats=4, budget=3)

    with pytest.raises(
        ValueError,
        match="Cannot elect correct number of candidates without breaking ties.",
    ):
        BlockPlurality(ranked_profile_tied, n_seats=1)

    with pytest.raises(ValueError, match="budget.*cannot exceed max_ranking_length"):
        BlockPlurality(ranked_profile_no_tied, n_seats=1, budget=5)

    with pytest.raises(TypeError, match="profile must be a RankProfile or ScoreProfile"):
        BlockPlurality("not a profile")  # type: ignore[invalid-argument-type]


def test_ranked_validate_profile():
    with pytest.raises(TypeError, match="has no ranking"):
        profile = RankProfile(ballots=[RankBallot(ranking=({"A"},)), RankBallot()])
        BlockPlurality(profile, n_seats=1)
