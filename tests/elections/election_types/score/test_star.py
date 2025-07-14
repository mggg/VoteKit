"""
pytest test-suite for the new STAR (Score-Then-Automatic-Runoff) election class.
Drop this file in  tests/elections/election_types/scores/test_star.py
and run:  pytest tests/elections/election_types/scores/test_star.py
"""
from votekit.elections.election_types.scores.star import Star
from votekit.elections import ElectionState
from votekit import PreferenceProfile, Ballot
import pandas as pd
import pytest


# ---------- sample profiles -------------

# 3 ballots, clear finalist pair (B,C) – B wins the run-off 2 : 1
profile_no_tie = PreferenceProfile(
    ballots=[
        Ballot(scores={"A": 5, "B": 4, "C": 3}),
        Ballot(scores={"A": 0, "B": 5, "C": 2}),
        Ballot(scores={"A": 3, "B": 3, "C": 5}),
    ]
)

# Only B and C 
profile_runoff_only = PreferenceProfile(
    ballots=[
        Ballot(scores={"B": 4, "C": 3}),
        Ballot(scores={"B": 5, "C": 2}),
        Ballot(scores={"B": 3, "C": 5}),
    ]
)

# Tabulation round three-way tie
profile_tie_tabulation = PreferenceProfile(
    ballots=[
        Ballot(scores={"A": 5, "B": 5, "C": 0}),
        Ballot(scores={"A": 0, "B": 0, "C": 5}),
    ]
)

# Large election
profile_large_election = PreferenceProfile(
    ballots=[
        Ballot(scores={"A": 5, "B": 1, "C": 0, "D": 1, "E": 2}),
        Ballot(scores={"A": 2, "B": 3, "C": 1, "D": 0, "E": 5}),
        Ballot(scores={"A": 1, "B": 3, "C": 0, "D": 1, "E": 5}),
        Ballot(scores={"A": 4, "B": 3, "C": 2, "D": 0, "E": 5}),
        Ballot(scores={"A": 5, "B": 1, "C": 0, "D": 3, "E": 4}),
        Ballot(scores={"A": 5, "B": 2, "C": 0, "D": 1, "E": 4}),
        Ballot(scores={"A": 4, "B": 3, "C": 0, "D": 3, "E": 5}),
        Ballot(scores={"A": 2, "B": 1, "C": 5, "D": 3, "E": 4}),
        Ballot(scores={"A": 5, "B": 2, "C": 0, "D": 1, "E": 3}),
    ]
)
# ------- states (no tie) -----

states_no_tie = [
    ElectionState(
        remaining=(frozenset({"B"}), frozenset({"C"}), frozenset({"A"})),
        scores={"A": 8, "B": 12, "C": 10},
    ),
    ElectionState(
        round_number=1,
        remaining=tuple(),
        elected=(frozenset({"B"}),),
        scores={"A": 8, "B": 12, "C": 10},
    )
]

# ---------- tests ----------

def test_basic_result():
    """Winner B, finalists (B,C), correct score totals & runoff counts."""
    election = Star(profile_no_tie, L=5)
    result = election.run_election()

    assert result["winner"] == "B"
    assert set(result["finalists"]) == {"B", "C"}
    assert result["scores"] == {"A": 8, "B": 12, "C": 10}


def test_runoff_only_profile():
    """If only finalists appear on ballots B must still win."""
    election = Star(profile_runoff_only, L=5)
    result = election.run_election()
    assert result["winner"] == "B"
    assert set(result["finalists"]) == {"B", "C"}


def test_tabulation_tie_without_tiebreak():
    """All three candidates tied on score without tiebreak winner is None."""
    res = Star(profile_tie_tabulation, L=5).run_election()
    assert res["winner"] is None


def test_tabulation_tie_with_most_top_ratings():
    """Same tie but with 'most_top_ratings' still ends in None (1=1)."""
    res = Star(profile_tie_tabulation, L=5, tiebreak="most_top_ratings").run_election()
    assert res["winner"] is None
    assert set(res["finalists"]) == {"A", "B"}

def test_state_list_and_helpers():
    election = Star(profile_no_tie, L=5)
    assert len(election.election_states) == 2
    assert election.election_states[0].scores == states_no_tie[0].scores
    assert election.election_states[1].elected == [frozenset({candidate}) for candidate in next(iter(states_no_tie[1].elected))]

    assert election.get_profile(0) == profile_no_tie
    _, state1 = election.get_step(1)
    assert state1 == election.election_states[1] 

def test_remaining_ranking():
    election = Star(profile_no_tie, L=5)
    assert election.get_remaining(0) == (frozenset({"B"}), frozenset({"C"}), frozenset({"A"}))
    assert election.get_remaining(1) == (frozenset({"A"}), frozenset({"C"}))
    assert election.get_ranking(0) == (frozenset({"B"}), frozenset({"C"}), frozenset({"A"}))

def test_error_conditions():
    with pytest.raises(ValueError, match="STAR requires at least two candidates."):
        Star(PreferenceProfile(ballots=[Ballot(scores={"Only": 5})]), L=5)

    with pytest.raises(TypeError, match="All ballots must have score dictionary."):
        Star(PreferenceProfile(ballots=[Ballot(ranking=("A","B"))]), L=5)

    with pytest.raises(TypeError, match="score limit"):
        Star(PreferenceProfile(ballots=[Ballot(scores={"A": 6, "B": 1})]), L=5)

    ## TODO: It seems non-positve weight error caught by Ballot

    # with pytest.raises(TypeError, match="positive weight"):
    #     Star(
    #         PreferenceProfile(
    #             ballots=[Ballot(scores={"A": 1, "B": 2}, weight=0)]
    #         ),
    #         L=5,
    #     )


def test_large_election_result_and_status_df():
    """Comprehensive check on the 9-ballot, 5-candidate example."""
    election = Star(profile_large_election, L=5)
    res = election.run_election()

    assert res["winner"] == "E"
    assert set(res["finalists"]) == {"E", "A"}

    # round-0 candidate ordering by total score
    expected_ranking0 = (
        frozenset({"E"}),
        frozenset({"A"}),
        frozenset({"B"}),
        frozenset({"D"}),
        frozenset({"C"}),
    )
    state0 = election.get_step(0)[-1]
    assert tuple(state0.remaining) == expected_ranking0

    # status-df helper
    df0_expected = pd.DataFrame(
        {"Status": ["Remaining"] * 5, "Round": [0] * 5},
        index=["E", "A", "B", "D", "C"],
    )
    assert election.get_status_df(0).equals(df0_expected)

    # after runoff (state-1) winner E elected
    state1 = election.get_step(1)[-1]
    print(state1.elected)
    assert state1.elected == [frozenset({"E"})]