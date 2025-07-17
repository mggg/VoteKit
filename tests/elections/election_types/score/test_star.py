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
profile_simple = PreferenceProfile(
    ballots=[
        Ballot(scores={"A": 5, "B": 4, "C": 3}),
        Ballot(scores={"A": 0, "B": 5, "C": 2}),
        Ballot(scores={"A": 3, "B": 3, "C": 5}),
    ]
)
runoff_simple = ["B", "C"]
winner_simple = ["B"]

# Only B and C 
profile_runoff_only = PreferenceProfile(
    ballots=[
        Ballot(scores={"B": 4, "C": 3}),
        Ballot(scores={"B": 5, "C": 2}),
        Ballot(scores={"B": 3, "C": 5}),
    ]
)
winner_runoff_only = ["B"]

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
runoff_large_election = ["E", "A"]
winner_large_election = ["E"]
# ------- states (no tie) -----

states_simple = [
    ElectionState(
        remaining=(frozenset({"B"}), frozenset({"C"}), frozenset({"A"})),
        scores={"A": 8.0, "B": 12.0, "C": 10.0},
    ),
    ElectionState(
        round_number=1,
        remaining=tuple(),
        elected=[frozenset({"B"}),],
        eliminated = [frozenset({'A'}), frozenset({'C'})],
        scores={"A": 8.0, "B": 12.0, "C": 10.0},
    )
]

profile_simple_round_1 = PreferenceProfile(
    ballots=[
        Ballot(scores={"A": 5, "C": 3}),
        Ballot(scores={"A": 0, "C": 2}),
        Ballot(scores={"A": 3, "C": 5}),
    ]
)

# ---------- tests ----------

def test_basic_result():
    """Winner B, finalists (B,C), correct score totals & runoff counts."""
    election = Star(profile_simple, L=5)
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
    election = Star(profile_simple, L=5)
    assert len(election.election_states) == 2
    assert election.election_states[0].scores == states_simple[0].scores
    assert election.election_states[1].elected == [frozenset({candidate}) for candidate in next(iter(states_simple[1].elected))]

    assert election.get_profile(0) == profile_simple
    _, state1 = election.get_step(1)
    assert state1 == election.election_states[1] 

def test_remaining_ranking():
    election = Star(profile_simple, L=5)
    assert election.get_remaining(0) == (frozenset({"B"}), frozenset({"C"}), frozenset({"A"}))
    assert election.get_remaining(1) == tuple()
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

## ------- standard tests ---------- ##
def test_init():
    e = Star(profile=profile_simple, L=5)
    print(e.get_elected())
    print(winner_simple)
    assert [e for e in next(iter(e.get_elected()[0]))] == winner_simple

def test_ties():
    e = Star(profile=profile_tie_tabulation, L=5, tiebreak="most_top_ratings")
    assert len(e.get_elected()) == 0

def test_state_list():
    e = Star(profile=profile_simple, L=5)
    print(e.election_states)
    print(" ")
    print(states_simple)
    assert e.election_states == states_simple

def test_get_profile():
    e = Star(profile=profile_simple, L=5)
    assert e.get_profile(0) == profile_simple
    assert e.get_profile(1) == profile_simple_round_1

def test_get_step():
    e = Star(profile=profile_simple, L=5)    
    assert e.get_step(1) == (profile_simple_round_1, states_simple[1])

def test_get_elected():
    e = Star(profile=profile_simple, L=5)
    assert e.get_elected(0) == tuple()
    assert e.get_elected(1) == (frozenset({"B"}),)

def test_get_eliminated():
    e = Star(profile=profile_simple, L=5)
    assert e.get_eliminated(0) == tuple()
    assert e.get_eliminated(1) == (frozenset({"C"}),frozenset({"A"}))

def test_get_remaining():
    e = Star(profile=profile_simple, L=5)
    assert e.get_remaining(0) == (frozenset({"B"}), frozenset({"C"}), frozenset({"A"}))
    assert e.get_remaining(1) == tuple()

def test_get_status_df():
    e = Star(profile=profile_simple, L=5)
    df0 = pd.DataFrame(
        {"Status": ["Remaining"] * 3, "Round": [0] * 3},
        index=["A", "B", "C"],
    )
    df1 = pd.DataFrame(
        {"Status": ["Elected", "Eliminated", "Eliminated"], "Round": [1] * 3},
        index=["B", "A", "C"],
    )
    print(df0)
    print(e.get_status_df(0))
    print(df1)
    print(e.get_status_df(1))
    assert e.get_status_df(0).sort_index().equals(df0.sort_index())
    assert e.get_status_df(1).sort_index().equals(df1.sort_index())

def test_errors():
    # m must be positive
    with pytest.raises(ValueError, match="L must be positive"):
        Star(profile=profile_simple, L=-1)

    # tiebreak must be none or must be most_top_ratings
    with pytest.raises(ValueError, match="tiebreak must be None or 'most_top_ratings'"):
        Star(profile=profile_simple, L=5, tiebreak="random")

def test_validate_profile():
    # there must be at least two candidates
    bad0 = PreferenceProfile(
        ballots=[Ballot(scores={"A": 5}, weight=10),]
    )
    with pytest.raises(ValueError, match="at least two"):
        Star(profile=bad0, L=5)

    # ballot must have score dictionary
    bad1 = PreferenceProfile(
        ballots=[Ballot(ranking=(frozenset({"X"}), frozenset({"Y"})))]
    )
    with pytest.raises(TypeError, match="score dictionary"):
        Star(profile=bad1, L=5)
    
    # ballot cannot violate score limit
    bad2 = PreferenceProfile(
        ballots=[Ballot(scores={"X": 6, "Y": 1})]
    )
    with pytest.raises(TypeError, match="score limit"):
        Star(profile=bad2, L=5)

    