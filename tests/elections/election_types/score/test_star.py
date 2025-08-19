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
import random
import time

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
        elected=tuple([frozenset({"B"}),]),
        eliminated = tuple([frozenset({'A'}), frozenset({'C'})]),
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
def test_init():
    e = Star(profile=profile_simple, L=5)
    assert [e for e in next(iter(e.get_elected()[0]))] == winner_simple

def test_ties():
    e = Star(profile=profile_tie_tabulation, L=5, tiebreak="most_top_ratings")
    assert len(e.get_elected()) == 1

def test_state_list():
    e = Star(profile=profile_simple, L=5)
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
    assert e.get_status_df(0).sort_index().equals(df0.sort_index())
    assert e.get_status_df(1).sort_index().equals(df1.sort_index())

def test_errors():
    # m must be positive
    with pytest.raises(ValueError, match="L must be positive"):
        Star(profile=profile_simple, L=-1)

    # tiebreak must be none or must be most_top_ratings
    with pytest.raises(ValueError, match="tiebreak must be 'most_top_ratings'"):
        Star(profile=profile_simple, L=5, tiebreak="no_tiebreak")

def test_validate_profile():    
    # ballot cannot violate score limit
    bad1 = PreferenceProfile(
        ballots=[Ballot(scores={"X": 6, "Y": 1})]
    )
    with pytest.raises(TypeError, match="score limit"):
        Star(profile=bad1, L=5)

# Stress test
n_ballots = 100_000
n_candidates = 30

# Build candidates
candidates = [f"C{i}" for i in range(n_candidates)]

# Build ballots with STAR scores (randomized)
ballots = [Ballot(scores={c: random.randint(0, 5) for c in candidates}, weight=1) for _ in range(n_ballots)]
profile_stress = PreferenceProfile(
            ballots=ballots,
            candidates=candidates,
)

def test_stress():
    # Stress test for OpenListPR with large profile
    times = []
    for _ in range(10):

        time0 = time.time()
        election = Star(profile_stress, L=5)
        time1 = time.time()
        times.append(time1 - time0)

    print("Average time:", sum(times)/len(times))
    assert sum(times)/len(times) < 1