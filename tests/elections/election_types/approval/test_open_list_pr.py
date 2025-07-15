"""
pytest test-suite for the Open-list Party-list PR election class.

To run:
    pytest tests/test_openlist_pr.py
"""
import pytest
import pandas as pd
from votekit import PreferenceProfile, Ballot
from votekit.elections import ElectionState
from votekit.elections.election_types.approval.open_list_pr import OpenListPR
        
        # ---------- samples ----------

# 3 seat, two parties
profile_simple = PreferenceProfile(
    ballots=[
        *[Ballot(ranking=(frozenset({"G1"}),)) for _ in range(40)],
        *[Ballot(ranking=(frozenset({"G2"}),)) for _ in range(25)],
        *[Ballot(ranking=(frozenset({"R1"}),)) for _ in range(30)],
        *[Ballot(ranking=(frozenset({"R2"}),)) for _ in range(20)],
    ]
)
party_map_simple = {"G1": "Green", "G2": "Green", "R1": "Red", "R2": "Red"}
winners_simple = [frozenset({"G1"}), frozenset({"R1"}), frozenset({"G2"})]  #["G1", "R1", "G2"]

# tie
profile_tie = PreferenceProfile(
    ballots=[Ballot(ranking=(frozenset({"A1"}),)) for _ in range(10)] + 
    [Ballot(ranking=(frozenset({"B1"}),)) for _ in range(10)]
)
party_map_tie = {"A1": "Alpha", "B1": "Beta"}


# Party with 1 nominee but wins a seat amount > nominees
profile_exhaustion = PreferenceProfile(
    ballots=[Ballot(ranking=(frozenset({"S1"}),)) for _ in range(12)] + 
    [Ballot(ranking=(frozenset({"D1"}),)) for _ in range(6)] + 
    [Ballot(ranking=(frozenset({"D2"}),)) for _ in range(3)])
party_map_exhaustion = {"S1": "Solo", "D1": "Duo", "D2": "Duo"}

# 3 seats 4 parties
profile_large = PreferenceProfile(
    ballots=
    [Ballot(ranking=(frozenset({"A"}),)) for _ in range(10)] +
    [Ballot(ranking=(frozenset({"B"}),)) for _ in range(7)] +
    [Ballot(ranking=(frozenset({"C"}),)) for _ in range(5)] +
    [Ballot(ranking=(frozenset({"D"}),)) for _ in range(12)] +
    [Ballot(ranking=(frozenset({"E"}),)) for _ in range(8)] +
    [Ballot(ranking=(frozenset({"F"}),)) for _ in range(15)] +
    [Ballot(ranking=(frozenset({"G"}),)) for _ in range(10)] +
    [Ballot(ranking=(frozenset({"H"}),)) for _ in range(5)]
)
party_map_large = {"A": "Party1", "B": "Party1", "C": "Party1", 
                   "D": "Party2", "E": "Party2", 
                   "F": "Party3", "G": "Party3", 
                   "H": "Party4"}

winners_large = ("F", "A", "D")

        # -------- states for testing -------------------

states_simple = [
    ElectionState(
        remaining=(frozenset({"G1", "R1", "G2", "R2"}),),
        scores={"Green": 65, "Red": 50},
    ),
    ElectionState(
        round_number=1,
        remaining=(frozenset({"R1"}), frozenset({"G2"}), frozenset({"R2"})),
        elected=(frozenset({"G1"})),
        scores={"Green": 32.5, "Red": 50},
    ),
    ElectionState(
        round_number=2,
        remaining=(frozenset({"G2"}),),
        elected=(frozenset({"G1"}), frozenset({"R1"})),
        scores={"Green": 32.5, "Red": 25},
    ),
]

        # ------------------ tests ------------------
def test_basic_allocation():
    """Correct winners & deterministic seat order for a simple election."""
    election = OpenListPR(profile_simple, m=3, party_map=party_map_simple, tiebreak="random")
    elected_list = [candidate for candidate in election.get_elected()]
    assert elected_list == winners_simple

def test_ballot_validation():
    """Ballots must contain exactly one ranked candidate."""
    bad_profile = PreferenceProfile(ballots=[Ballot(ranking=(frozenset({"X"}), frozenset({"Y"})))])
    with pytest.raises(ValueError, match="exactly one"):
        OpenListPR(bad_profile, m=1, tiebreak="random")

def test_tie_needs_tiebreak():
    """A tie in party scores must raise unless tiebreak strategy supplied."""
    # Without tiebreak
    with pytest.raises(ValueError, match="Tie between parties"):
        OpenListPR(profile_tie, m=1, party_map=party_map_tie)

    # With random tiebreak should succeed
    election = OpenListPR(profile_tie, m=1, party_map=party_map_tie, tiebreak="random")
    assert len(election.get_elected()) == 1
    assert election.get_elected()[0] in {frozenset({"A1"}), frozenset({"B1"})}

def test_party_exhaustion():
    """Solo party seats its only nominee, then drops out."""
    election = OpenListPR(profile_exhaustion, m=2, party_map=party_map_exhaustion, tiebreak="random")
    assert election.run_election()["winners"] == ("S1", "D1")

    assert election.election_states[1].scores["Solo"] == 12
    assert election.election_states[2].scores["Solo"] == 0

# ---------- states ---------- #

def test_state_list_and_helpers():
    election = OpenListPR(profile_simple, m=3, party_map=party_map_simple, tiebreak="random")
    
    # three rounds = three stored states (+ initial state)
    assert len(election.election_states) == 4
    assert election.election_states[1].scores == states_simple[0].scores
    assert election.election_states[2].scores == states_simple[1].scores
    assert election.election_states[3].scores == states_simple[2].scores

    assert election.get_profile(0) == profile_simple
    prof1, state1 = election.get_step(1)
    assert state1 == election.election_states[1]
    assert next(iter(winners_simple[1])) in prof1.candidates 

def test_large_election():
    election = OpenListPR(profile_large, m=3, party_map=party_map_large, tiebreak="random")
    res = election.run_election()

    assert res["winners"] == winners_large
    assert res["party_seats"] == {"Party1": 1, "Party2": 1, "Party3": 1, "Party4": 0}


def test_error_conditions():
    with pytest.raises(ValueError, match="must be positive"):
        OpenListPR(profile_simple, m=0, party_map=party_map_simple)
