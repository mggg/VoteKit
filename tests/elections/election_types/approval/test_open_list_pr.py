import pytest
import random
import pandas as pd
from votekit import PreferenceProfile, Ballot
from votekit.elections import ElectionState
from votekit.elections.election_types.approval.open_list_pr import OpenListPR
import time
        
        # ---------- samples ----------

# 3 seat, two parties
profile_simple = PreferenceProfile(
    ballots=[
        Ballot(scores={"G1": 1}, weight = 40),
        Ballot(scores={"G2": 1}, weight = 25),
        Ballot(scores={"R1": 1}, weight = 30),
        Ballot(scores={"R2": 1}, weight = 20),
    ]
)
party_map_simple = {"G1": "Green", "G2": "Green", "R1": "Red", "R2": "Red"}
winners_simple = [frozenset({"G1"}), frozenset({"R1"}), frozenset({"G2"})]  #["G1", "R1", "G2"]

# tie
profile_tie = PreferenceProfile(
    ballots=[
        Ballot(scores={"A1": 1}, weight=10),
        Ballot(scores={"B1": 1}, weight=10)
    ] 
)
party_map_tie = {"A1": "Alpha", "B1": "Beta"}


# Party with 1 nominee but wins a seat amount > nominees
profile_exhaustion = PreferenceProfile(
    ballots=[Ballot(scores={"S1":1}, weight=12),
    Ballot(scores={"D1":1}, weight=3), 
    Ballot(scores={"D2":1}, weight=2)]
    )
party_map_exhaustion = {"S1": "Solo", "D1": "Duo", "D2": "Duo"}

# 3 seats 4 parties
profile_large = PreferenceProfile(
    ballots=
    [Ballot(scores={"A": 1}, weight=7),
    Ballot(scores={"B": 1}, weight=5),
    Ballot(scores={"C": 1}, weight=10),
    Ballot(scores={"D": 1}, weight=3),
    Ballot(scores={"E": 1}, weight=8),
    Ballot(scores={"F": 1}, weight=10),
    Ballot(scores={"G": 1}, weight=15),
    Ballot(scores={"H": 1}, weight=5),]
)
party_map_large = {"A": "Party1", "B": "Party1", "C": "Party1", 
                   "D": "Party2", "E": "Party2", 
                   "F": "Party3", "G": "Party3", 
                   "H": "Party4"}

winners_large = ("G", "C", "F")

        # -------- states for testing -------------------

states_simple = [
    ElectionState(
        remaining=(frozenset({"G1"}), frozenset({"G2"}), frozenset({"R1"}), frozenset({"R2"})),
        scores={},
        eliminated=(frozenset({}),),
    ),
    ElectionState(
        round_number=1,
        remaining=[frozenset({"G2"}), frozenset({"R1"}), frozenset({"R2"})],
        elected=[frozenset({"G1"})],
        scores={"Green": 65.0, "Red": 50.0},
        eliminated=tuple()
    ),
    ElectionState(
        round_number=2,
        remaining=[frozenset({"G2"}), frozenset({"R2"})],
        elected=[frozenset({"R1"})],
        scores={"Green": 32.5, "Red": 50.0},
        eliminated=tuple(),
    ),
]

profile_simple_round_1 = PreferenceProfile(
    ballots=[
        Ballot(scores={"G1": 1}, weight = 40),
        Ballot(scores={"G2": 1}, weight = 25),
        Ballot(scores={"R1": 1}, weight = 30),
        Ballot(scores={"R2": 1}, weight = 20),
    ]
)

        # ------------------ tests ------------------
def test_basic_allocation():
    """Correct winners & deterministic seat order for a simple election."""
    election = OpenListPR(profile_simple, party_map=party_map_simple, m=3, tiebreak="random")
    elected_list = [candidate for candidate in election.get_elected()]
    assert elected_list == winners_simple

def test_ballot_validation():
    """Ballots accept candidates with a score of 1."""
    bad_profile1 = PreferenceProfile(ballots=[Ballot(scores={"X": 2, "Y": 1}, weight=10)])
    with pytest.raises(TypeError, match="violates score limit"):
        OpenListPR(bad_profile1, party_map={"X":"XParty", "Y":"YParty"}, m=1, tiebreak="random")

def test_tie_needs_tiebreak():
    """A tie in party scoress must raise unless tiebreak strategy supplied."""
    # Without tiebreak
    with pytest.raises(ValueError, match="Tie"):
        OpenListPR(profile_tie, party_map=party_map_tie, m=1, tiebreak="deter")

    # With random tiebreak should succeed
    election = OpenListPR(profile_tie,  party_map=party_map_tie, m=1,tiebreak="random")
    assert len(election.get_elected()) == 1
    assert election.get_elected()[0] in {frozenset({"A1"}), frozenset({"B1"})}

def test_party_exhaustion():
    """Solo party seats its only nominee, then drops out."""
    election = OpenListPR(profile_exhaustion, party_map=party_map_exhaustion, m=2, tiebreak="random")

    assert election.election_states[1].scores["Solo"] == 12
    assert election.election_states[2].scores["Solo"] == 0

def test_state_list_and_helpers():    
    election = OpenListPR(profile_simple,party_map=party_map_simple,  m=3, tiebreak="random")
    
    # three rounds = three stored states (+ initial state)
    assert len(election.election_states) == 4
    assert election.election_states[0].scores == states_simple[0].scores
    assert election.election_states[1].scores == states_simple[1].scores
    assert election.election_states[2].scores == states_simple[2].scores

    assert election.get_profile(0) == profile_simple
    prof1, state1 = election.get_step(1)
    assert state1 == election.election_states[1]
    assert next(iter(winners_simple[1])) in prof1.candidates 

def test_error_conditions():
    with pytest.raises(ValueError, match="must be positive"):
        OpenListPR(profile_simple, party_map=party_map_simple, m=0)

def test_init():
    e = OpenListPR(profile_simple, party_map=party_map_simple, m=3, tiebreak="random")
    assert [c for c in e.get_elected()] == winners_simple

def test_ties():
    e_random = OpenListPR(profile_tie, party_map=party_map_tie,  m=1,tiebreak="random")
    assert len([c for s in e_random.get_elected() for c in s]) == 1

def test_state_list():
    e = OpenListPR(profile_simple, party_map=party_map_simple, m=2, tiebreak="random")
    assert e.election_states == states_simple

def test_get_profile():
    e = OpenListPR(profile_simple,  party_map=party_map_simple, m=2,tiebreak="random")
    assert e.get_profile(0) == profile_simple
    assert e.get_profile(1) == profile_simple_round_1

def test_get_step():
    e = OpenListPR(profile_simple,  party_map=party_map_simple, m=2,tiebreak="random")
    print(e.get_step(0))
    print(e.get_step(1))
    print(profile_simple_round_1, states_simple[1])
    assert e.get_step(1) == (profile_simple_round_1, states_simple[1])


def test_get_elected():
    e = OpenListPR(profile_simple, party_map=party_map_simple, m=2)
    assert e.get_elected(0) == tuple()
    assert e.get_elected(1) == (frozenset({"G1"}),)
    assert e.get_elected(2) == (frozenset({"G1"}),frozenset({"R1"}))

def test_get_eliminated():
    e = OpenListPR(profile_simple, party_map=party_map_simple, tiebreak="random", m=2)
    assert e.get_eliminated(0) == tuple()
    assert e.get_eliminated(1) == tuple()

def test_get_remaining():
    e = OpenListPR(profile_simple, party_map=party_map_simple, tiebreak="random", m=2)
    assert e.get_remaining(0) == (frozenset({"G1"}), frozenset({"G2"}), frozenset({"R1"}), frozenset({"R2"}))
    assert e.get_remaining(1) == (frozenset({"G2"}), frozenset({"R1"}), frozenset({"R2"}))

def test_get_status_df():
    profile_no_ties = PreferenceProfile(
        ballots=[
            Ballot(scores={"A": 1}, weight=3),
            Ballot(scores={"B": 1}, weight=2),
            Ballot(scores= {"C": 1}, weight=1),
        ],
        candidates=("A", "B", "C"),
    )
    party_map = {"A": "Alpha", "B": "Beta", "C": "Charlie"}
    e = OpenListPR(profile_no_ties, party_map=party_map, tiebreak="random", m=2)

    df_0 = pd.DataFrame(
        {"Status": ["Remaining"] * 3, "Round": [0] * 3},
        index=["A", "B", "C"],
    )
    df_1 = pd.DataFrame(
        {"Status": ["Elected", "Remaining", "Remaining"], "Round": [1] * 3},
        index=["A", "B", "C"],
    )

    assert e.get_status_df(0).sort_index().equals(df_0.sort_index())
    assert e.get_status_df(1).sort_index().equals(df_1.sort_index())


def test_errors():
    # m must be positive
    with pytest.raises(ValueError, match="m must be positive"):
        OpenListPR(profile_simple, m=0, party_map=party_map_simple, tiebreak="random")

    # cannot elect more seats than there are candidates
    with pytest.raises(ValueError, match="Not enough candidates"):
        OpenListPR(profile_simple, m=5, party_map=party_map_simple, tiebreak="random")

def test_validate_profile():
    # ballot ranking too long
    bad1 = PreferenceProfile(
        ballots=[Ballot(scores={"X": 2, "Y": 1}, weight=10)]
    )
    with pytest.raises(TypeError, match="violates score limit"):
        OpenListPR(bad1, m=1, party_map={"X": "P", "Y": "P"})

## ---- Stress Testing ---- ##
n_ballots = 100_000
n_candidates = 30

# Build candidates
candidates = [f"C{i}" for i in range(n_candidates)]

# Built ballots
ballots = [Ballot(scores={random.choice(candidates): 1}, weight=1) for _ in range(n_ballots)]

profile_stress = PreferenceProfile(
    ballots=ballots,
    candidates=candidates,
)

def test_stress():
    time0 = time.time()
    # Stress test for OpenListPR with large profile
    election = OpenListPR(profile_stress, m=10, party_map={c: "Party{c}" for c in candidates}, tiebreak="random")
    time1 = time.time()
    print(f"Stress test completed in {time1 - time0:.2f} seconds")
    assert len(election.get_elected()) <= 10 
    time_stress = time1 - time0
    assert (time_stress) < 10