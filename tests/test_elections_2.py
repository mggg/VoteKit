from fractions import Fraction
from pathlib import Path

from votekit.ballot import Ballot
from votekit.cvr_loaders import load_csv
from votekit.election_state import ElectionState
from votekit.election_types import (
    Bloc,
    SNTV,
    SNTV_STV_Hybrid,
    DominatingSets,
    CondoBorda,
    Borda,
)
from votekit.pref_profile import PreferenceProfile
from votekit.utils import fractional_transfer


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data/csv"
TEST_PROFILE = load_csv(DATA_DIR / "test_election_B.csv")
dom_ballot_list = [
    Ballot(
        id=None, ranking=[{"C"}, {"A"}, {"D"}, {"E"}, {"B"}], weight=Fraction(10, 1)
    ),
    Ballot(
        id=None, ranking=[{"A"}, {"B"}, {"C"}, {"E"}, {"D"}], weight=Fraction(10, 1)
    ),
    Ballot(
        id=None, ranking=[{"D"}, {"A"}, {"B"}, {"C"}, {"E"}], weight=Fraction(10, 1)
    ),
]
DOM_TEST_PROFILE = PreferenceProfile(ballots=dom_ballot_list)


def equal_electionstates(state1, state2):
    assert state1.get_all_winners() == state2.get_all_winners()
    assert state1.get_all_eliminated() == state2.get_all_eliminated()
    assert state1.get_rankings() == state2.get_rankings()


def compare_io_bloc(profile, seats, target_state):
    bloc_election = Bloc(profile=profile, seats=seats)
    outcome = bloc_election.run_election()
    # Make assertions
    equal_electionstates(outcome, target_state)


def compare_io_sntv(profile, seats, target_state):
    sntv_election = SNTV(profile=profile, seats=seats)
    outcome = sntv_election.run_election()
    # Make assertions
    equal_electionstates(outcome, target_state)


def compare_io_hybrid(profile, r1_cutoff, seats, target_state):
    hybrid_election = SNTV_STV_Hybrid(
        profile=profile,
        r1_cutoff=r1_cutoff,
        seats=seats,
        transfer=fractional_transfer,
    )
    outcome = hybrid_election.run_election()
    # Make assertions
    equal_electionstates(outcome, target_state)


def compare_io_domset(profile, target_state):
    domset_election = DominatingSets(profile=profile)
    outcome = domset_election.run_election()
    # Make assertions
    equal_electionstates(outcome, target_state)


def compare_io_condoborda(profile, seats, target_state):
    condoborda_election = CondoBorda(
        profile=profile,
        seats=seats,
    )
    outcome = condoborda_election.run_election()
    # Make assertations
    equal_electionstates(outcome, target_state)


def compare_io_borda(profile, seats, score_vector, target_state):
    borda_election = Borda(profile=profile, seats=seats, score_vector=score_vector)
    outcome = borda_election.run_election()
    # Make assertations
    equal_electionstates(outcome, target_state)


def test_bloc_onewinner():
    bloc_target1 = ElectionState(
        curr_round=1,
        elected=["B"],
        eliminated=["F", "E", "I", "H", "G", "D", "C", "A"],
        remaining=[],
        profile=PreferenceProfile(),
    )
    compare_io_bloc(profile=TEST_PROFILE, seats=1, target_state=bloc_target1)


def test_bloc_twowinner():
    bloc_target2 = ElectionState(
        curr_round=1,
        elected=["A", "B"],
        eliminated=["F", "I", "H", "G", "E", "D", "C"],
        remaining=[],
        profile=PreferenceProfile(),
    )
    compare_io_bloc(profile=TEST_PROFILE, seats=2, target_state=bloc_target2)


def test_bloc_fourwinner():
    bloc_target3 = ElectionState(
        curr_round=1,
        elected=["A", "B", "C", "D"],
        eliminated=["H", "F", "I", "G", "E"],
        remaining=[],
        profile=PreferenceProfile(),
    )
    compare_io_bloc(profile=TEST_PROFILE, seats=4, target_state=bloc_target3)


def test_sntv_onewinner():
    sntv_target1 = ElectionState(
        curr_round=1,
        elected=["B"],
        eliminated=["F", "E", "I", "H", "G", "D", "C", "A"],
        remaining=[],
        profile=PreferenceProfile(),
    )
    compare_io_sntv(profile=TEST_PROFILE, seats=1, target_state=sntv_target1)


def test_sntv_fivewinner():
    sntv_target2 = ElectionState(
        curr_round=1,
        elected=["B", "A", "C", "D", "G"],
        eliminated=["F", "E", "I", "H"],
        remaining=[],
        profile=PreferenceProfile(),
    )
    compare_io_sntv(profile=TEST_PROFILE, seats=5, target_state=sntv_target2)


# def test_hybrid_cutfour_onewinner():
#     # STV is still stochastic for winners
#     # So this test needs to be flexible on 2 possible outcomes
#     hybrid_target1a = ElectionState(
#         curr_round=1,
#         elected=["A"],
#         eliminated=["F", "E", "I", "H", "G", "D", "C", "B"],
#         remaining=[],
#         profile=PreferenceProfile(),
#     )
#     compare_io_hybrid(
#         profile=TEST_PROFILE, r1_cutoff=4, seats=1, target_state=hybrid_target1
#     )


def test_hybrid_cutfive_threewinner():
    hybrid_target2 = ElectionState(
        curr_round=1,
        elected=["B", "A", "C"],
        eliminated=["F", "E", "I", "H", "G", "D"],
        remaining=[],
        profile=PreferenceProfile(),
    )
    compare_io_hybrid(
        profile=TEST_PROFILE, r1_cutoff=5, seats=3, target_state=hybrid_target2
    )


def test_dom_set_fivecand():
    dom_target1 = ElectionState(
        curr_round=1,
        elected=[{"A"}],
        eliminated=[{"E"}, {"B", "C", "D"}],
        remaining=list(),
        profile=PreferenceProfile(),
    )
    compare_io_domset(profile=DOM_TEST_PROFILE, target_state=dom_target1)


def test_condoborda_fivecand():
    condoborda_target1 = ElectionState(
        curr_round=1,
        elected=["A", "C", "D"],
        eliminated=["B", "E"],
        remaining=list(),
        profile=PreferenceProfile(),
    )
    compare_io_condoborda(
        profile=DOM_TEST_PROFILE, seats=3, target_state=condoborda_target1
    )


def test_borda_three_winner():
    borda_target1 = ElectionState(
        curr_round=1,
        elected=["A", "B", "C"],
        eliminated=["F", "I", "H", "G", "E", "D"],
        remaining=list(),
        profile=PreferenceProfile(),
    )
    compare_io_borda(
        profile=TEST_PROFILE, seats=3, score_vector=None, target_state=borda_target1
    )
