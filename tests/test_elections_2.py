from pathlib import Path

from votekit.cvr_loaders import load_csv
from votekit.election_state import ElectionState
import votekit.elections.election_types as et
from votekit.pref_profile import PreferenceProfile


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data/csv"
TEST_PROFILE_B = load_csv(DATA_DIR / "test_election_B.csv")
TEST_PROFILE_C = load_csv(DATA_DIR / "test_election_C.csv")


def equal_electionstates(state1, state2):
    assert state1.winners() == state2.winners()
    assert state1.eliminated() == state2.eliminated()
    assert state1.rankings() == state2.rankings()


def compare_io_bloc(profile, seats, target_state):
    bloc_election = et.Bloc(profile=profile, seats=seats, tiebreak="none")
    outcome = bloc_election.run_election()
    # Make assertions
    equal_electionstates(outcome, target_state)


def compare_io_sntv(profile, seats, target_state):
    sntv_election = et.SNTV(profile=profile, seats=seats, tiebreak="none")
    outcome = sntv_election.run_election()
    # Make assertions
    equal_electionstates(outcome, target_state)


def compare_io_hybrid(profile, r1_cutoff, seats, target_state):
    hybrid_election = et.SNTV_STV_Hybrid(
        profile=profile,
        r1_cutoff=r1_cutoff,
        seats=seats,
        transfer=et.fractional_transfer,
        tiebreak="none",
    )
    outcome = hybrid_election.run_election()
    # Make assertions
    equal_electionstates(outcome, target_state)


def compare_io_domset(profile, target_state):
    domset_election = et.DominatingSets(profile=profile)
    outcome = domset_election.run_election()
    # Make assertions
    equal_electionstates(outcome, target_state)


def compare_io_condoborda(profile, seats, target_state):
    condoborda_election = et.CondoBorda(profile=profile, seats=seats, tiebreak="none")
    outcome = condoborda_election.run_election()
    # Make assertations
    equal_electionstates(outcome, target_state)


def compare_io_borda(profile, seats, score_vector, target_state):
    borda_election = et.Borda(
        profile=profile, seats=seats, score_vector=score_vector, tiebreak="none"
    )
    outcome = borda_election.run_election()
    # Make assertations
    equal_electionstates(outcome, target_state)


def test_bloc_onewinner():
    bloc_target1 = ElectionState(
        curr_round=1,
        elected=[{"B"}],
        eliminated_cands=[{"A", "C"}, {"D"}, {"G", "H", "I"}, {"E", "F"}],
        remaining=[],
        profile=PreferenceProfile(),
    )
    compare_io_bloc(profile=TEST_PROFILE_B, seats=1, target_state=bloc_target1)


def test_bloc_fivewinner():
    bloc_target3 = ElectionState(
        curr_round=1,
        elected=[{"A", "B"}, {"C"}, {"D", "E"}],
        eliminated_cands=[{"G", "I"}, {"F", "H"}],
        remaining=[],
        profile=PreferenceProfile(),
    )
    compare_io_bloc(profile=TEST_PROFILE_B, seats=5, target_state=bloc_target3)


def test_sntv_onewinner():
    sntv_target1 = ElectionState(
        curr_round=1,
        elected=[{"B"}],
        eliminated_cands=[{"A", "C"}, {"D"}, {"G", "H", "I"}, {"E", "F"}],
        remaining=[],
        profile=PreferenceProfile(),
    )
    compare_io_sntv(profile=TEST_PROFILE_B, seats=1, target_state=sntv_target1)


def test_sntv_fourwinner():
    sntv_target2 = ElectionState(
        curr_round=1,
        elected=[{"B"}, {"A", "C"}, {"D"}],
        eliminated_cands=[{"G", "H", "I"}, {"E", "F"}],
        remaining=[],
        profile=PreferenceProfile(),
    )
    compare_io_sntv(profile=TEST_PROFILE_B, seats=4, target_state=sntv_target2)


def test_hybrid_cutfour_twowinner():
    hybrid_target2 = ElectionState(
        curr_round=1,
        elected=[{"B", "A"}],
        eliminated_cands=[{"C"}, {"D"}, {"G", "H", "I"}, {"E", "F"}],
        remaining=[],
        profile=PreferenceProfile(),
    )
    compare_io_hybrid(
        profile=TEST_PROFILE_B, r1_cutoff=4, seats=2, target_state=hybrid_target2
    )


def test_dom_set_fivecand():
    dom_target1 = ElectionState(
        curr_round=1,
        elected=[{"A"}],
        eliminated_cands=[{"B", "C", "D"}, {"E"}],
        remaining=list(),
        profile=PreferenceProfile(),
    )
    compare_io_domset(profile=TEST_PROFILE_C, target_state=dom_target1)


def test_condoborda_fivecand():
    condoborda_target1 = ElectionState(
        curr_round=1,
        elected=[{"A"}, {"C"}, {"D"}],
        eliminated_cands=[{"B"}, {"E"}],
        remaining=list(),
        profile=PreferenceProfile(),
    )
    compare_io_condoborda(
        profile=TEST_PROFILE_C, seats=3, target_state=condoborda_target1
    )


def test_borda_three_winner():
    borda_target1 = ElectionState(
        curr_round=1,
        elected=[{"A", "B"}, {"C"}],
        eliminated_cands=[{"D"}, {"E", "G"}, {"H", "I"}, {"F"}],
        remaining=list(),
        profile=PreferenceProfile(),
    )
    compare_io_borda(
        profile=TEST_PROFILE_B, seats=3, score_vector=None, target_state=borda_target1
    )
