import votekit.election_types as et
from votekit.ballot import Ballot
from votekit.profile import PreferenceProfile
from votekit.election_state import ElectionState
from votekit.cvr_loaders import rank_column_csv

from fractions import Fraction
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TEST_PROFILE = rank_column_csv(DATA_DIR / "test_election_B.csv")
condo_ballot_list = [
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
CONDO_TEST_PROFILE = PreferenceProfile(ballots=condo_ballot_list)


def equal_electionstates(state1, state2):
    print(state1.get_all_winners(), state1.get_all_eliminated(), state1.get_rankings())
    print(state2.get_all_winners(), state2.get_all_eliminated(), state2.get_rankings())

    return (
        (state1.get_all_winners() == state2.get_all_winners())
        and (state1.get_all_eliminated() == state2.get_all_eliminated())
        and (state1.get_rankings() == state2.get_rankings())
    )


def compare_io_bloc(profile, seats, target_state):
    bloc_election = et.Bloc(profile=profile, seats=seats)
    outcome = bloc_election.run_election()
    return equal_electionstates(outcome, target_state)


def compare_io_sntv(profile, seats, target_state):
    sntv_election = et.SNTV(profile=profile, seats=seats)
    outcome = sntv_election.run_election()
    return equal_electionstates(outcome, target_state)


def compare_io_hybrid(profile, r1_cutoff, seats, target_state):
    hybrid_election = et.SNTV_STV_Hybrid(
        profile=profile,
        r1_cutoff=r1_cutoff,
        seats=seats,
        transfer=et.fractional_transfer,
    )
    outcome = hybrid_election.run_election()
    return equal_electionstates(outcome, target_state)


def compare_io_domset(profile, target_state):
    domset_election = et.DominatingSets(profile=profile)
    outcome = domset_election.run_election()

    print(outcome.get_rankings())
    print(target_state.get_rankings())
    return equal_electionstates(outcome, target_state)


def compare_io_condoborda(profile, seats, target_state):
    condoborda_election = et.CondoBorda(
        profile=profile,
        seats=seats,
    )
    outcome = condoborda_election.run_election()

    print(outcome.get_rankings())
    print(target_state.get_rankings())
    return equal_electionstates(outcome, target_state)


def test_bloc_onewinner():
    bloc_target1 = ElectionState(
        curr_round=1,
        elected=["B"],
        eliminated=["F", "E", "I", "H", "G", "D", "A", "C"],
        remaining=[],
        profile=PreferenceProfile(),
    )
    assert compare_io_bloc(profile=TEST_PROFILE, seats=1, target_state=bloc_target1)


def test_bloc_twowinner():
    bloc_target2 = ElectionState(
        curr_round=1,
        elected=["A", "B"],
        eliminated=["F", "I", "H", "G", "E", "D", "C"],
        remaining=[],
        profile=PreferenceProfile(),
    )
    assert compare_io_bloc(profile=TEST_PROFILE, seats=2, target_state=bloc_target2)


def test_bloc_fourwinner():
    bloc_target3 = ElectionState(
        curr_round=1,
        elected=["A", "B", "C", "D"],
        eliminated=["H", "F", "I", "G", "E"],
        remaining=[],
        profile=PreferenceProfile(),
    )
    assert compare_io_bloc(profile=TEST_PROFILE, seats=4, target_state=bloc_target3)


def test_sntv_onewinner():
    sntv_target1 = ElectionState(
        curr_round=1,
        elected=["B"],
        eliminated=["F", "E", "I", "H", "G", "D", "A", "C"],
        remaining=[],
        profile=PreferenceProfile(),
    )
    assert compare_io_sntv(profile=TEST_PROFILE, seats=1, target_state=sntv_target1)


def test_sntv_fivewinner():
    sntv_target2 = ElectionState(
        curr_round=1,
        elected=["B", "C", "A", "D", "G"],
        eliminated=["F", "E", "I", "H"],
        remaining=[],
        profile=PreferenceProfile(),
    )
    assert compare_io_sntv(profile=TEST_PROFILE, seats=5, target_state=sntv_target2)


def test_hybrid_cutfour_onewinner():
    hybrid_target1 = ElectionState(
        curr_round=1,
        elected=["B"],
        eliminated=["F", "E", "I", "H", "G", "D", "C", "A"],
        remaining=[],
        profile=PreferenceProfile(),
    )
    assert compare_io_hybrid(
        profile=TEST_PROFILE, r1_cutoff=4, seats=1, target_state=hybrid_target1
    )


def test_hybrid_cutfive_threewinner():
    hybrid_target2 = ElectionState(
        curr_round=1,
        elected=["B", "A", "C"],
        eliminated=["F", "E", "I", "H", "G", "D"],
        remaining=[],
        profile=PreferenceProfile(),
    )
    assert compare_io_hybrid(
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
    assert compare_io_domset(profile=CONDO_TEST_PROFILE, target_state=dom_target1)


def test_condoborda_fivecand():
    condoborda_target1 = ElectionState(
        curr_round=1,
        elected=["A", "C", "D"],
        eliminated=["B", "E"],
        remaining=list(),
        profile=PreferenceProfile(),
    )
    assert compare_io_condoborda(
        profile=CONDO_TEST_PROFILE, seats=3, target_state=condoborda_target1
    )
