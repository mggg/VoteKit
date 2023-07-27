import votekit.election_types as et
from votekit.pref_profile import PreferenceProfile
from votekit.ballot import Ballot
from votekit.election_state import ElectionState
from fractions import Fraction


def list_equal(list1, list2):
    if len(list1) == len(list2):
        if all([l1 == l2 for l1, l2 in zip(list1, list2)]):
            return True
    return False


def read_profile(file):
    ballot_pool = []
    with open(file, "r") as f:
        for line in f:
            cands = line.strip().split(",")
            ballot_pool.append(
                Ballot(ranking=[set(c) for c in cands], weight=Fraction(1))
            )

    return PreferenceProfile(ballots=ballot_pool)


def equal_ElectionStates(state1, state2):
    return (
        list_equal(state1.get_all_winners(), state2.get_all_winners())
        and list_equal(state1.get_all_eliminated(), state2.get_all_eliminated())
        and list_equal(state1.get_rankings(), state2.get_rankings())
    )


def _test_Bloc(profile, seats, target_state):
    bloc_election = et.Bloc(profile=profile, seats=seats)
    outcome = bloc_election.run_election()
    return equal_ElectionStates(outcome, target_state)


def _test_SNTV(profile, seats, target_state):
    sntv_election = et.SNTV(profile=profile, seats=seats)
    outcome = sntv_election.run_election()
    return equal_ElectionStates(outcome, target_state)


def _test_Hybrid(profile, r1_cutoff, seats, target_state):
    hybrid_election = et.SNTV_STV_Hybrid(
        profile=profile,
        r1_cutoff=r1_cutoff,
        seats=seats,
        transfer=et.fractional_transfer,
    )
    outcome = hybrid_election.run_election()
    return equal_ElectionStates(outcome, target_state)


def test_battery_Bloc():
    test_profile = read_profile("data/zach_test_ballots.csv")

    bloc_target1 = ElectionState(
        curr_round=1,
        elected=["B"],
        eliminated=["H", "F", "E", "I", "G", "D", "C", "A"],
        remaining=[],
        profile=PreferenceProfile(),
    )
    bloc_test1 = _test_Bloc(profile=test_profile, seats=1, target_state=bloc_target1)

    bloc_target2 = ElectionState(
        curr_round=1,
        elected=["A", "B"],
        eliminated=["F", "I", "H", "G", "E", "D", "C"],
        remaining=[],
        profile=PreferenceProfile(),
    )
    bloc_test2 = _test_Bloc(profile=test_profile, seats=2, target_state=bloc_target2)

    bloc_target3 = ElectionState(
        curr_round=1,
        elected=["A", "B", "C", "D"],
        eliminated=["H", "F", "I", "G", "E"],
        remaining=[],
        profile=PreferenceProfile(),
    )
    bloc_test3 = _test_Bloc(profile=test_profile, seats=4, target_state=bloc_target3)

    assert bloc_test1 and bloc_test2 and bloc_test3


def test_battery_SNTV():
    test_profile = read_profile("data/zach_test_ballots.csv")

    sntv_target1 = ElectionState(
        curr_round=1,
        elected=["B"],
        eliminated=["H", "F", "E", "I", "G", "D", "C", "A"],
        remaining=[],
        profile=PreferenceProfile(),
    )
    sntv_test1 = _test_SNTV(profile=test_profile, seats=1, target_state=sntv_target1)

    sntv_target2 = ElectionState(
        curr_round=1,
        elected=["B", "A", "C", "D", "G"],
        eliminated=["H", "F", "E", "I"],
        remaining=[],
        profile=PreferenceProfile(),
    )
    sntv_test2 = _test_SNTV(profile=test_profile, seats=5, target_state=sntv_target2)

    assert sntv_test1 and sntv_test2


def test_battery_Hybrid():
    test_profile = read_profile("data/zach_test_ballots.csv")
    hybrid_target1 = ElectionState(
        curr_round=1,
        elected=["B"],
        eliminated=["H", "F", "E", "I", "G", "D", "C", "A"],
        remaining=[],
        profile=PreferenceProfile(),
    )
    hybrid_test1 = _test_Hybrid(
        profile=test_profile, r1_cutoff=3, seats=1, target_state=hybrid_target1
    )

    hybrid_target2 = ElectionState(
        curr_round=1,
        elected=["B", "A", "C"],
        eliminated=["H", "F", "E", "I", "G", "D"],
        remaining=[],
        profile=PreferenceProfile(),
    )
    hybrid_test2 = _test_Hybrid(
        profile=test_profile, r1_cutoff=5, seats=3, target_state=hybrid_target2
    )

    assert hybrid_test1 and hybrid_test2
