from votekit.cvr_loaders import load_scottish
from votekit.elections import STV, fastSTV
import numpy as np
from tqdm import tqdm  # tqdm is not currently a dependency... oh well
import os

simult = False


def is_same_election_state(e1, e2):
    return (
        e1.round_number == e2.round_number
        and e1.remaining == e2.remaining
        and e1.elected == e2.elected
        and e1.eliminated == e2.eliminated
        and e1.tiebreaks == e2.tiebreaks
        and e1.scores.keys() == e2.scores.keys()
        and all(np.isclose(e1.scores[k], e2.scores[k]) for k in e1.scores.keys())
    )


def check_election_states(profile):
    new = fastSTV(profile, m=2, simultaneous=simult)
    old = STV(profile, m=2, simultaneous=simult)

    new_states = new.election_states
    old_states = old.election_states

    if len(new_states) != len(old_states):
        print("Mismatch in number of election states!")
        print("Old number of states:", len(old_states))
        print("New number of states:", len(new_states))
        return True

    for i, (new_state, old_state) in enumerate(zip(new_states, old_states)):
        if not is_same_election_state(new_state, old_state):
            print("Mismatch found during election_states!")
            print("Old:")
            print(old_state)
            print("New:")
            print(new_state)
            print("State number:", i)
            return True


def check_full_election(profile):
    new = fastSTV(profile, m=2, simultaneous=simult)
    old = STV(profile, m=2, simultaneous=simult)

    num_rounds = len(new._fpv_by_round)

    for i in range(num_rounds):
        # check that get_rankings returns the same pandas df string
        if old.get_status_df(i).to_string(
            index=True, justify="justify"
        ) != new.get_status_df(i).to_string(index=True, justify="justify"):
            print("Mismatch found during get_status_df!")
            print("Old:")
            print(old.get_status_df(i).to_string(index=True, justify="justify"))
            print("New:")
            print(new.get_status_df(i).to_string(index=True, justify="justify"))
            print("Round number:", i)
            return True
        # check that get_elected returns the same thing
        if old.get_elected(i) != new.get_elected(i):
            print("Mismatch found during get_elected!")
            print("Old:")
            print(old.get_elected(i))
            print("New:")
            print(new.get_elected(i))
            print("Round number:", i)
            return True
        # check that get_remaining returns the same thing
        if old.get_remaining(i) != new.get_remaining(i):
            print("Mismatch found during get_remaining!")
            print("Old:")
            print(old.get_remaining(i))
            print("New:")
            print(new.get_remaining(i))
            print("Round number:", i)
            return True
        # check that get_eliminated returns the same thing
        if old.get_eliminated(i) != new.get_eliminated(i):
            print("Mismatch found during get_eliminated!")
            print("Old:")
            print(old.get_eliminated(i))
            print("New:")
            print(new.get_eliminated(i))
            print("Round number:", i)
            return True
        # check that get_ranking returns the same thing
        if old.get_ranking(i) != new.get_ranking(i):
            print("Mismatch found during get_ranking!")
            print("Old:")
            print(old.get_ranking(i))
            print("New:")
            print(new.get_ranking(i))
            print("Round number:", i)
            return True


def test_scot_methods():
    for filename in tqdm(os.listdir("data/8_cands")):
        if filename.endswith(".csv"):
            pf1 = load_scottish(f"data/8_cands/{filename}")[0]
            if check_full_election(pf1):
                print(f"Election {filename} failed")
                assert False
    assert True


def test_scot_states():
    for filename in tqdm(os.listdir("data/8_cands")):
        if filename.endswith(".csv"):
            pf1 = load_scottish(f"data/8_cands/{filename}")[0]
            if check_election_states(pf1):
                print(f"Election {filename} failed")
                assert False
    assert True
