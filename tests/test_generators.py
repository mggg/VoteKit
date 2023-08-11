from votekit.ballot_generator import PlackettLuce, BradleyTerry, AlternatingCrossover
from votekit.profile import PreferenceProfile
import pytest
import math

twobloc = {
    "blocs": {"R": 0.6, "D": 0.4},
    "cohesion": {"R": 0.7, "D": 0.6},
    "alphas": {"R": {"R": 0.5, "D": 1}, "D": {"R": 1, "D": 0.5}},
}

cands = {"R": ["A1", "B1", "C1"], "D": ["A2", "B2"]}
cands_lst = ["A", "B", "C"]

test_slate = {"R": {"A1": 0.1, "B1": 0.5, "C1": 0.4}, "D": {"A2": 0.2, "B2": 0.5}}
test_voter_prop = {"R": 0.5, "D": 0.5}


def test_setparams_pl():
    pl = PlackettLuce(number_of_ballots=3, candidates=cands, hyperparameters=twobloc)
    # check params were set
    assert pl.slate_voter_prop == {"R": 0.6, "D": 0.4}
    interval = pl.pref_interval_by_slate
    # check if intervals add up to one
    assert math.isclose(sum(interval["R"].values()), 1)
    assert math.isclose(sum(interval["D"].values()), 1)


def test_bad_cands_input():
    with pytest.raises(TypeError):
        PlackettLuce(number_of_ballots=3, candidates=cands_lst, hyperparameters=twobloc)


def test_pl_both_inputs():
    gen = PlackettLuce(
        number_of_ballots=3,
        candidates=cands,
        pref_interval_by_slate=test_slate,
        slate_voter_prop=test_voter_prop,
        hyperparameters=twobloc,
    )
    # check that this attribute matches hyperparam input
    assert gen.slate_voter_prop == {"R": 0.6, "D": 0.4}


def test_bt_single_bloc():
    bloc = {
        "blocs": {"R": 1.0},
        "cohesion": {"R": 0.7},
        "alphas": {"R": {"R": 0.5, "D": 1}},
    }
    cands = {"R": ["X", "Y", "Z"], "D": ["A", "B"]}

    gen = BradleyTerry(number_of_ballots=3, candidates=cands, hyperparameters=bloc)
    interval = gen.pref_interval_by_slate
    assert math.isclose(sum(interval["R"].values()), 1)


def test_incorrect_blocs():
    params = {
        "blocs": {"R": 0.7, "D": 0.4},
        "cohesion": {"R": 0.7, "D": 0.6},
        "alphas": {"R": {"R": 0.5, "D": 1}, "D": {"R": 1, "D": 0.5}},
    }
    cands = {"R": ["A1", "B1", "C1"], "D": ["A2", "B2"]}

    with pytest.raises(ValueError):
        PlackettLuce(number_of_ballots=3, candidates=cands, hyperparameters=params)


def test_ac_profile_from_params():
    params = {
        "blocs": {"R": 0.6, "D": 0.4},
        "cohesion": {"R": 0.7, "D": 0.6},
        "alphas": {"R": {"R": 0.5, "D": 1}, "D": {"R": 1, "D": 0.5}},
        "crossover": {"R": {"D": 0.5}, "D": {"R": 0.6}},
    }
    cands = {"R": ["A1", "B1", "C1"], "D": ["A2", "B2"]}
    ac = AlternatingCrossover(
        number_of_ballots=3, candidates=cands, hyperparameters=params
    )
    ballots = ac.generate_profile()
    assert isinstance(ballots, PreferenceProfile)
