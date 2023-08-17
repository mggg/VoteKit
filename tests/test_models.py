import pytest
from votekit.models import Simulation
from votekit.ballot_generator_OLD import (
    PlackettLuce,
    BradleyTerry,
    AlternatingCrossover,
)
from votekit.profile import PreferenceProfile
from unittest.mock import MagicMock


class DummyGenerated(Simulation):
    ballots = {"PL": PlackettLuce, "BT": BradleyTerry, "AC": AlternatingCrossover}

    def run_simulation():
        pass

    def sim_election():
        pass


class DummyActual(Simulation):
    ballots = MagicMock(spec=PreferenceProfile)

    def run_simulation():
        pass

    def sim_election():
        pass


def test_gen_ballots():
    model = DummyGenerated()
    cands = {"R": ["A1", "B1", "C1"], "D": ["A2", "B2"]}
    specs = {
        "blocs": {"R": 0.6, "D": 0.4},
        "cohesion": {"R": 0.7, "D": 0.6},
        "alphas": {"R": {"R": 0.5, "D": 1}, "D": {"R": 1, "D": 0.5}},
        "crossover": {"R": {"D": 0.5}, "D": {"R": 0.6}},
    }
    profiles = model.generate_ballots(num_ballots=10, candidates=cands, params=specs)
    name, model = profiles[0]
    assert name == "PL"
    assert isinstance(model, PreferenceProfile)
    # check that a profile exists for each ballot generator
    assert len(profiles) == 3


def test_gen_with_real_data():
    model = DummyActual()
    cands = {"R": ["A1", "B1", "C1"], "D": ["A2", "B2"]}
    params = {}
    with pytest.raises(TypeError):
        model.generate_ballots(num_ballots=10, candidates=cands, hyperparams=params)
