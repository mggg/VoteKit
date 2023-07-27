from votekit.ballot_generator import (
    IC,
    IAC,
    PlackettLuce,
    BradleyTerry,
    AlternatingCrossover,
    CambridgeSampler,
    OneDimSpatial,
)
from votekit.profile import PreferenceProfile
from pathlib import Path

# import pytest


def test_IC_completion():
    ic = IC(
        number_of_ballots=100, candidates=["W1", "W2", "C1", "C2"], ballot_length=None
    )
    profile = ic.generate_profile()
    # return profile is PreferenceProfile
    assert type(profile) is PreferenceProfile


def test_IAC_completion():
    iac = IAC(
        number_of_ballots=100, candidates=["W1", "W2", "C1", "C2"], ballot_length=None
    )
    profile = iac.generate_profile()
    # return profile is PreferenceProfile
    assert type(profile) is PreferenceProfile


def test_PL_completion():
    pl = PlackettLuce(
        number_of_ballots=100,
        candidates=["W1", "W2", "C1", "C2"],
        ballot_length=None,
        pref_interval_by_slate={
            "W": {"W1": 0.4, "W2": 0.3, "C1": 0.2, "C2": 0.1},
            "C": {"W1": 0.2, "W2": 0.2, "C1": 0.3, "C2": 0.3},
        },
        slate_voter_prop={"W": 0.7, "C": 0.3},
    )
    profile = pl.generate_profile()
    # return profile is PreferenceProfile
    assert type(profile) is PreferenceProfile


def test_BT_completion():
    bt = BradleyTerry(
        number_of_ballots=100,
        candidates=["W1", "W2", "C1", "C2"],
        ballot_length=None,
        pref_interval_by_slate={
            "W": {"W1": 0.4, "W2": 0.3, "C1": 0.2, "C2": 0.1},
            "C": {"W1": 0.2, "W2": 0.2, "C1": 0.3, "C2": 0.3},
        },
        slate_voter_prop={"W": 0.7, "C": 0.3},
    )
    profile = bt.generate_profile()
    # return profile is PreferenceProfile
    assert type(profile) is PreferenceProfile


def test_AC_completion():
    ac = AlternatingCrossover(
        number_of_ballots=100,
        candidates=["W1", "W2", "C1", "C2"],
        ballot_length=None,
        slate_to_candidate={"W": ["W1", "W2"], "C": ["C1", "C2"]},
        pref_interval_by_slate={
            "W": {"W1": 0.4, "W2": 0.3, "C1": 0.2, "C2": 0.1},
            "C": {"W1": 0.2, "W2": 0.2, "C1": 0.3, "C2": 0.3},
        },
        slate_voter_prop={"W": 0.7, "C": 0.3},
        slate_crossover_rate={"W": {"C": 0.3}, "C": {"W": 0.1}},
    )
    profile = ac.generate_profile()
    # return profile is PreferenceProfile
    assert type(profile) is PreferenceProfile


def test_Cambridge_completion():
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / "data/"
    path = Path(DATA_DIR, "Cambridge_09to17_ballot_types.p")

    cs = CambridgeSampler(
        number_of_ballots=100,
        candidates=["W1", "W2", "C1", "C2"],
        ballot_length=None,
        slate_to_candidate={"W": ["W1", "W2"], "C": ["C1", "C2"]},
        pref_interval_by_slate={
            "W": {"W1": 0.4, "W2": 0.3, "C1": 0.2, "C2": 0.1},
            "C": {"W1": 0.2, "W2": 0.2, "C1": 0.3, "C2": 0.3},
        },
        slate_voter_prop={"W": 0.7, "C": 0.3},
        slate_crossover_rate={"W": {"C": 0.3}, "C": {"W": 0.1}},
        path=path,
    )
    profile = cs.generate_profile()
    # return profile is PreferenceProfile
    assert type(profile) is PreferenceProfile


def test_1D_completion():
    ods = OneDimSpatial(
        number_of_ballots=100, candidates=["W1", "W2", "C1", "C2"], ballot_length=None
    )
    profile = ods.generate_profile()
    # return profile is PreferenceProfile
    assert type(profile) is PreferenceProfile


if __name__ == "__main__":
    test_IC_completion()
    test_IAC_completion()
    test_PL_completion()
    test_BT_completion()
    test_AC_completion()
    test_Cambridge_completion()
    test_1D_completion()
