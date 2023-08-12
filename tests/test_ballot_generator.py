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
import scipy.stats as stats
import numpy as np
import itertools as it
import votekit.ballot_generator as bg
from votekit.ballot import Ballot


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
        bloc_voter_prop={"W": 0.7, "C": 0.3},
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
        bloc_voter_prop={"W": 0.7, "C": 0.3},
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
        pref_interval_by_bloc={
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


def binomial_confidence_interval(probability, n_attempts, alpha=0.99):
    # Calculate the mean and standard deviation of the binomial distribution
    mean = n_attempts * probability
    std_dev = np.sqrt(n_attempts * probability * (1 - probability))

    # Calculate the confidence interval
    z_score = stats.norm.ppf(
        alpha + 0.5 * (1 - alpha)
    )  # Z-score for 99% confidence level
    margin_of_error = z_score * (std_dev / np.sqrt(n_attempts))
    conf_interval = (mean - margin_of_error, mean + margin_of_error)

    return conf_interval


def do_ballot_probs_match_ballot_dist(
    ballot_prob_dict: dict, generated_profile: PreferenceProfile
):
    n_ballots = generated_profile.num_ballots()
    ballot_conf_dict = {
        b: binomial_confidence_interval(p, n_attempts=n_ballots)
        for b, p in ballot_prob_dict.items()
    }
    for b in ballot_conf_dict.keys():
        if not (ballot_conf_dict[b][0] < generated_profile[b] < ballot_conf_dict[b][1]):
            return False
    return True


def test_ic_distribution():
    # Set-up
    number_of_ballots = 1000
    ballot_length = 4
    candidates = ["A", "B", "C", "D"]

    # Find ballot probs
    possible_rankings = it.permutations(candidates, ballot_length)
    possible_ballots = [
        Ballot([{cand} for cand in ranking], weight=1) for ranking in possible_rankings
    ]
    ballot_prob_dict = {b: 1 / 6 for b in possible_ballots}

    # Generate ballots
    generated_profile = bg.IC(
        number_of_ballots=number_of_ballots,
        ballot_length=ballot_length,
        candidates=candidates,
    )

    # Test
    assert do_ballot_probs_match_ballot_dist(ballot_prob_dict, generated_profile)


def test_iac_completion():
    ...


def test_pl_distribution():
    # Set-up
    number_of_ballots = 1000
    ballot_length = 4
    candidates = ["A", "B", "C", "D"]

    # Find ballot probs
    possible_rankings = it.permutations(candidates, ballot_length)
    possible_ballots = [
        Ballot([{cand} for cand in ranking], weight=1) for ranking in possible_rankings
    ]
    ballot_prob_dict = {b: 1 / 6 for b in possible_ballots}

    # Generate ballots
    generated_profile = bg.IC(
        number_of_ballots=number_of_ballots,
        ballot_length=ballot_length,
        candidates=candidates,
    )

    # Test
    assert do_ballot_probs_match_ballot_dist(ballot_prob_dict, generated_profile)


if __name__ == "__main__":
    test_IC_completion()
    test_IAC_completion()
    test_PL_completion()
    test_BT_completion()
    test_AC_completion()
    test_Cambridge_completion()
    test_1D_completion()
