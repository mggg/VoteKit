import numpy as np

from votekit.ballot_generator import (
    ImpartialAnonymousCulture,
    ImpartialCulture,
    name_PlackettLuce,
    name_BradleyTerry,
    AlternatingCrossover,
    CambridgeSampler,
    OneDimSpatial,
    BallotSimplex,
    slate_PlackettLuce,
    slate_BradleyTerry,
    name_Cumulative,
)
from votekit.pref_profile import PreferenceProfile
from votekit.pref_interval import PreferenceInterval

# set seed for more consistent tests
np.random.seed(8675309)


def test_IC_completion():
    ic = ImpartialCulture(candidates=["W1", "W2", "C1", "C2"])
    profile = ic.generate_profile(number_of_ballots=100)
    assert type(profile) is PreferenceProfile
    assert profile.num_ballots() == 100


def test_IAC_completion():
    iac = ImpartialAnonymousCulture(candidates=["W1", "W2", "C1", "C2"])
    profile = iac.generate_profile(number_of_ballots=100)
    assert type(profile) is PreferenceProfile
    assert profile.num_ballots() == 100


def test_NPL_completion():
    pl = name_PlackettLuce(
        candidates=["W1", "W2", "C1", "C2"],
        pref_intervals_by_bloc={
            "W": {
                "W": PreferenceInterval({"W1": 0.4, "W2": 0.3}),
                "C": PreferenceInterval({"C1": 0.2, "C2": 0.1}),
            },
            "C": {
                "W": PreferenceInterval({"W1": 0.2, "W2": 0.2}),
                "C": PreferenceInterval({"C1": 0.3, "C2": 0.3}),
            },
        },
        bloc_voter_prop={"W": 0.7, "C": 0.3},
        cohesion_parameters={"W": {"W": 0.7, "C": 0.3}, "C": {"C": 0.9, "W": 0.1}},
    )
    profile = pl.generate_profile(number_of_ballots=100)
    assert type(profile) is PreferenceProfile

    result = pl.generate_profile(number_of_ballots=100, by_bloc=True)
    assert type(result) is tuple
    profile_dict, agg_prof = result
    assert isinstance(profile_dict, dict)
    assert (type(profile_dict["W"])) is PreferenceProfile
    assert type(agg_prof) is PreferenceProfile
    assert agg_prof.num_ballots() == 100


def test_name_Cumulative_completion():
    cumu = name_Cumulative(
        candidates=["W1", "W2", "C1", "C2"],
        pref_intervals_by_bloc={
            "W": {
                "W": PreferenceInterval({"W1": 0.4, "W2": 0.3}),
                "C": PreferenceInterval({"C1": 0.2, "C2": 0.1}),
            },
            "C": {
                "W": PreferenceInterval({"W1": 0.2, "W2": 0.2}),
                "C": PreferenceInterval({"C1": 0.3, "C2": 0.3}),
            },
        },
        bloc_voter_prop={"W": 0.7, "C": 0.3},
        num_votes=3,
        cohesion_parameters={"W": {"W": 0.7, "C": 0.3}, "C": {"C": 0.9, "W": 0.1}},
    )
    profile = cumu.generate_profile(number_of_ballots=100)
    assert type(profile) is PreferenceProfile

    result = cumu.generate_profile(number_of_ballots=100, by_bloc=True)
    assert type(result) is tuple
    profile_dict, agg_prof = result
    assert isinstance(profile_dict, dict)
    assert (type(profile_dict["W"])) is PreferenceProfile
    assert type(agg_prof) is PreferenceProfile
    assert agg_prof.num_ballots() == 100


def test_NBT_completion():
    bt = name_BradleyTerry(
        candidates=["W1", "W2", "C1", "C2"],
        pref_intervals_by_bloc={
            "W": {
                "W": PreferenceInterval({"W1": 0.4, "W2": 0.3}),
                "C": PreferenceInterval({"C1": 0.2, "C2": 0.1}),
            },
            "C": {
                "W": PreferenceInterval({"W1": 0.2, "W2": 0.2}),
                "C": PreferenceInterval({"C1": 0.3, "C2": 0.3}),
            },
        },
        bloc_voter_prop={"W": 0.7, "C": 0.3},
        cohesion_parameters={"W": {"W": 0.7, "C": 0.3}, "C": {"C": 0.9, "W": 0.1}},
    )
    profile = bt.generate_profile(number_of_ballots=100)
    assert type(profile) is PreferenceProfile

    result = bt.generate_profile(number_of_ballots=100, by_bloc=True)
    assert type(result) is tuple
    profile_dict, agg_prof = result
    assert isinstance(profile_dict, dict)
    assert (type(profile_dict["W"])) is PreferenceProfile
    assert type(agg_prof) is PreferenceProfile
    assert agg_prof.num_ballots() == 100


def test_SPL_completion():
    sp = slate_PlackettLuce(
        candidates=["W1", "W2", "C1", "C2"],
        slate_to_candidates={"W": ["W1", "W2"], "C": ["C1", "C2"]},
        pref_intervals_by_bloc={
            "W": {
                "W": PreferenceInterval({"W1": 0.4, "W2": 0.3}),
                "C": PreferenceInterval({"C1": 0.2, "C2": 0.1}),
            },
            "C": {
                "W": PreferenceInterval({"W1": 0.2, "W2": 0.2}),
                "C": PreferenceInterval({"C1": 0.3, "C2": 0.3}),
            },
        },
        bloc_voter_prop={"W": 0.7, "C": 0.3},
        cohesion_parameters={"W": {"W": 0.7, "C": 0.3}, "C": {"C": 0.9, "W": 0.1}},
    )
    profile = sp.generate_profile(number_of_ballots=100)
    assert type(profile) is PreferenceProfile

    result = sp.generate_profile(number_of_ballots=100, by_bloc=True)
    assert type(result) is tuple
    profile_dict, agg_prof = result
    assert isinstance(profile_dict, dict)
    assert (type(profile_dict["W"])) is PreferenceProfile
    assert type(agg_prof) is PreferenceProfile
    assert agg_prof.num_ballots() == 100


def test_SPL_completion_zero_cand():
    """
    Ensure that SPL can handle candidates with 0 support.
    """
    sp = slate_PlackettLuce(
        candidates=["W1", "W2", "C1", "C2"],
        slate_to_candidates={"W": ["W1", "W2"], "C": ["C1", "C2"]},
        pref_intervals_by_bloc={
            "W": {
                "W": PreferenceInterval({"W1": 0.4, "W2": 0.3}),
                "C": PreferenceInterval({"C1": 0.2, "C2": 0}),
            },
            "C": {
                "W": PreferenceInterval({"W1": 0.2, "W2": 0.2}),
                "C": PreferenceInterval({"C1": 0.3, "C2": 0.3}),
            },
        },
        bloc_voter_prop={"W": 0.7, "C": 0.3},
        cohesion_parameters={"W": {"W": 0.7, "C": 0.3}, "C": {"C": 0.9, "W": 0.1}},
    )
    profile = sp.generate_profile(number_of_ballots=100)
    assert type(profile) is PreferenceProfile

    result = sp.generate_profile(number_of_ballots=100, by_bloc=True)
    assert type(result) is tuple
    profile_dict, agg_prof = result
    assert isinstance(profile_dict, dict)
    assert (type(profile_dict["W"])) is PreferenceProfile
    assert type(agg_prof) is PreferenceProfile
    assert agg_prof.num_ballots() == 100


def test_SBT_completion_zero_cand():
    """
    Ensure that SBT can handle candidates with 0 support.
    """
    sp = slate_BradleyTerry(
        candidates=["W1", "W2", "C1", "C2"],
        slate_to_candidates={"W": ["W1", "W2"], "C": ["C1", "C2"]},
        pref_intervals_by_bloc={
            "W": {
                "W": PreferenceInterval({"W1": 0.4, "W2": 0.3}),
                "C": PreferenceInterval({"C1": 0.2, "C2": 0}),
            },
            "C": {
                "W": PreferenceInterval({"W1": 0.2, "W2": 0.2}),
                "C": PreferenceInterval({"C1": 0.3, "C2": 0.3}),
            },
        },
        bloc_voter_prop={"W": 0.7, "C": 0.3},
        cohesion_parameters={"W": {"W": 0.7, "C": 0.3}, "C": {"C": 0.9, "W": 0.1}},
    )
    profile = sp.generate_profile(number_of_ballots=100)
    assert type(profile) is PreferenceProfile

    result = sp.generate_profile(number_of_ballots=100, by_bloc=True)
    assert type(result) is tuple
    profile_dict, agg_prof = result
    assert isinstance(profile_dict, dict)
    assert (type(profile_dict["W"])) is PreferenceProfile
    assert type(agg_prof) is PreferenceProfile
    assert agg_prof.num_ballots() == 100


def test_SBT_completion():
    sbt = slate_BradleyTerry(
        candidates=["W1", "W2", "C1", "C2"],
        slate_to_candidates={"W": ["W1", "W2"], "C": ["C1", "C2"]},
        pref_intervals_by_bloc={
            "W": {
                "W": PreferenceInterval({"W1": 0.4, "W2": 0.3}),
                "C": PreferenceInterval({"C1": 0.2, "C2": 0.1}),
            },
            "C": {
                "W": PreferenceInterval({"W1": 0.2, "W2": 0.2}),
                "C": PreferenceInterval({"C1": 0.3, "C2": 0.3}),
            },
        },
        bloc_voter_prop={"W": 0.7, "C": 0.3},
        cohesion_parameters={"W": {"W": 0.7, "C": 0.3}, "C": {"C": 0.9, "W": 0.1}},
    )
    profile = sbt.generate_profile(number_of_ballots=100)
    assert type(profile) is PreferenceProfile

    result = sbt.generate_profile(number_of_ballots=100, by_bloc=True)
    assert type(result) is tuple
    profile_dict, agg_prof = result
    assert isinstance(profile_dict, dict)
    assert (type(profile_dict["W"])) is PreferenceProfile
    assert type(agg_prof) is PreferenceProfile
    assert agg_prof.num_ballots() == 100


def test_AC_completion():
    ac = AlternatingCrossover(
        candidates=["W1", "W2", "C1", "C2"],
        slate_to_candidates={"W": ["W1", "W2"], "C": ["C1", "C2"]},
        pref_intervals_by_bloc={
            "W": {
                "W": PreferenceInterval({"W1": 0.4, "W2": 0.3}),
                "C": PreferenceInterval({"C1": 0.2, "C2": 0.1}),
            },
            "C": {
                "W": PreferenceInterval({"W1": 0.2, "W2": 0.2}),
                "C": PreferenceInterval({"C1": 0.3, "C2": 0.3}),
            },
        },
        bloc_voter_prop={"W": 0.7, "C": 0.3},
        cohesion_parameters={"W": {"W": 0.7, "C": 0.3}, "C": {"C": 0.9, "W": 0.1}},
    )
    profile = ac.generate_profile(number_of_ballots=100)
    assert type(profile) is PreferenceProfile
    assert profile.num_ballots() == 100


def test_1D_completion():
    ods = OneDimSpatial(candidates=["W1", "W2", "C1", "C2"])
    profile = ods.generate_profile(number_of_ballots=100)
    assert type(profile) is PreferenceProfile
    assert profile.num_ballots() == 100


def test_Cambridge_completion():
    cs = CambridgeSampler(
        candidates=["W1", "W2", "C1", "C2"],
        slate_to_candidates={"A": ["W1", "W2"], "B": ["C1", "C2"]},
        pref_intervals_by_bloc={
            "A": {
                "A": PreferenceInterval({"W1": 0.4, "W2": 0.3}),
                "B": PreferenceInterval({"C1": 0.2, "C2": 0.1}),
            },
            "B": {
                "A": PreferenceInterval({"W1": 0.2, "W2": 0.2}),
                "B": PreferenceInterval({"C1": 0.3, "C2": 0.3}),
            },
        },
        bloc_voter_prop={"A": 0.7, "B": 0.3},
        cohesion_parameters={"A": {"A": 0.7, "B": 0.3}, "B": {"B": 0.9, "A": 0.1}},
    )
    profile = cs.generate_profile(number_of_ballots=100)
    assert type(profile) is PreferenceProfile

    result = cs.generate_profile(number_of_ballots=100, by_bloc=True)
    assert type(result) is tuple
    profile_dict, agg_prof = result
    assert isinstance(profile_dict, dict)
    assert (type(profile_dict["A"])) is PreferenceProfile
    assert type(agg_prof) is PreferenceProfile
    assert agg_prof.num_ballots() == 100


def test_ballot_simplex_from_point():
    candidates = ["W1", "W2", "C1", "C2"]
    pt = {"W1": 1 / 4, "W2": 1 / 4, "C1": 1 / 4, "C2": 1 / 4}

    generated_profile = BallotSimplex.from_point(
        point=pt, candidates=candidates
    ).generate_profile(number_of_ballots=10)
    # Test
    assert isinstance(generated_profile, PreferenceProfile)
    assert generated_profile.num_ballots() == 10


def test_ballot_simplex_from_alpha():
    number_of_ballots = 100
    candidates = ["W1", "W2", "C1", "C2"]

    generated_profile = BallotSimplex.from_alpha(
        alpha=0, candidates=candidates
    ).generate_profile(number_of_ballots=number_of_ballots)
    assert generated_profile.num_ballots() == 100
