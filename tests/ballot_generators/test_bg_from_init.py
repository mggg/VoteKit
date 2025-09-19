import numpy as np
import pytest

from votekit.ballot_generator import (
    generate_iac_profile,
    generate_ic_profile,
    AlternatingCrossover,
    CambridgeSampler,
    generate_1d_spacial_profile,
    generate_slate_pl_profile,
    generate_slate_pl_profiles_by_bloc,
    slate_BradleyTerry,
    name_Cumulative,
    BlocSlateConfig,
    generate_name_bt_profile,
    generate_name_bt_profiles_by_bloc,
    generate_name_pl_profile,
    generate_name_pl_profiles_by_bloc,
)
from votekit.pref_profile import RankProfile, ScoreProfile
from votekit.pref_interval import PreferenceInterval

# set seed for more consistent tests
np.random.seed(8675309)


def test_IC_completion():
    profile = generate_ic_profile(
        candidates=["W1", "W2", "C1", "C2"], number_of_ballots=100
    )
    assert type(profile) is RankProfile
    assert profile.total_ballot_wt == 100


def test_IAC_completion():
    profile = generate_iac_profile(
        candidates=["W1", "W2", "C1", "C2"], number_of_ballots=100
    )
    assert type(profile) is RankProfile
    assert profile.total_ballot_wt == 100


def test_NPL_completion():
    config = BlocSlateConfig(
        n_voters=100,
        slate_to_candidates={"W": ["W1", "W2"], "C": ["C1", "C2"]},
        bloc_proportions={"W": 0.7, "C": 0.3},
        preference_mapping={
            "W": {
                "W": PreferenceInterval({"W1": 0.4, "W2": 0.3}),
                "C": PreferenceInterval({"C1": 0.2, "C2": 0.1}),
            },
            "C": {
                "W": PreferenceInterval({"W1": 0.2, "W2": 0.2}),
                "C": PreferenceInterval({"C1": 0.3, "C2": 0.3}),
            },
        },
        cohesion_mapping={"W": {"W": 0.7, "C": 0.3}, "C": {"C": 0.9, "W": 0.1}},
    )
    profile = generate_name_pl_profile(config)
    assert type(profile) is RankProfile
    assert profile.total_ballot_wt == 100

    profile_dict = generate_name_pl_profiles_by_bloc(config)
    assert isinstance(profile_dict, dict)
    assert (type(profile_dict["W"])) is RankProfile  # type: ignore


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
    assert type(profile) is ScoreProfile

    result = cumu.generate_profile(number_of_ballots=100, by_bloc=True)
    assert type(result) is tuple
    profile_dict, agg_prof = result
    assert isinstance(profile_dict, dict)
    assert (type(profile_dict["W"])) is ScoreProfile
    assert type(agg_prof) is ScoreProfile
    assert agg_prof.total_ballot_wt == 100


def test_NBT_completion():
    config = BlocSlateConfig(
        n_voters=100,
        slate_to_candidates={"W": ["W1", "W2"], "C": ["C1", "C2"]},
        bloc_proportions={"W": 0.7, "C": 0.3},
        preference_mapping={
            "W": {
                "W": PreferenceInterval({"W1": 0.4, "W2": 0.3}),
                "C": PreferenceInterval({"C1": 0.2, "C2": 0.1}),
            },
            "C": {
                "W": PreferenceInterval({"W1": 0.2, "W2": 0.2}),
                "C": PreferenceInterval({"C1": 0.3, "C2": 0.3}),
            },
        },
        cohesion_mapping={"W": {"W": 0.7, "C": 0.3}, "C": {"C": 0.9, "W": 0.1}},
    )

    profile = generate_name_bt_profile(config)
    assert type(profile) is RankProfile

    profile_dict = generate_name_bt_profiles_by_bloc(config)
    assert isinstance(profile_dict, dict)
    assert (type(profile_dict["W"])) is RankProfile


def test_SPL_completion():
    config = BlocSlateConfig(
        n_voters=100,
        slate_to_candidates={"W": ["W1", "W2"], "C": ["C1", "C2"]},
        bloc_proportions={"W": 0.7, "C": 0.3},
        preference_mapping={
            "W": {
                "W": PreferenceInterval({"W1": 0.4, "W2": 0.3}),
                "C": PreferenceInterval({"C1": 0.2, "C2": 0.1}),
            },
            "C": {
                "W": PreferenceInterval({"W1": 0.2, "W2": 0.2}),
                "C": PreferenceInterval({"C1": 0.3, "C2": 0.3}),
            },
        },
        cohesion_mapping={"W": {"W": 0.7, "C": 0.3}, "C": {"C": 0.9, "W": 0.1}},
    )
    profile = generate_slate_pl_profile(config)
    assert type(profile) is RankProfile
    assert profile.total_ballot_wt == 100

    profile_dict = generate_slate_pl_profiles_by_bloc(config)

    assert isinstance(profile_dict, dict)
    assert (type(profile_dict["W"])) is RankProfile


def test_SPL_completion_zero_cand():
    """
    Ensure that SPL can handle candidates with 0 support.
    """
    config = BlocSlateConfig(
        n_voters=100,
        slate_to_candidates={"W": ["W1", "W2"], "C": ["C1", "C2"]},
        bloc_proportions={"W": 0.7, "C": 0.3},
        preference_mapping={
            "W": {
                "W": PreferenceInterval({"W1": 0.4, "W2": 0.3}),
                "C": PreferenceInterval({"C1": 0.2, "C2": 0}),
            },
            "C": {
                "W": PreferenceInterval({"W1": 0.2, "W2": 0.2}),
                "C": PreferenceInterval({"C1": 0.3, "C2": 0.3}),
            },
        },
        cohesion_mapping={"W": {"W": 0.7, "C": 0.3}, "C": {"C": 0.9, "W": 0.1}},
    )
    profile = generate_slate_pl_profile(config)
    assert type(profile) is RankProfile
    assert profile.total_ballot_wt == 100

    profile_dict = generate_slate_pl_profiles_by_bloc(config)
    assert isinstance(profile_dict, dict)
    assert (type(profile_dict["W"])) is RankProfile


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
    assert type(profile) is RankProfile

    result = sp.generate_profile(number_of_ballots=100, by_bloc=True)
    assert type(result) is tuple
    profile_dict, agg_prof = result
    assert isinstance(profile_dict, dict)
    assert (type(profile_dict["W"])) is RankProfile
    assert type(agg_prof) is RankProfile
    assert agg_prof.total_ballot_wt == 100


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
    assert type(profile) is RankProfile

    result = sbt.generate_profile(number_of_ballots=100, by_bloc=True)
    assert type(result) is tuple
    profile_dict, agg_prof = result
    assert isinstance(profile_dict, dict)
    assert (type(profile_dict["W"])) is RankProfile
    assert type(agg_prof) is RankProfile
    assert agg_prof.total_ballot_wt == 100


def test_AC_completion():
    with pytest.raises(NotImplementedError):
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
        ac.generate_profile(number_of_ballots=100)


def test_1D_completion():
    profile = generate_1d_spacial_profile(
        candidates=["W1", "W2", "C1", "C2"], number_of_ballots=100
    )
    assert type(profile) is RankProfile
    assert profile.total_ballot_wt == 100


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
    assert type(profile) is RankProfile

    result = cs.generate_profile(number_of_ballots=100, by_bloc=True)
    assert type(result) is tuple
    profile_dict, agg_prof = result
    assert isinstance(profile_dict, dict)
    assert (type(profile_dict["A"])) is RankProfile
    assert type(agg_prof) is RankProfile
    assert agg_prof.total_ballot_wt == 100


def test_Cambridge_completion_W_C_bloc():
    # W as majority
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
        W_bloc="A",
        C_bloc="B",
    )
    profile = cs.generate_profile(number_of_ballots=100)
    assert type(profile) is RankProfile

    result = cs.generate_profile(number_of_ballots=100, by_bloc=True)
    assert type(result) is tuple
    profile_dict, agg_prof = result
    assert isinstance(profile_dict, dict)
    assert (type(profile_dict["A"])) is RankProfile
    assert type(agg_prof) is RankProfile
    assert agg_prof.total_ballot_wt == 100

    # W as minority
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
        W_bloc="B",
        C_bloc="A",
    )
    profile = cs.generate_profile(number_of_ballots=100)
    assert type(profile) is RankProfile

    result = cs.generate_profile(number_of_ballots=100, by_bloc=True)
    assert type(result) is tuple
    profile_dict, agg_prof = result
    assert isinstance(profile_dict, dict)
    assert (type(profile_dict["A"])) is RankProfile
    assert type(agg_prof) is RankProfile
    assert agg_prof.total_ballot_wt == 100
