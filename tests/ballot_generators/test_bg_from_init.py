import numpy as np
import pytest

from votekit.ballot_generator import (
    iac_profile_generator,
    ic_profile_generator,
    AlternatingCrossover,
    cambridge_profile_generator,
    cambridge_profiles_by_bloc_generator,
    onedim_spacial_profile_generator,
    slate_pl_profile_generator,
    slate_pl_profiles_by_bloc_generator,
    slate_bt_profile_generator,
    slate_bt_profiles_by_bloc_generator,
    name_cumulative_profile_generator,
    name_cumulative_ballot_generator_by_bloc,
    BlocSlateConfig,
    name_bt_profile_generator,
    name_bt_profiles_by_bloc_generator,
    name_pl_profile_generator,
    name_pl_profiles_by_bloc_generator,
)
from votekit.pref_profile import RankProfile, ScoreProfile
from votekit.pref_interval import PreferenceInterval

# set seed for more consistent tests
np.random.seed(8675309)


def test_IC_completion():
    profile = ic_profile_generator(
        candidates=["W1", "W2", "C1", "C2"], number_of_ballots=100
    )
    assert type(profile) is RankProfile
    assert profile.total_ballot_wt == 100


def test_IAC_completion():
    profile = iac_profile_generator(
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
    profile = name_pl_profile_generator(config)
    assert type(profile) is RankProfile
    assert profile.total_ballot_wt == 100

    profile_dict = name_pl_profiles_by_bloc_generator(config)
    assert isinstance(profile_dict, dict)
    assert (type(profile_dict["W"])) is RankProfile  # type: ignore


def test_name_cumulative_completion():
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
    profile = name_cumulative_profile_generator(config)
    assert type(profile) is ScoreProfile
    assert profile.total_ballot_wt == 100

    profile_dict = name_cumulative_ballot_generator_by_bloc(config)
    assert isinstance(profile_dict, dict)
    assert (type(profile_dict["W"])) is ScoreProfile


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

    profile = name_bt_profile_generator(config)
    assert type(profile) is RankProfile

    profile_dict = name_bt_profiles_by_bloc_generator(config)
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
    profile = slate_pl_profile_generator(config)
    assert type(profile) is RankProfile
    assert profile.total_ballot_wt == 100

    profile_dict = slate_pl_profiles_by_bloc_generator(config)

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
    profile = slate_pl_profile_generator(config)
    assert type(profile) is RankProfile
    assert profile.total_ballot_wt == 100

    profile_dict = slate_pl_profiles_by_bloc_generator(config)
    assert isinstance(profile_dict, dict)
    assert (type(profile_dict["W"])) is RankProfile


def test_SBT_completion_zero_cand():
    """
    Ensure that SBT can handle candidates with 0 support.
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
    profile = slate_bt_profile_generator(config)
    assert type(profile) is RankProfile
    assert profile.total_ballot_wt == 100

    profile_dict = slate_bt_profiles_by_bloc_generator(config)
    assert isinstance(profile_dict, dict)
    assert (type(profile_dict["W"])) is RankProfile


def test_SBT_completion():
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
    profile = slate_bt_profile_generator(config)
    assert type(profile) is RankProfile
    assert profile.total_ballot_wt == 100

    profile_dict = slate_bt_profiles_by_bloc_generator(config)
    assert isinstance(profile_dict, dict)
    assert (type(profile_dict["W"])) is RankProfile


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
    profile = onedim_spacial_profile_generator(
        candidates=["W1", "W2", "C1", "C2"], number_of_ballots=100
    )
    assert type(profile) is RankProfile
    assert profile.total_ballot_wt == 100


def test_Cambridge_completion():
    config = BlocSlateConfig(
        n_voters=100,
        slate_to_candidates={"A": ["W1", "W2"], "B": ["C1", "C2"]},
        bloc_proportions={"A": 0.7, "B": 0.3},
        preference_mapping={
            "A": {
                "A": PreferenceInterval({"W1": 0.4, "W2": 0.3}),
                "B": PreferenceInterval({"C1": 0.2, "C2": 0.1}),
            },
            "B": {
                "A": PreferenceInterval({"W1": 0.2, "W2": 0.2}),
                "B": PreferenceInterval({"C1": 0.3, "C2": 0.3}),
            },
        },
        cohesion_mapping={"A": {"A": 0.7, "B": 0.3}, "B": {"B": 0.9, "A": 0.1}},
    )
    profile = cambridge_profile_generator(config)
    assert type(profile) is RankProfile
    assert profile.total_ballot_wt == 100

    profile_dict = cambridge_profiles_by_bloc_generator(config)
    assert isinstance(profile_dict, dict)
    assert (type(profile_dict["A"])) is RankProfile


def test_Cambridge_completion_W_C_bloc():
    # W as majority
    config = BlocSlateConfig(
        n_voters=100,
        slate_to_candidates={"A": ["W1", "W2"], "B": ["C1", "C2"]},
        bloc_proportions={"A": 0.7, "B": 0.3},
        preference_mapping={
            "A": {
                "A": PreferenceInterval({"W1": 0.4, "W2": 0.3}),
                "B": PreferenceInterval({"C1": 0.2, "C2": 0.1}),
            },
            "B": {
                "A": PreferenceInterval({"W1": 0.2, "W2": 0.2}),
                "B": PreferenceInterval({"C1": 0.3, "C2": 0.3}),
            },
        },
        cohesion_mapping={"A": {"A": 0.7, "B": 0.3}, "B": {"B": 0.9, "A": 0.1}},
    )
    profile = cambridge_profile_generator(
        config,
        majority_bloc="A",
        minority_bloc="B",
    )
    assert type(profile) is RankProfile
    assert profile.total_ballot_wt == 100

    profile_dict = cambridge_profiles_by_bloc_generator(
        config,
        majority_bloc="A",
        minority_bloc="B",
    )
    assert isinstance(profile_dict, dict)
    assert (type(profile_dict["A"])) is RankProfile

    # W as minority
    profile = cambridge_profile_generator(
        config,
        majority_bloc="B",
        minority_bloc="A",
    )
    assert type(profile) is RankProfile
    assert profile.total_ballot_wt == 100

    profile_dict = cambridge_profiles_by_bloc_generator(
        config,
        majority_bloc="B",
        minority_bloc="A",
    )
    assert isinstance(profile_dict, dict)
    assert (type(profile_dict["A"])) is RankProfile
