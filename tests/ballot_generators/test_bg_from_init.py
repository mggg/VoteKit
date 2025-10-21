import numpy as np

from votekit.ballot_generator import (
    iac_profile_generator,
    ic_profile_generator,
    onedim_spacial_profile_generator,
    slate_bt_profile_generator,
    slate_bt_profiles_by_bloc_generator,
    BlocSlateConfig,
    name_bt_profile_generator,
    name_bt_profiles_by_bloc_generator,
    name_pl_profile_generator,
    name_pl_profiles_by_bloc_generator,
)
from votekit.pref_profile import RankProfile
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


def test_1D_completion():
    profile = onedim_spacial_profile_generator(
        candidates=["W1", "W2", "C1", "C2"], number_of_ballots=100
    )
    assert type(profile) is RankProfile
    assert profile.total_ballot_wt == 100
