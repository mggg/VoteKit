from votekit.ballot_generator import (
    slate_bt_profile_generator,
    slate_bt_profiles_by_bloc_generator,
    BlocSlateConfig,
)
from votekit.pref_profile import RankProfile
from votekit.pref_interval import PreferenceInterval


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


def test_slate_BT_distribution(do_ballot_probs_match_ballot_dist_rank_profile):
    config = BlocSlateConfig(
        n_voters=100,
        bloc_proportions={"A": 0.99999, "B": 0.00001},
        slate_to_candidates={"A": ["X", "Y"], "B": ["Z"]},
        cohesion_mapping={"A": {"A": 0.8, "B": 0.2}, "B": {"A": 0.2, "B": 0.8}},
        preference_mapping={
            "A": {
                "A": PreferenceInterval({"X": 0.9, "Y": 0.1}),
                "B": PreferenceInterval({"Z": 1}),
            },
            "B": {
                "A": PreferenceInterval({"X": 0.9, "Y": 0.1}),
                "B": PreferenceInterval({"Z": 1}),
            },
        },
    )

    pp = slate_bt_profile_generator(config)

    ballot_prob_dict = {
        "XYZ": config.cohesion_df["A"].loc["A"] ** 2
        * config.get_preference_interval_for_bloc_and_slate(
            bloc_name="A", slate_name="A"
        ).interval["X"],
        "YXZ": config.cohesion_df["A"].loc["A"] ** 2
        * config.get_preference_interval_for_bloc_and_slate(
            bloc_name="A", slate_name="A"
        ).interval["Y"],
        "XZY": config.cohesion_df["A"].loc["A"]
        * config.cohesion_df["A"].loc["B"]
        * config.get_preference_interval_for_bloc_and_slate(
            bloc_name="A", slate_name="A"
        ).interval["X"],
        "YZX": config.cohesion_df["A"].loc["A"]
        * config.cohesion_df["A"].loc["B"]
        * config.get_preference_interval_for_bloc_and_slate(
            bloc_name="A", slate_name="A"
        ).interval["Y"],
        "ZXY": config.cohesion_df["A"].loc["B"] ** 2
        * config.get_preference_interval_for_bloc_and_slate(
            bloc_name="A", slate_name="A"
        ).interval["X"],
        "ZYX": config.cohesion_df["A"].loc["B"] ** 2
        * config.get_preference_interval_for_bloc_and_slate(
            bloc_name="A", slate_name="A"
        ).interval["X"],
    }

    assert isinstance(pp, RankProfile)
    # Test
    assert do_ballot_probs_match_ballot_dist_rank_profile(ballot_prob_dict, pp)
