import pytest

import votekit.ballot_generator as bg
from votekit import PreferenceInterval
from votekit.pref_profile import ScoreProfile


def test_name_cumulative_total_points_None_is_n_cands():
    n_voters = 100
    bloc_proportions = {"all_voters": 1}
    slate_to_candidates = {"all_voters": ["A", "B", "C"]}

    preference_mapping = {
        "all_voters": {
            "all_voters": PreferenceInterval({"A": 0.80, "B": 0.15, "C": 0.05})
        }
    }

    cohesion_mapping = {"all_voters": {"all_voters": 1}}

    config = bg.BlocSlateConfig(
        n_voters=n_voters,
        bloc_proportions=bloc_proportions,
        slate_to_candidates=slate_to_candidates,
        preference_mapping=preference_mapping,
        cohesion_mapping=cohesion_mapping,
    )

    profile = bg.name_cumulative_profile_generator(config)

    df = profile.df

    assert all(df[["A", "B", "C"]].sum(axis=1) == 3)
    assert all(df[["A", "B", "C"]].max(axis=1) <= 3)
    assert (df[["A", "B", "C"]].sum(axis=1) * df["Weight"]).sum() == len(
        config.candidates
    ) * n_voters


def test_name_cumulative_total_points_less_than_n_candidates():
    total_points = 2
    n_voters = 100
    bloc_proportions = {"all_voters": 1}
    slate_to_candidates = {"all_voters": ["A", "B", "C"]}

    preference_mapping = {
        "all_voters": {
            "all_voters": PreferenceInterval({"A": 0.80, "B": 0.15, "C": 0.05})
        }
    }

    cohesion_mapping = {"all_voters": {"all_voters": 1}}

    config = bg.BlocSlateConfig(
        n_voters=n_voters,
        bloc_proportions=bloc_proportions,
        slate_to_candidates=slate_to_candidates,
        preference_mapping=preference_mapping,
        cohesion_mapping=cohesion_mapping,
    )

    profile = bg.name_cumulative_profile_generator(config, total_points=total_points)

    df = profile.df

    assert all(df[["A", "B", "C"]].sum(axis=1) == total_points)
    assert all(df[["A", "B", "C"]].max(axis=1) <= total_points)
    assert (
        df[["A", "B", "C"]].sum(axis=1) * df["Weight"]
    ).sum() == total_points * n_voters


def test_name_cumulative_total_points_more_than_n_candidates():
    total_points = 12
    n_voters = 10
    bloc_proportions = {"all_voters": 1}
    slate_to_candidates = {"all_voters": ["A", "B", "C"]}

    preference_mapping = {
        "all_voters": {
            "all_voters": PreferenceInterval({"A": 0.80, "B": 0.15, "C": 0.05})
        }
    }

    cohesion_mapping = {"all_voters": {"all_voters": 1}}

    config = bg.BlocSlateConfig(
        n_voters=n_voters,
        bloc_proportions=bloc_proportions,
        slate_to_candidates=slate_to_candidates,
        preference_mapping=preference_mapping,
        cohesion_mapping=cohesion_mapping,
    )

    profile = bg.name_cumulative_profile_generator(config, total_points=total_points)

    df = profile.df

    assert all(df[["A", "B", "C"]].sum(axis=1) == total_points)
    assert all(df[["A", "B", "C"]].max(axis=1) <= total_points)
    assert (
        df[["A", "B", "C"]].sum(axis=1) * df["Weight"]
    ).sum() == total_points * n_voters


def test_name_cumulative_total_points_zero_errors():
    total_points = 0
    n_voters = 10
    bloc_proportions = {"all_voters": 1}
    slate_to_candidates = {"all_voters": ["A", "B", "C"]}

    preference_mapping = {
        "all_voters": {
            "all_voters": PreferenceInterval({"A": 0.80, "B": 0.15, "C": 0.05})
        }
    }

    cohesion_mapping = {"all_voters": {"all_voters": 1}}

    config = bg.BlocSlateConfig(
        n_voters=n_voters,
        bloc_proportions=bloc_proportions,
        slate_to_candidates=slate_to_candidates,
        preference_mapping=preference_mapping,
        cohesion_mapping=cohesion_mapping,
    )

    with pytest.raises(ValueError, match="must be a positive integer"):
        bg.name_cumulative_profile_generator(config, total_points=total_points)


def test_name_cumulative_total_points_negative_errors():
    total_points = -1
    n_voters = 10
    bloc_proportions = {"all_voters": 1}
    slate_to_candidates = {"all_voters": ["A", "B", "C"]}

    preference_mapping = {
        "all_voters": {
            "all_voters": PreferenceInterval({"A": 0.80, "B": 0.15, "C": 0.05})
        }
    }

    cohesion_mapping = {"all_voters": {"all_voters": 1}}

    config = bg.BlocSlateConfig(
        n_voters=n_voters,
        bloc_proportions=bloc_proportions,
        slate_to_candidates=slate_to_candidates,
        preference_mapping=preference_mapping,
        cohesion_mapping=cohesion_mapping,
    )

    with pytest.raises(ValueError, match="must be a positive integer"):
        bg.name_cumulative_profile_generator(config, total_points=total_points)


def test_name_cumulative_completion():
    config = bg.BlocSlateConfig(
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
    profile = bg.name_cumulative_profile_generator(config)
    assert type(profile) is ScoreProfile
    assert profile.total_ballot_wt == 100

    profile_dict = bg.name_cumulative_ballot_generator_by_bloc(config)
    assert isinstance(profile_dict, dict)
    assert (type(profile_dict["W"])) is ScoreProfile


def test_name_cumulative_distribution(do_ballot_probs_match_ballot_dist_score_profile):
    config = bg.BlocSlateConfig(
        n_voters=100,
        slate_to_candidates={"W": ["A", "B"]},
        bloc_proportions={"W": 1},
        preference_mapping={"W": {"W": PreferenceInterval({"A": 0.4, "B": 0.6})}},
        cohesion_mapping={"W": {"W": 1}},
    )

    pp = bg.name_cumulative_profile_generator(config)

    ballot_prob_dict = {
        "AA": config.get_preference_interval_for_bloc_and_slate("W", "W").interval["A"]
        ** 2,
        "AB": config.get_preference_interval_for_bloc_and_slate("W", "W").interval["A"]
        * config.get_preference_interval_for_bloc_and_slate("W", "W").interval["B"],
        "BA": config.get_preference_interval_for_bloc_and_slate("W", "W").interval["A"]
        * config.get_preference_interval_for_bloc_and_slate("W", "W").interval["B"],
        "BB": config.get_preference_interval_for_bloc_and_slate("W", "W").interval["B"]
        ** 2,
    }

    assert isinstance(pp, ScoreProfile)
    # Test
    assert do_ballot_probs_match_ballot_dist_score_profile(ballot_prob_dict, pp)
