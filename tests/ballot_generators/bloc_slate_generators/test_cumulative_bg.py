import pytest

import votekit.ballot_generator as bg
from votekit import PreferenceInterval


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
