import numpy as np
import pytest

import votekit.ballot_generator as bg
from votekit import PreferenceInterval
from votekit.ballot import ScoreBallot
from votekit.ballot_generator.bloc_slate_generator import cumulative
from votekit.ballot_generator.bloc_slate_generator._profile_construction import _apportion_bloc_rows
from votekit.pref_profile import ScoreProfile


@pytest.mark.parametrize("n_voters", [2, 37])
def test_cumulative_constructs_matrices_with_the_same_multinomial_samples(
    monkeypatch, two_bloc_two_slate_mixed_config_cambridge, n_voters
):
    config = two_bloc_two_slate_mixed_config_cambridge
    config.n_voters = n_voters
    candidates = tuple(config.candidates)
    preferences = config.get_combined_preference_intervals_by_bloc()
    bloc_rows = _apportion_bloc_rows(config)
    rng = np.random.default_rng(42)
    expected = {}
    for bloc, rows in bloc_rows.items():
        probabilities = [preferences[bloc].interval[candidate] for candidate in candidates]
        counts = rng.multinomial(5, probabilities, size=rows.stop - rows.start)
        expected[bloc] = ScoreProfile(
            ballots=tuple(
                ScoreBallot(scores=dict(zip(candidates, row, strict=True))) for row in counts
            ),
            candidates=candidates,
        )

    def reject_ballot_or_dataframe_construction(*args, **kwargs):
        raise AssertionError("Cumulative generation must construct profiles from score matrices.")

    monkeypatch.setattr(cumulative, "BATCH_SIZE", 7)
    with monkeypatch.context() as patches:
        patches.setattr(ScoreBallot, "__init__", reject_ballot_or_dataframe_construction)
        for name in ("__init__", "__add__", "to_df", "group_ballots"):
            patches.setattr(ScoreProfile, name, reject_ballot_or_dataframe_construction)

        combined = cumulative.name_cumulative_profile_generator(
            config, total_points=5, group_ballots=False, rng_seed=42
        )
        by_bloc = cumulative.name_cumulative_ballot_generator_by_bloc(
            config, total_points=5, group_ballots=False, rng_seed=42
        )
        grouped = cumulative.name_cumulative_profile_generator(config, total_points=5, rng_seed=42)
        grouped_by_bloc = cumulative.name_cumulative_ballot_generator_by_bloc(
            config, total_points=5, rng_seed=42
        )

    assert combined.total_ballot_wt == n_voters
    assert combined.num_ballots == n_voters
    assert combined.candidates == candidates
    assert grouped.ballots == combined.group_ballots().ballots
    np.testing.assert_array_equal(np.nansum(combined._matrix.scores, axis=1), 5)
    np.testing.assert_array_equal(
        combined._matrix.scores,
        np.concatenate([profile._matrix.scores for profile in expected.values()]),
    )
    for bloc, profile in by_bloc.items():
        assert profile.ballots == expected[bloc].ballots
        assert grouped_by_bloc[bloc].ballots == expected[bloc].group_ballots().ballots
        assert profile.candidates == candidates
        assert profile._matrix.score_columns == candidates
        assert not profile._matrix.scores.flags.writeable
        assert not np.shares_memory(profile._matrix.scores, combined._matrix.scores)


def test_cumulative_retains_unscored_candidates_and_normalizes_names():
    config = bg.BlocSlateConfig(
        n_voters=13,
        allow_zero_support_candidates=True,
        slate_to_candidates={"slate": [" A ", 7, " B "]},
        bloc_proportions={"bloc": 1.0},
        preference_mapping={
            "bloc": {
                "slate": PreferenceInterval(
                    {" A ": 1.0, 7: 0.0, " B ": 0.0}, allow_zero_support=True
                )
            }
        },
        cohesion_mapping={"bloc": {"slate": 1.0}},
    )
    profile = cumulative.name_cumulative_profile_generator(config, total_points=2, rng_seed=42)

    assert profile.candidates == ("A", 7, "B")
    assert profile.candidates_cast == ("A",)
    assert profile.num_ballots == 1
    assert profile.total_ballot_wt == 13
    np.testing.assert_array_equal(profile._matrix.scores, [[2.0, np.nan, np.nan]])
    assert profile.ballots == (ScoreBallot(scores={"A": 2.0}, weight=13),)
    assert tuple(config.candidates) == (" A ", 7, " B ")


def test_name_cumulative_total_points_None_is_n_cands():
    n_voters = 100
    bloc_proportions = {"all_voters": 1}
    slate_to_candidates = {"all_voters": ["A", "B", "C"]}

    preference_mapping = {
        "all_voters": {"all_voters": PreferenceInterval({"A": 0.80, "B": 0.15, "C": 0.05})}
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

    df = profile.to_df()

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
        "all_voters": {"all_voters": PreferenceInterval({"A": 0.80, "B": 0.15, "C": 0.05})}
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

    df = profile.to_df()

    assert all(df[["A", "B", "C"]].sum(axis=1) == total_points)
    assert all(df[["A", "B", "C"]].max(axis=1) <= total_points)
    assert (df[["A", "B", "C"]].sum(axis=1) * df["Weight"]).sum() == total_points * n_voters


def test_name_cumulative_total_points_more_than_n_candidates():
    total_points = 12
    n_voters = 10
    bloc_proportions = {"all_voters": 1}
    slate_to_candidates = {"all_voters": ["A", "B", "C"]}

    preference_mapping = {
        "all_voters": {"all_voters": PreferenceInterval({"A": 0.80, "B": 0.15, "C": 0.05})}
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

    df = profile.to_df()

    assert all(df[["A", "B", "C"]].sum(axis=1) == total_points)
    assert all(df[["A", "B", "C"]].max(axis=1) <= total_points)
    assert (df[["A", "B", "C"]].sum(axis=1) * df["Weight"]).sum() == total_points * n_voters


def test_name_cumulative_total_points_zero_errors():
    total_points = 0
    n_voters = 10
    bloc_proportions = {"all_voters": 1}
    slate_to_candidates = {"all_voters": ["A", "B", "C"]}

    preference_mapping = {
        "all_voters": {"all_voters": PreferenceInterval({"A": 0.80, "B": 0.15, "C": 0.05})}
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
        "all_voters": {"all_voters": PreferenceInterval({"A": 0.80, "B": 0.15, "C": 0.05})}
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


def test_name_cumulative_completion_with_mixed_candidates():
    config = bg.BlocSlateConfig(
        n_voters=100,
        slate_to_candidates={"W": ["W1", "W2"], "C": [1, 2]},
        bloc_proportions={"W": 0.7, "C": 0.3},
        preference_mapping={
            "W": {
                "W": PreferenceInterval({"W1": 0.4, "W2": 0.3}),
                "C": PreferenceInterval({1: 0.2, 2: 0.1}),
            },
            "C": {
                "W": PreferenceInterval({"W1": 0.2, "W2": 0.2}),
                "C": PreferenceInterval({1: 0.3, 2: 0.3}),
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
        "AA": config.get_preference_interval_for_bloc_and_slate("W", "W").interval["A"] ** 2,
        "AB": config.get_preference_interval_for_bloc_and_slate("W", "W").interval["A"]
        * config.get_preference_interval_for_bloc_and_slate("W", "W").interval["B"],
        "BA": config.get_preference_interval_for_bloc_and_slate("W", "W").interval["A"]
        * config.get_preference_interval_for_bloc_and_slate("W", "W").interval["B"],
        "BB": config.get_preference_interval_for_bloc_and_slate("W", "W").interval["B"] ** 2,
    }

    assert isinstance(pp, ScoreProfile)
    # Test
    assert do_ballot_probs_match_ballot_dist_score_profile(ballot_prob_dict, pp)
