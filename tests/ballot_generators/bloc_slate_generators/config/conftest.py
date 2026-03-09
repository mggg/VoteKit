"""Shared fixtures for bloc-slate config tests."""

import pandas as pd
import pytest

from votekit import PreferenceInterval


@pytest.fixture
def valid_config():
    slate_to_candidates = {"slate_1": ["A", "B"], "slate_2": ["X", "Y"]}
    preference_mapping = {
        "bloc_1": {
            "slate_1": PreferenceInterval({"A": 0.4, "B": 0.1}),
            "slate_2": PreferenceInterval({"X": 0.1, "Y": 0.9}),
        },
        "bloc_2": {
            "slate_1": PreferenceInterval({"A": 0.05, "B": 0.05}),
            "slate_2": PreferenceInterval({"X": 0.45, "Y": 0.45}),
        },
    }
    bloc_proportions = {"bloc_1": 0.8, "bloc_2": 0.2}
    cohesion_mapping = {
        "bloc_1": {"slate_1": 0.9, "slate_2": 0.1},
        "bloc_2": {"slate_2": 0.8, "slate_1": 0.2},
    }
    return dict(
        slate_to_candidates=slate_to_candidates,
        preference_mapping=preference_mapping,
        bloc_proportions=bloc_proportions,
        cohesion_mapping=cohesion_mapping,
    )


@pytest.fixture
def alt_valid_config():
    slate_to_candidates = {"slate_1": ["A", "B"], "slate_2": ["X", "Y"]}
    preference_mapping = pd.DataFrame(
        {
            "bloc_1": {"A": 0.8, "B": 0.2, "X": 0.1, "Y": 0.9},
            "bloc_2": {"A": 0.5, "B": 0.5, "X": 0.5, "Y": 0.5},
        }
    ).T
    bloc_proportions = {"bloc_1": 0.8, "bloc_2": 0.2}
    cohesion_mapping = pd.DataFrame(
        {
            "bloc_1": {"slate_1": 0.9, "slate_2": 0.1},
            "bloc_2": {"slate_2": 0.8, "slate_1": 0.1},
        }
    ).T
    return dict(
        slate_to_candidates=slate_to_candidates,
        preference_mapping=preference_mapping,
        bloc_proportions=bloc_proportions,
        cohesion_mapping=cohesion_mapping,
    )


@pytest.fixture
def extra_profile_settings():
    # Canonical structure used across all profiles
    slates = {"slate_1": ["A", "B"], "slate_2": ["X", "Y"]}
    blocs = {"bloc_1": 0.8, "bloc_2": 0.2}

    # --- Cohesion (handy for the slate-update tests)
    cohesion_df = pd.DataFrame(
        {
            "slate_1": {"bloc_1": 0.9, "bloc_2": 0.2},
            "slate_2": {"bloc_1": 0.1, "bloc_2": 0.8},
        }
    ).astype(float)

    # --- Preference: BASE (matches candidates exactly)
    pref_df_base = pd.DataFrame(
        {
            "A": {"bloc_1": 0.8, "bloc_2": 0.5},
            "B": {"bloc_1": 0.2, "bloc_2": 0.5},
            "X": {"bloc_1": 0.1, "bloc_2": 0.5},
            "Y": {"bloc_1": 0.9, "bloc_2": 0.5},
        }
    )

    # Empty → __update_preference_df_on_candidate_change should create (-1.0) matrix
    pref_df_empty = pd.DataFrame()

    # Extra stray column 'Z' → should be dropped by the updater
    pref_df_with_extra_col = pref_df_base.assign(Z={"bloc_1": 0.0, "bloc_2": 0.0})  # type: ignore[call-arg]

    # Same columns, different order → should be reordered to match config.candidates
    pref_df_out_of_order = pref_df_base[["B", "A", "Y", "X"]].copy()

    # Same as base but using PreferenceInterval objects
    pref_map_intervals = {
        "bloc_1": {
            "slate_1": PreferenceInterval({"A": 0.8, "B": 0.2}),
            "slate_2": PreferenceInterval({"X": 0.1, "Y": 0.9}),
        },
        "bloc_2": {
            "slate_1": PreferenceInterval({"A": 0.5, "B": 0.5}),
            "slate_2": PreferenceInterval({"X": 0.5, "Y": 0.5}),
        },
    }

    return dict(
        slates=slates,
        blocs=blocs,
        cohesion_df=cohesion_df,
        pref_df_base=pref_df_base,
        pref_df_empty=pref_df_empty,
        pref_df_with_extra_col=pref_df_with_extra_col,
        pref_df_out_of_order=pref_df_out_of_order,
        pref_map_intervals=pref_map_intervals,
    )
