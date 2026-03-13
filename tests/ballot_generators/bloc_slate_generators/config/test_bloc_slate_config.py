"""BlocSlateConfig behavior tests."""

import math
import re
import warnings
from typing import cast

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from votekit import PreferenceInterval
from votekit.ballot_generator.bloc_slate_generator.config import (
    BlocSlateConfig,
    ConfigurationWarning,
)

#   BlocSlateConfig
# =====================


def test_simple_init():
    with pytest.warns(ConfigurationWarning):
        config = BlocSlateConfig(n_voters=10)
    assert config.n_voters == 10
    assert config.slate_to_candidates == {}
    assert config.bloc_proportions == {}
    assert config.preference_df.empty
    assert config.cohesion_df.empty
    assert config.read_dirichlet_alphas() is None


def test_non_positve_voter_errors():
    with pytest.raises(ValueError, match="must be > 0"):
        BlocSlateConfig(n_voters=0)
    with pytest.raises(ValueError, match="must be > 0"):
        BlocSlateConfig(n_voters=-1)


def test_non_int_voter_errors():
    with pytest.raises(TypeError, match="Number of voters must be cleanly convertible to an int."):
        BlocSlateConfig(n_voters=3.14)  # type: ignore[arg-type]


def test_allow_zero_support_candidates_type_errors(valid_config):
    with pytest.raises(TypeError, match="allow_zero_support_candidates must be a bool."):
        BlocSlateConfig(
            **valid_config,
            n_voters=100,
            allow_zero_support_candidates=1,  # type: ignore[arg-type]
        )

    config = BlocSlateConfig(**valid_config, n_voters=100)
    with pytest.raises(TypeError, match="allow_zero_support_candidates must be a bool."):
        config.allow_zero_support_candidates = "yes"  # type: ignore[assignment]


def test_reassign_int_voters():
    with pytest.warns(ConfigurationWarning):
        config = BlocSlateConfig(n_voters=10)
    config.n_voters = 25
    assert config.n_voters == 25
    config.n_voters = 5.0  # float that is cleanly convertible to int # type: ignore[assignment]
    assert config.n_voters == 5


def test_reassign_float_voters_errors():
    with pytest.warns(ConfigurationWarning):
        config = BlocSlateConfig(n_voters=10)
    with pytest.raises(TypeError, match="Number of voters must be cleanly convertible to an int."):
        config.n_voters = 5.5  # type: ignore[assignment]


def test_make_new_config_from_existing_values(valid_config):
    with pytest.warns(ConfigurationWarning):
        config1 = BlocSlateConfig(**valid_config, n_voters=100)
        config2 = BlocSlateConfig(
            n_voters=config1.n_voters,
        )

    config2.slate_to_candidates = config1.slate_to_candidates
    config2.bloc_proportions = config1.bloc_proportions
    config2.preference_df = config1.preference_df
    config2.cohesion_df = config1.cohesion_df
    assert config1 == config2


def test_make_new_config_from_copy_with_alphas_works(valid_config):
    config1 = BlocSlateConfig(**valid_config, n_voters=100)
    with pytest.warns(ConfigurationWarning, match="Preference intervals have already been set"):
        config1.set_dirichlet_alphas(
            {
                "bloc_1": {"slate_1": 10, "slate_2": 0.2},
                "bloc_2": {"slate_1": 0.1, "slate_2": 1},
            }
        )

    config2 = config1.copy()

    assert config1 == config2
    alpha1 = config1.read_dirichlet_alphas()
    alpha2 = config2.read_dirichlet_alphas()
    assert alpha1 is not None and alpha2 is not None
    pdt.assert_frame_equal(alpha1, alpha2)

    config1.resample_preference_intervals_from_dirichlet_alphas()
    assert config1 != config2  # now different after resampling


def test_missing_attributes_errors(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100)
    with pytest.raises(
        AttributeError, match=re.escape("'BlocSlateConfig' object has no attribute")
    ):
        _ = config.non_existent_attribute  # type: ignore[attr-defined]


def test_valid_config(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100)
    assert config.n_voters == 100
    assert config.slate_to_candidates == valid_config["slate_to_candidates"]
    assert config.bloc_proportions == valid_config["bloc_proportions"]
    pref_df = pd.DataFrame(
        {"bloc_1": [0.8, 0.2, 0.1, 0.9], "bloc_2": [0.5, 0.5, 0.5, 0.5]},
    ).T
    pref_df.rename(columns={i: v for i, v in enumerate(["A", "B", "X", "Y"])}, inplace=True)
    assert config.preference_df.equals(pref_df)

    cohesion_df = pd.DataFrame(
        {
            "bloc_1": {"slate_1": 0.9, "slate_2": 0.1},
            "bloc_2": {"slate_2": 0.8, "slate_1": 0.2},
        }
    ).T
    assert config.cohesion_df.equals(cohesion_df)


def test_alt_valid_config(alt_valid_config):
    config = BlocSlateConfig(**alt_valid_config, n_voters=100)
    assert config.n_voters == 100
    assert config.slate_to_candidates == alt_valid_config["slate_to_candidates"]
    assert config.bloc_proportions == alt_valid_config["bloc_proportions"]
    pref_df = pd.DataFrame(
        {"bloc_1": [0.8, 0.2, 0.1, 0.9], "bloc_2": [0.5, 0.5, 0.5, 0.5]},
    ).T
    pref_df.rename(columns={i: v for i, v in enumerate(["A", "B", "X", "Y"])}, inplace=True)
    assert config.preference_df.equals(pref_df)

    cohesion_df = pd.DataFrame(
        {
            "bloc_1": {"slate_1": 0.9, "slate_2": 0.1},
            "bloc_2": {"slate_2": 0.8, "slate_1": 0.1},
        }
    ).T
    assert config.cohesion_df.equals(cohesion_df)


def test_valid_config_allows_zero_preferences_and_bloc_named_slates():
    config = BlocSlateConfig(
        n_voters=10_000,
        allow_zero_support_candidates=True,
        bloc_proportions={"Alpha": 0.8, "Xenon": 0.2},
        slate_to_candidates={"Alpha": ["A", "B"], "Xenon": ["X", "Y"]},
        preference_mapping={
            "Alpha": {
                "Alpha": PreferenceInterval({"A": 0.8, "B": 0.2}),
                "Xenon": PreferenceInterval({"X": 0.0, "Y": 1.0}, allow_zero_support=True),
            },
            "Xenon": {
                "Alpha": PreferenceInterval({"A": 0.5, "B": 0.5}),
                "Xenon": PreferenceInterval({"X": 0.5, "Y": 0.5}),
            },
        },
        cohesion_mapping={
            "Alpha": {"Alpha": 0.9, "Xenon": 0.1},
            "Xenon": {"Xenon": 0.9, "Alpha": 0.1},
        },
    )

    assert config.is_valid(raise_errors=True)
    assert config.preference_df.loc["Alpha", "X"] == 0.0


def test_pref_mapping_bad_types_errors(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100)
    with pytest.raises(
        TypeError,
        match=re.escape("must be Mapping[str, float|int] or PreferenceInterval, got 'int'"),
    ):
        config.preference_df = {  # type: ignore[assignment]
            "bloc_1": {"slate_1": 2},
            "bloc_2": {"slate_2": 4},
        }


def test_pref_df_with_dict(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100)
    original_df = config.preference_df.copy()

    preference_mapping = {
        "bloc_1": {
            "slate_1": {"A": 0.8, "B": 0.2},
            "slate_2": {"X": 0.1, "Y": 0.9},
        },
        "bloc_2": {
            "slate_1": {"A": 0.5, "B": 0.5},
            "slate_2": {"X": 0.5, "Y": 0.5},
        },
    }

    config.preference_df = preference_mapping  # type: ignore[assignment]

    assert config.preference_df.equals(original_df)


def test_normalize_pref_df(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100)
    original_df = config.preference_df.copy()

    preference_mapping = {
        "bloc_1": {
            "slate_1": {"A": 4, "B": 1},
            "slate_2": {"X": 1, "Y": 9},
        },
        "bloc_2": {
            "slate_1": {"A": 1, "B": 1},
            "slate_2": {"X": 5, "Y": 5},
        },
    }

    config.preference_df = preference_mapping  # type: ignore[assignment]

    assert not config.preference_df.equals(original_df)

    config.normalize_preference_intervals()
    assert config.preference_df.equals(original_df)


def test_pref_df_with_blocs_missing(valid_config):
    with pytest.warns(ConfigurationWarning) as records:
        config = BlocSlateConfig(n_voters=100)
        config.bloc_proportions = None  # type: ignore[assignment]
        config.preference_df = valid_config["preference_mapping"]
    assert any("no blocs are defined" in str(r.message) for r in records)


def test_pref_mapping_missing_slates_errors(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100)

    preference_mapping = {
        "bloc_1": {
            "slate_1": PreferenceInterval({"A": 0.4, "B": 0.1}),
            "slate_2": PreferenceInterval({"X": 0.1, "Y": 0.9}),
            "slate_3": PreferenceInterval({}),
        },
        "bloc_2": {
            "slate_1": PreferenceInterval({"A": 0.05, "B": 0.05}),
            "slate_2": PreferenceInterval({"X": 0.45, "Y": 0.45}),
        },
    }

    with pytest.warns(
        ConfigurationWarning,
        match=re.escape(
            "Preference mapping for bloc 'bloc_1' has slates ['slate_1', 'slate_2', 'slate_3'] "
            "but config has slates ['slate_1', 'slate_2']"
        ),
    ):
        config.preference_df = preference_mapping  # type: ignore[assignment]


def test_noop_on_append_slate_to_candidates(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100)
    original_slate_to_candidates = config.slate_to_candidates.copy()
    config.slate_to_candidates["slate_2"].append("X")
    assert config.slate_to_candidates == original_slate_to_candidates


def test_noop_on_extend_slate_to_candidates(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100)
    original_slate_to_candidates = config.slate_to_candidates.copy()
    config.slate_to_candidates["slate_2"].extend(["X"])
    assert config.slate_to_candidates == original_slate_to_candidates


def test_modify_cands_in_slate_to_candidates(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100)
    config.slate_to_candidates["slate_2"][0] = "Z"
    assert config.slate_to_candidates == {"slate_1": ["A", "B"], "slate_2": ["Z", "Y"]}
    assert (config.preference_df["Z"] == -1).all()


def test_modify_cands_with_cand_from_other_slate(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100)
    with pytest.raises(ValueError, match="already exist in slate"):
        config.slate_to_candidates["slate_2"][0] = "A"

    with pytest.raises(ValueError, match="already exist in slate"):
        config.slate_to_candidates["slate_2"] = ["X", "Y", "A"]


def test_modify_bloc_proportions_errors(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100)
    with pytest.raises(RuntimeError, match="Cannot set bloc proportions directly"):
        config.bloc_proportions["bloc_1"] = 0.5


def test_cohesion_df_duplicate_bloc_error(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100)
    with pytest.raises(ValueError, match=r"\(blocs\) contains duplicates."):
        config.cohesion_df = pd.concat(
            [
                config.cohesion_df,
                pd.DataFrame({"bloc_2": {"slate_1": 0.9, "slate_2": 0.1}}).T,
            ]
        )


def test_cohesion_df_duplicate_slate_error(valid_config):
    with pytest.warns(ConfigurationWarning):
        config = BlocSlateConfig(**valid_config, n_voters=100)
        with pytest.raises(ValueError, match=r"\(slates\) contains duplicates"):
            df = pd.DataFrame(config.cohesion_df).rename(columns={"slate_2": "slate_1"})
            config.cohesion_df = df


def test_preference_df_duplicate_cands_errors(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Preference interval for bloc 'bloc_2' and slate "
            "'slate_2' has candidates ['X'] which appear in other slates in the same bloc."
        ),
    ):
        config.preference_df = {  # type: ignore[assignment]
            "bloc_1": {
                "slate_1": PreferenceInterval({"A": 0.8, "B": 0.15}),
                "slate_2": PreferenceInterval({"X": 0.1, "Y": 0.9}),
            },
            "bloc_2": {
                "slate_1": PreferenceInterval({"A": 0.05, "B": 0.05, "X": 0.1}),
                "slate_2": PreferenceInterval({"X": 0.45, "Y": 0.45}),
            },
        }


def _call_validate(config: BlocSlateConfig, mapping):
    # Name-mangled call to the private validator
    return config._BlocSlateConfig__validate_cohesion_df_mapping_keys_ok_in_config(  # type: ignore[attr-defined]
        mapping
    )


# --- DF input: happy path / mismatches -----------------------------------


def test_validate_cohesion_df_ok_returns_true_no_warnings(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=False)
    df = config.cohesion_df.copy()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ok = _call_validate(config, df)
    assert ok is True
    assert not any(isinstance(x.message, ConfigurationWarning) for x in w)


def test_validate_cohesion_df_wrong_blocs_warns_and_false(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=False)
    # Same slates, wrong blocs
    df = pd.DataFrame(
        {"slate_1": {"bloc_1": 0.9}, "slate_2": {"bloc_3": 0.1}}  # bloc_3 unexpected
    ).astype(float)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ok = _call_validate(config, df)
    assert ok is False
    msgs = [str(x.message) for x in w if isinstance(x.message, ConfigurationWarning)]
    assert any("expected exactly the blocs" in m for m in msgs)


def test_validate_cohesion_df_wrong_slates_warns_and_false(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=False)
    # Correct blocs, wrong slates
    df = pd.DataFrame(
        {
            "slate_1": {"bloc_1": 0.9, "bloc_2": 0.1},
            "slate_X": {"bloc_1": 0.1, "bloc_2": 0.9},
        }
    ).astype(float)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ok = _call_validate(config, df)
    assert ok is False
    msgs = [str(x.message) for x in w if isinstance(x.message, ConfigurationWarning)]
    assert any("expected exactly the slates" in m for m in msgs)


# --- Mapping input: per-bloc slate mismatches + union mismatch ------------


def test_validate_cohesion_mapping_per_bloc_mismatch_and_union_warns(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=False)
    # Each bloc omits a slate → per-bloc mismatch messages + overall slate-set mismatch
    mapping = {
        "bloc_1": {"slate_1": 0.9},  # missing slate_2
        "bloc_2": {"slate_1": 0.2},  # missing slate_2
    }
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ok = _call_validate(config, mapping)
    assert ok is False
    msgs = [str(x.message) for x in w if isinstance(x.message, ConfigurationWarning)]
    # One message per bloc about their slate mismatch
    assert any("Cohesion mapping for bloc 'bloc_1' has slates" in m for m in msgs)
    assert any("Cohesion mapping for bloc 'bloc_2' has slates" in m for m in msgs)
    # Plus an overall "expected exactly the slates ..." message
    assert any("expected exactly the slates" in m for m in msgs)


# --- Special messages when config has no blocs / no slates ----------------


def test_validate_cohesion_mapping_has_blocs_but_config_has_none_message():
    # Config with NO blocs and NO slates
    with pytest.warns(ConfigurationWarning):
        config = BlocSlateConfig(n_voters=100, silent=False)
        mapping = {"bloc_1": {}}  # has a bloc, but config.blocs == []
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ok = _call_validate(config, mapping)
    assert ok is False
    msgs = [str(x.message) for x in w if isinstance(x.message, ConfigurationWarning)]
    assert any("has voter blocs but no blocs are defined in bloc_proportions" in m for m in msgs)
    # No slate message here because both sides have empty slate sets


def test_validate_cohesion_mapping_has_slates_but_config_has_none_message():
    # Config with blocs but NO slates
    config = BlocSlateConfig(
        n_voters=100,
        bloc_proportions={"bloc_1": 1.0},  # define blocs
        silent=False,
    )
    mapping = {"bloc_1": {"slate_1": 1.0}}  # config.slates == []
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ok = _call_validate(config, mapping)
    assert ok is False
    msgs = [str(x.message) for x in w if isinstance(x.message, ConfigurationWarning)]
    # Per-bloc slate mismatch
    assert any("Cohesion mapping for bloc 'bloc_1' has slates" in m for m in msgs)
    # Special "no slates are defined" message
    assert any("has slates but no slates are defined in slate_to_candidates" in m for m in msgs)


# --- Silent mode: no warnings but still returns False on mismatch ----------


def test_validate_cohesion_mapping_mismatch_silent_suppresses_warnings(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=True)
    mapping = {
        "bloc_1": {"slate_1": 1.0},
        "bloc_2": {"slate_1": 1.0},
    }  # missing slate_2
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ok = _call_validate(config, mapping)
    assert ok is False
    assert not any(isinstance(x.message, ConfigurationWarning) for x in w)


# ====================================================================================================================================


def _det_errs(config: BlocSlateConfig):
    # Call the private method via name-mangling
    return config._BlocSlateConfig__determine_errors()  # type: ignore[attr-defined]


def _messages(errs):
    # Use args[0] to avoid KeyError's quoting in str(e)
    return [e.args[0] for e in errs]


def test_determine_errors_on_valid_returns_empty(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=True)
    assert _det_errs(config) == []


# --- Top-level size/emptiness checks --------------------------------------


def test_error_when_voters_less_than_num_blocs(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=1, silent=True)  # 2 blocs
    msgs = _messages(_det_errs(config))
    assert any("must be >= number of blocs (2)" in m for m in msgs)


def test_error_when_no_blocs_defined():
    config = BlocSlateConfig(n_voters=10, silent=True)  # no blocs/slates/frames
    msgs = _messages(_det_errs(config))
    assert any("At least one voter bloc must be defined." in m for m in msgs)


def test_error_when_no_slates_defined_with_blocs():
    config = BlocSlateConfig(n_voters=10, bloc_proportions={"b1": 1.0}, silent=True)
    msgs = _messages(_det_errs(config))
    assert any("At least one slate and candidate list must be defined." in m for m in msgs)


def test_error_when_preference_df_empty(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=True)
    # Empty in place to bypass __setattr__ normalization
    config.preference_df.drop(config.preference_df.index, inplace=True)
    msgs = _messages(_det_errs(config))
    assert any("Preference mapping must be non-empty." in m for m in msgs)


def test_error_when_cohesion_df_empty(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=True)
    config.cohesion_df.drop(config.cohesion_df.index, inplace=True)
    msgs = _messages(_det_errs(config))
    assert any("Cohesion mapping must be non-empty." in m for m in msgs)


# --- Bloc proportion-specific errors --------------------------------------


def test_error_for_nonpositive_bloc_proportion(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=True)
    config.bloc_proportions = {  # type: ignore[assignment]
        "bloc_1": 0.0,
        "bloc_2": 1.00,
    }  # allowed by setter; flagged here
    msgs = _messages(_det_errs(config))
    assert any("has non-positive proportion 0.000000" in m for m in msgs)


def test_error_for_validate_after_low_sum_bloc_poportion(valid_config):
    with pytest.warns(ConfigurationWarning) as records:
        config = BlocSlateConfig(**valid_config, n_voters=100, silent=True)
        config.silent = False
        del config.bloc_proportions["bloc_1"]
        config.is_valid()

    assert any("sum to 1" in str(r.message) for r in records)


# --- preference_df structural errors --------------------------------------


def test_error_when_preference_columns_mismatch_candidates(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=True)
    # Remove one candidate column in place
    any_col = config.preference_df.columns[0]
    config.preference_df.drop(columns=[any_col], inplace=True)
    msgs = _messages(_det_errs(config))
    assert any("preference_df columns (candidates) must be exactly" in m for m in msgs)


def test_error_when_preference_index_mismatch_blocs(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=True)
    # Drop a bloc row in place
    any_bloc = config.preference_df.index[0]
    config.preference_df.drop(index=[any_bloc], inplace=True)
    msgs = _messages(_det_errs(config))
    assert any("preference_df index (blocs) must be exactly" in m for m in msgs)


def test_error_when_preference_contains_unset_minus_one(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=True)
    # Set a sentinel -1.0 that should be flagged
    b = config.preference_df.index[0]
    c = config.slate_to_candidates["slate_1"][0]
    config.preference_df.loc[b, c] = -1.0
    msgs = _messages(_det_errs(config))
    assert any("has values that have not been set (indicated with value of -1)" in m for m in msgs)


def test_error_when_preference_contains_invalid_negative_not_unset(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=True)
    bloc = config.preference_df.index[0]
    cand = config.slate_to_candidates["slate_1"][0]
    config.preference_df.loc[bloc, cand] = -0.25
    msgs = _messages(_det_errs(config))
    assert any("invalid negative values" in m and "Use -1 to mark unset values." in m for m in msgs)


def test_error_when_preference_contains_non_finite_values(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=True)
    bloc = config.preference_df.index[0]
    cand = config.slate_to_candidates["slate_1"][0]
    config.preference_df.loc[bloc, cand] = np.nan
    msgs = _messages(_det_errs(config))
    assert any("contains non-finite values" in m for m in msgs)


def test_error_when_preference_row_sum_not_one_per_slate(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=True)
    # Break slate_1 row sum for a bloc (set both to zero)
    bloc = config.preference_df.index[0]
    cands = list(config.slate_to_candidates["slate_1"])
    config.preference_df.loc[bloc, cands] = [0.5, 0.2]
    msgs = _messages(_det_errs(config))
    assert any("preference_df row for bloc" in m and "must sum to 1, got 0.7" in m for m in msgs)


def test_error_when_preference_row_is_missing_candidate(valid_config):
    # should not be allowed to leave X our of bloc 2 slate 2
    preference_mapping = {
        "bloc_1": {
            "slate_1": PreferenceInterval({"A": 0.4, "B": 0.1}),
            "slate_2": PreferenceInterval({"X": 0.1, "Y": 0.9}),
        },
        "bloc_2": {
            "slate_1": PreferenceInterval({"A": 0.05, "B": 0.05}),
            "slate_2": PreferenceInterval({"Y": 0.45}),
        },
    }
    with pytest.raises(
        ValueError,
        match="preference_df row for bloc 'bloc_2' has values that have not been set "
        r"\(indicated with value of -1\)\.",
    ):
        BlocSlateConfig(
            bloc_proportions=valid_config["bloc_proportions"],
            slate_to_candidates=valid_config["slate_to_candidates"],
            preference_mapping=preference_mapping,
            cohesion_mapping=valid_config["cohesion_mapping"],
            n_voters=100,
        ).is_valid(raise_errors=True)

    # should not be allowed to leave X our of bloc 1 slate 2
    preference_mapping = {
        "bloc_1": {
            "slate_1": {"A": 0.8, "B": 0.2},
            "slate_2": {"Y": 1.0},
        },
        "bloc_2": {
            "slate_1": {"A": 0.5, "B": 0.5},
            "slate_2": {"X": 0.5, "Y": 0.5},
        },
    }
    with pytest.raises(
        ValueError,
        match="preference_df row for bloc 'bloc_1' has values that have not been set "
        r"\(indicated with value of -1\)\.",
    ):
        BlocSlateConfig(
            bloc_proportions=valid_config["bloc_proportions"],
            slate_to_candidates=valid_config["slate_to_candidates"],
            preference_mapping=preference_mapping,
            cohesion_mapping=valid_config["cohesion_mapping"],
            n_voters=100,
        ).is_valid(raise_errors=True)


def test_error_when_preference_row_sum_not_one_with_zero_support_candidate(
    valid_config,
):
    config = BlocSlateConfig(**valid_config, n_voters=100)

    preference_mapping = {
        "bloc_1": {
            "slate_1": {"A": 0.8, "B": 0.2},
            "slate_2": {"X": 0, "Y": 0.9},
        },
        "bloc_2": {
            "slate_1": {"A": 0.5, "B": 0.5},
            "slate_2": {"X": 0.5, "Y": 0.5},
        },
    }

    with pytest.raises(
        ValueError,
        match="preference_df row for bloc 'bloc_1' and candidates "
        r"\['X', 'Y'\] must sum to 1, got 0.9",
    ):
        config.preference_df = preference_mapping  # type: ignore[assignment]
        config.is_valid(raise_errors=True)


def test_error_when_preference_row_has_zero_support_by_default(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100)

    preference_mapping = {
        "bloc_1": {
            "slate_1": {"A": 1.0, "B": 0.0},
            "slate_2": {"X": 0.1, "Y": 0.9},
        },
        "bloc_2": {
            "slate_1": {"A": 0.5, "B": 0.5},
            "slate_2": {"X": 0.5, "Y": 0.5},
        },
    }
    config.preference_df = preference_mapping  # type: ignore[assignment]

    with pytest.raises(ValueError, match="must have strictly positive support"):
        config.is_valid(raise_errors=True)


def test_zero_support_allowed_when_flag_set(valid_config):
    config = BlocSlateConfig(
        **valid_config,
        n_voters=100,
        allow_zero_support_candidates=True,
    )

    preference_mapping = {
        "bloc_1": {
            "slate_1": {"A": 1.0, "B": 0.0},
            "slate_2": {"X": 0.1, "Y": 0.9},
        },
        "bloc_2": {
            "slate_1": {"A": 0.5, "B": 0.5},
            "slate_2": {"X": 0.5, "Y": 0.5},
        },
    }
    config.preference_df = preference_mapping  # type: ignore[assignment]

    assert config.is_valid(raise_errors=True)


def test_init_with_zero_support_preference_mapping_rejected_by_default(valid_config):
    zero_support_pref_mapping = {
        "bloc_1": {
            "slate_1": {"A": 1.0, "B": 0.0},
            "slate_2": {"X": 0.1, "Y": 0.9},
        },
        "bloc_2": {
            "slate_1": {"A": 0.5, "B": 0.5},
            "slate_2": {"X": 0.5, "Y": 0.5},
        },
    }
    config = BlocSlateConfig(
        n_voters=100,
        slate_to_candidates=valid_config["slate_to_candidates"],
        bloc_proportions=valid_config["bloc_proportions"],
        cohesion_mapping=valid_config["cohesion_mapping"],
        preference_mapping=zero_support_pref_mapping,
    )

    with pytest.raises(ValueError, match="must have strictly positive support"):
        config.is_valid(raise_errors=True)


def test_init_with_zero_support_preference_mapping_allowed_with_flag(valid_config):
    zero_support_pref_mapping = {
        "bloc_1": {
            "slate_1": {"A": 1.0, "B": 0.0},
            "slate_2": {"X": 0.1, "Y": 0.9},
        },
        "bloc_2": {
            "slate_1": {"A": 0.5, "B": 0.5},
            "slate_2": {"X": 0.5, "Y": 0.5},
        },
    }
    config = BlocSlateConfig(
        n_voters=100,
        slate_to_candidates=valid_config["slate_to_candidates"],
        bloc_proportions=valid_config["bloc_proportions"],
        cohesion_mapping=valid_config["cohesion_mapping"],
        preference_mapping=zero_support_pref_mapping,
        allow_zero_support_candidates=True,
    )

    assert config.is_valid(raise_errors=True)


def test_toggle_allow_zero_support_candidates_after_init(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100)
    config.preference_df.loc["bloc_1", ["A", "B"]] = [0.0, 1.0]

    with pytest.raises(ValueError, match="must have strictly positive support"):
        config.is_valid(raise_errors=True)

    config.allow_zero_support_candidates = True
    assert config.is_valid(raise_errors=True)


def test_toggle_allow_zero_support_candidates_to_false_invalidates_config(valid_config):
    config = BlocSlateConfig(
        **valid_config,
        n_voters=100,
        allow_zero_support_candidates=True,
    )
    config.preference_df.loc["bloc_1", ["A", "B"]] = [0.0, 1.0]
    assert config.is_valid(raise_errors=True)

    config.allow_zero_support_candidates = False
    with pytest.raises(ValueError, match="must have strictly positive support"):
        config.is_valid(raise_errors=True)


def test_preference_df_dataframe_assignment_with_zero_support_respects_flag(
    valid_config,
):
    config = BlocSlateConfig(**valid_config, n_voters=100)
    preference_df_with_zero = config.preference_df.copy()
    preference_df_with_zero.loc["bloc_1", ["A", "B"]] = [1.0, 0.0]

    config.preference_df = preference_df_with_zero
    with pytest.raises(ValueError, match="must have strictly positive support"):
        config.is_valid(raise_errors=True)

    config.allow_zero_support_candidates = True
    assert config.is_valid(raise_errors=True)


# --- cohesion_df structural/content errors --------------------------------


def test_error_when_cohesion_columns_mismatch_slates(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=True)
    # Rename a column to create mismatch
    old = config.cohesion_df.columns[0]
    config.cohesion_df.rename(columns={old: f"{old}_X"}, inplace=True)
    msgs = _messages(_det_errs(config))
    assert any("cohesion_df columns (slates) must be exactly" in m for m in msgs)


def test_error_when_cohesion_index_mismatch_blocs(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=True)
    # Rename an index to create mismatch
    old = config.cohesion_df.index[0]
    config.cohesion_df.rename(index={old: f"{old}_X"}, inplace=True)
    msgs = _messages(_det_errs(config))
    assert any("cohesion_df index (blocs) must be exactly" in m for m in msgs)


def test_error_when_cohesion_contains_unset_minus_one(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=True)
    b = config.cohesion_df.index[0]
    s = config.cohesion_df.columns[0]
    config.cohesion_df.loc[b, s] = -1.0
    msgs = _messages(_det_errs(config))
    assert any(
        "cohesion_df row for bloc" in m and "have not been set (indicated with value of -1)" in m
        for m in msgs
    )


def test_error_when_cohesion_contains_invalid_negative_not_unset(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=True)
    bloc = config.cohesion_df.index[0]
    slate = config.cohesion_df.columns[0]
    config.cohesion_df.loc[bloc, slate] = -0.4
    msgs = _messages(_det_errs(config))
    assert any("invalid negative values" in m and "Use -1 to mark unset values." in m for m in msgs)


def test_error_when_cohesion_contains_non_finite_values(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=True)
    bloc = config.cohesion_df.index[0]
    slate = config.cohesion_df.columns[0]
    config.cohesion_df.loc[bloc, slate] = np.inf
    msgs = _messages(_det_errs(config))
    assert any("contains non-finite values" in m for m in msgs)


def test_error_when_cohesion_row_sum_not_one(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=True)
    b = config.cohesion_df.index[0]
    # Push row sum away from 1
    slates = list(config.cohesion_df.columns)
    if len(slates) >= 2:
        config.cohesion_df.loc[b, slates[:2]] = [0.9, 0.9]  # sum 1.8
    else:
        # Fallback for single-slate edge (shouldn't happen with valid_config)
        config.cohesion_df.loc[b, slates[0]] = 1.8
    msgs = _messages(_det_errs(config))
    assert any("cohesion_df row for bloc" in m and "must sum to 1" in m for m in msgs)


# ---------- is_valid / warnings / errors ----------


def test_is_valid_true_on_fully_valid(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=False)
    assert config.is_valid() is True


def test_is_valid_warns_when_invalid_and_raise_warnings_true(valid_config):
    with pytest.warns(ConfigurationWarning) as records:
        config = BlocSlateConfig(**valid_config, n_voters=100, silent=False)
        # Make invalid: wipe preference_df to trigger "must be non-empty" + others
        config.preference_df = pd.DataFrame()
        assert config.is_valid(raise_errors=False, raise_warnings=True) is False

    assert any("Preference mapping must be non-empty" in str(r.message) for r in records)


def test_is_valid_raises_first_error_when_raise_errors_true(valid_config):
    with pytest.warns(ConfigurationWarning):
        config = BlocSlateConfig(**valid_config, n_voters=100, silent=False)
        # Break cohesion_df completely
        config.cohesion_df = pd.DataFrame()
        with pytest.raises(ValueError, match="Cohesion mapping must be non-empty"):
            config.is_valid(raise_errors=True)


# ---------- normalization helpers ----------


def test_normalize_cohesion_df_sets_minus_one_to_zero_and_row_sums_to_one(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=True)
    # Introduce a sentinel -1.0 that should be zeroed before normalization
    config.cohesion_df.loc["bloc_1", "slate_1"] = -1.0
    config.normalize_cohesion_df()
    assert np.allclose(config.cohesion_df.sum(axis=1).to_numpy(), 1.0, atol=1e-9)
    assert (config.cohesion_df.values >= 0).all()


def test_normalize_cohesion_df_rejects_invalid_negative_values(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=True)
    config.cohesion_df.loc["bloc_1", "slate_1"] = -0.2
    with pytest.raises(
        ValueError,
        match="contains invalid negative values .*Use -1 to mark unset values",
    ):
        config.normalize_cohesion_df()


def test_normalize_cohesion_df_rejects_zero_sum_row(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=True)
    config.cohesion_df.loc["bloc_1", :] = -1.0

    with pytest.raises(
        ValueError,
        match=re.escape(
            "cohesion_df row for bloc 'bloc_1' sums to 0 after replacing unset values "
            "with 0 and cannot be normalized."
        ),
    ):
        config.normalize_cohesion_df()


def test_normalize_preference_intervals_sets_minus_one_to_zero_and_row_sums_by_slate(
    valid_config,
):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=True)
    # Add a new candidate column with -1 default via add_slate (covered below),
    # but here simulate directly:
    new_cols = ["Z1", "Z2"]
    for c in new_cols:
        config.preference_df[c] = -1.0
    # Normalization should zero those and renormalize per-slate groups
    config.normalize_preference_intervals()
    # Check each slate's columns per bloc sum to ~1
    for _, cand_list in config.slate_to_candidates.items():
        sub = config.preference_df[list(cand_list)]
        assert np.allclose(sub.sum(axis=1).to_numpy(), 1.0, atol=1e-9)


def test_normalize_preference_intervals_rejects_invalid_negative_values(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=True)
    config.preference_df.loc["bloc_1", "A"] = -0.2
    with pytest.raises(
        ValueError,
        match="contains invalid negative values .*Use -1 to mark unset values",
    ):
        config.normalize_preference_intervals()


def test_normalize_preference_intervals_rejects_zero_sum_slate_row(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=True)
    config.add_slate("slate_3", ["P", "Q"])

    with pytest.raises(
        ValueError,
        match=re.escape(
            "preference_df row for bloc 'bloc_1' and slate 'slate_3' sums to 0 after "
            "replacing unset values with 0 and cannot be normalized."
        ),
    ):
        config.normalize_preference_intervals()


# ---------- __setattr__-driven updates to DFs ----------


def test_setting_slate_to_candidates_adds_new_candidate_columns_with_minus_one(
    valid_config,
):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=True)
    new_cand = "Z"
    new_map = config.slate_to_candidates.to_dict()
    new_map["slate_2"] = new_map["slate_2"] + [new_cand]
    config.slate_to_candidates = new_map  # type: ignore[assignment]
    assert new_cand in config.preference_df.columns
    # freshly added candidate gets -1 before any normalization
    assert (config.preference_df[new_cand] == -1.0).all()
    # Cohesion DF should also have columns matching slates and be reordered if needed
    assert set(config.cohesion_df.columns) == set(config.slates)


def test_setting_slate_to_candidates_reorders_cohesion_columns_when_only_order_changes(
    valid_config,
):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=True)

    reordered = {
        "slate_2": list(config.slate_to_candidates["slate_2"]),
        "slate_1": list(config.slate_to_candidates["slate_1"]),
    }
    config.slate_to_candidates = reordered  # type: ignore[assignment]

    assert list(config.cohesion_df.columns) == config.slates


def test_setting_bloc_proportions_adds_and_removes_rows_with_minus_one(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=True)

    # Now remove a bloc
    bp2 = config.bloc_proportions.copy()
    bp2.pop("bloc_1")
    total = sum(bp2.values())
    bp2 = {k: v / total for k, v in bp2.items()}  # re-normalize
    config.bloc_proportions = bp2  # type: ignore[assignment]
    assert "bloc_1" not in config.preference_df.index
    assert "bloc_1" not in config.cohesion_df.index


def test_setting_bloc_proportions_reorders_rows_when_only_order_changes(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=True)

    reordered = {
        "bloc_2": config.bloc_proportions["bloc_2"],
        "bloc_1": config.bloc_proportions["bloc_1"],
    }
    config.bloc_proportions = reordered  # type: ignore[assignment]

    assert list(config.preference_df.index) == config.blocs
    assert list(config.cohesion_df.index) == config.blocs


# ---------- dirichlet alphas ----------


def test_set_dirichlet_alphas_warns_when_overwriting_existing_prefs(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=False)
    # Build alphas mapping bloc -> slate -> alpha
    alphas = {bloc: {slate: 2.0 for slate in config.slates} for bloc in config.blocs}
    with pytest.warns(ConfigurationWarning, match="overwrite the existing preference"):
        config.set_dirichlet_alphas(alphas)

    # After setting, prefs are resampled; for each slate's candidates, rows sum to ~1
    for _, cand_list in config.slate_to_candidates.items():
        sub = config.preference_df[list(cand_list)]
        assert np.allclose(sub.sum(axis=1).to_numpy(), 1.0, atol=1e-9)

    # read/clear round trip
    rd = config.read_dirichlet_alphas()
    assert rd is not None
    assert set(rd.index) == set(config.blocs)
    assert set(rd.columns) == set(config.slates)

    config.clear_dirichlet_alphas()
    assert config.read_dirichlet_alphas() is None


# ---------- read-only properties ----------


def test_readonly_properties(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=True)
    with pytest.raises(AttributeError, match="read-only property"):
        setattr(config, "candidates", [])
    with pytest.raises(AttributeError, match="read-only property"):
        setattr(config, "slates", [])
    with pytest.raises(AttributeError, match="read-only property"):
        setattr(config, "blocs", [])


# ---------- copy semantics ----------


def test_copy_is_independent(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=True)
    clone = config.copy()

    # Equal content initially
    assert clone.n_voters == config.n_voters
    assert clone.slate_to_candidates.to_dict() == config.slate_to_candidates.to_dict()
    assert clone.bloc_proportions == config.bloc_proportions.copy()
    assert clone.preference_df.equals(config.preference_df)

    # Mutate original; copy should not change
    config.slate_to_candidates["slate_2"].append("Z")
    config.bloc_proportions = {"bloc_1": 0.51, "bloc_2": 0.49}  # type: ignore[assignment]
    assert not clone.slate_to_candidates.to_dict() == config.slate_to_candidates.to_dict()
    assert not clone.bloc_proportions == config.bloc_proportions.copy()


# ---------- additional validation warnings ----------


def test_pref_mapping_with_unknown_candidate_warns(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=False)
    bad = {
        "bloc_1": {
            "slate_1": {"A": 0.7, "B": 0.3},
            "slate_2": {"Z": 1.0},
        },  # Z not in config
        "bloc_2": {"slate_1": {"A": 0.5, "B": 0.5}, "slate_2": {"X": 0.5, "Y": 0.5}},
    }
    with pytest.warns(
        ConfigurationWarning,
        match=r"Preference contains candidates not present in slate_to_candidates.",
    ):
        config.preference_df = bad  # type: ignore[assignment]


# ---------- add/remove slate ----------


def test_add_and_remove_slate_updates_frames(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=True)

    config.add_slate("slate_3", ["P", "Q"])
    assert "slate_3" in config.slates
    assert {"P", "Q"}.issubset(set(config.preference_df.columns))
    assert (config.preference_df[["P", "Q"]] == -1.0).all().all()
    assert "slate_3" in config.cohesion_df.columns
    assert (config.cohesion_df["slate_3"] == -1.0).all()

    config.remove_slate("slate_3")
    assert "slate_3" not in config.slates
    assert "P" not in config.preference_df.columns
    assert "Q" not in config.preference_df.columns
    assert "slate_3" not in config.cohesion_df.columns


def test_add_slate_with_invalid_type(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=True)

    with pytest.raises(TypeError, match="slate_candidate_list must be a sequence of str"):
        config.add_slate("slate_3", 1)  # type: ignore[list-item]


def test_add_slate_with_duplicated_list(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=True)

    with pytest.raises(
        ValueError, match="slate_candidate_list cannot contain duplicate candidates."
    ):
        config.add_slate("slate_3", ["P", "Q", "Q"])


# ---------- remove_candidates ----------


def test_remove_candidates_removes_columns_and_maybe_slates(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=True)
    # Remove a candidate that exists
    existing = config.slate_to_candidates["slate_2"][0]  # e.g., 'X'
    config.remove_candidates(existing)
    assert existing not in config.candidates
    assert existing not in config.preference_df.columns


# ------- rename candidate -------


def test_rename_candidate_updates_columns_and_slate_mapping(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=True)
    new_names = {"A": "B", "B": "X", "X": "Z"}
    config.rename_candidates(new_names)
    assert config.is_valid()
    new_df = {
        "bloc_1": {"B": 0.8, "X": 0.2, "Z": 0.1, "Y": 0.9},
        "bloc_2": {"B": 0.5, "X": 0.5, "Z": 0.5, "Y": 0.5},
    }

    pdt.assert_frame_equal(pd.DataFrame(new_df).T, config.preference_df)


def test_rename_candidate_type_errors(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=True)
    new_names = 3
    with pytest.raises(TypeError, match="candidate_mapping must be a mapping"):
        config.rename_candidates(new_names)  # type: ignore[arg-type]

    new_names = {1: "B", "B": "X", "Z": "Y"}  # Z not in config

    with pytest.raises(TypeError, match=re.escape("Candidate mapping keys must be a 'str', got")):
        config.rename_candidates(cast(dict[str, str], new_names))

    new_names = {"A": 1, "B": "X", "Z": "Y"}  # Z not in config
    with pytest.raises(TypeError, match=re.escape("Candidate mapping values must be a 'str', got")):
        config.rename_candidates(cast(dict[str, str], new_names))


def test_rename_candidate_with_missing_name(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=True)
    new_names = {"A": "B", "B": "X", "Z": "Y"}  # Z not in config
    with pytest.raises(ValueError, match="not present in configuration"):
        config.rename_candidates(new_names)


def test_rename_candidate_with_duplicate_names(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=True)
    new_names = {"A": "B"}  # B already exists
    with pytest.raises(ValueError, match="Candidate mapping results in duplicate candidate names."):
        config.rename_candidates(new_names)


# -----------  check some update paths --------------


def _update_prefs(config: BlocSlateConfig):
    # call the private method via name-mangling
    config._BlocSlateConfig__update_preference_df_on_candidate_change()  # type: ignore[attr-defined]


def _update_cohesion(config: BlocSlateConfig):
    config._BlocSlateConfig__update_cohesion_df_on_slate_change()  # type: ignore[attr-defined]


def test_pref_update_creates_shape_from_empty(extra_profile_settings):
    config = BlocSlateConfig(
        n_voters=100,
        slate_to_candidates=extra_profile_settings["slates"],
        bloc_proportions=extra_profile_settings["blocs"],
        preference_mapping=pd.DataFrame(),  # start empty on purpose
        cohesion_mapping=extra_profile_settings["cohesion_df"],
        silent=True,
    )
    _update_prefs(config)
    assert list(config.preference_df.index) == list(extra_profile_settings["blocs"].keys())
    assert (
        list(config.preference_df.columns)
        == extra_profile_settings["slates"]["slate_1"] + extra_profile_settings["slates"]["slate_2"]
    )
    assert (config.preference_df.values == -1.0).all()


def test_pref_update_drops_columns_not_in_config(extra_profile_settings):
    config = BlocSlateConfig(
        n_voters=100,
        slate_to_candidates=extra_profile_settings["slates"],
        bloc_proportions=extra_profile_settings["blocs"],
        preference_mapping=extra_profile_settings["pref_df_with_extra_col"],
        cohesion_mapping=extra_profile_settings["cohesion_df"],
        silent=True,
    )
    # update should remove the column even though the init has it
    assert "Z" not in config.preference_df.columns


def test_pref_update_adds_new_candidate_with_minus_one(extra_profile_settings):
    config = BlocSlateConfig(
        n_voters=100,
        slate_to_candidates=extra_profile_settings["slates"],
        bloc_proportions=extra_profile_settings["blocs"],
        preference_mapping=extra_profile_settings["pref_df_base"],
        cohesion_mapping=extra_profile_settings["cohesion_df"],
        silent=True,
    )
    # Add a new candidate to an existing slate
    new_map = config.slate_to_candidates.to_dict()
    new_map["slate_2"] = new_map["slate_2"] + ["Z"]
    config.slate_to_candidates = (  # type: ignore[assignment]
        new_map  # triggers internal updater too, but we also call directly
    )
    _update_prefs(config)
    assert "Z" in config.preference_df.columns
    assert (config.preference_df["Z"] == -1.0).all()


def test_pref_update_reorders_to_match_config_candidate_order(extra_profile_settings):
    config = BlocSlateConfig(
        n_voters=100,
        slate_to_candidates=extra_profile_settings["slates"],
        bloc_proportions=extra_profile_settings["blocs"],
        preference_mapping=extra_profile_settings["pref_df_out_of_order"],
        cohesion_mapping=extra_profile_settings["cohesion_df"],
        silent=True,
    )
    _update_prefs(config)
    assert list(config.preference_df.columns) == config.candidates


def test_cohesion_update_creates_shape_from_empty(extra_profile_settings):
    config = BlocSlateConfig(
        n_voters=100,
        slate_to_candidates=extra_profile_settings["slates"],
        bloc_proportions=extra_profile_settings["blocs"],
        preference_mapping=extra_profile_settings["pref_df_base"],
        cohesion_mapping=pd.DataFrame(),
        silent=True,
    )
    _update_cohesion(config)
    assert list(config.cohesion_df.index) == list(extra_profile_settings["blocs"].keys())
    assert list(config.cohesion_df.columns) == list(extra_profile_settings["slates"].keys())
    assert (config.cohesion_df.values == -1.0).all()


def test_cohesion_update_adds_and_removes_slates(extra_profile_settings):
    config = BlocSlateConfig(
        n_voters=100,
        slate_to_candidates=extra_profile_settings["slates"],
        bloc_proportions=extra_profile_settings["blocs"],
        preference_mapping=extra_profile_settings["pref_df_base"],
        cohesion_mapping=extra_profile_settings["cohesion_df"],
        silent=True,
    )
    # Add a slate
    new_map = config.slate_to_candidates.to_dict()
    new_map["slate_3"] = ["P", "Q"]
    config.slate_to_candidates = new_map  # type: ignore[assignment]
    _update_cohesion(config)
    assert "slate_3" in config.cohesion_df.columns
    assert (config.cohesion_df["slate_3"] == -1.0).all()

    # Remove a slate
    new_map.pop("slate_1")
    config.slate_to_candidates = new_map  # type: ignore[assignment]
    _update_cohesion(config)
    assert "slate_1" not in config.cohesion_df.columns


def _update_prefs_bloc_change(config: BlocSlateConfig):
    # name-mangled call into the private method
    config._BlocSlateConfig__update_preference_df_on_bloc_change()  # type: ignore[attr-defined]


def test_pref_update_on_bloc_change_creates_shape_from_empty():
    # Have slates/candidates and blocs, but NO preference mapping -> empty preference_df
    slates = {"slate_1": ["A", "B"], "slate_2": ["X", "Y"]}
    blocs = {"bloc_1": 0.7, "bloc_2": 0.3}

    config = BlocSlateConfig(
        n_voters=100,
        slate_to_candidates=slates,
        bloc_proportions=blocs,
        preference_mapping=None,  # keep preference_df empty
        cohesion_mapping=None,  # irrelevant here
        silent=True,
    )
    assert config.preference_df.empty  # precondition

    _update_prefs_bloc_change(config)  # should create the -1.0 matrix

    assert list(config.preference_df.index) == list(blocs.keys())
    # candidates are flattened from slates in order
    assert list(config.preference_df.columns) == slates["slate_1"] + slates["slate_2"]
    assert config.preference_df.dtypes.eq(float).all()  # type: ignore[union-attr]
    assert (config.preference_df.values == -1.0).all()


def test_pref_update_on_bloc_change_empty_via_setter_path():
    # Same setup, but invoke via setting bloc_proportions (exercise public path)
    slates = {"slate_1": ["A", "B"], "slate_2": ["X", "Y"]}
    blocs = {"bloc_1": 0.7, "bloc_2": 0.3}

    config = BlocSlateConfig(
        n_voters=100,
        slate_to_candidates=slates,
        bloc_proportions=blocs,
        preference_mapping=None,  # keep preference_df empty
        cohesion_mapping=None,
        silent=True,
    )
    assert config.preference_df.empty  # precondition

    # Re-assign bloc_proportions to trigger _update_preference_and_cohesion_blocs()
    config.bloc_proportions = blocs  # type: ignore[assignment]

    assert list(config.preference_df.index) == list(blocs.keys())
    assert list(config.preference_df.columns) == slates["slate_1"] + slates["slate_2"]
    assert (config.preference_df.values == -1.0).all()


def test_unset_candidate_preferences_sets_to_negative_1(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=True)
    # Set some preferences to -1.0 to simulate unset
    config.unset_candidate_preferences(["A", "X"])
    assert (config.preference_df["A"] == -1.0).all()
    assert (config.preference_df["X"] == -1.0).all()


def test_unset_candidate_preferences_does_nothing_on_non_cands(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=True)
    # Set some preferences to -1.0 to simulate unset
    config.unset_candidate_preferences(["Q"])
    assert ((config.preference_df != -1.0).all()).all()


def test_add_slate_appends_candidates_and_updates_frames(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=True)

    # Add a new slate with duplicate candidate names to exercise
    config.add_slate("slate_3", ["P", "Q"])

    assert "slate_3" in config.slates
    assert config.slate_to_candidates["slate_3"] == ["P", "Q"]

    # Preference DF: new candidate columns exist and are -1.0
    assert {"P", "Q"}.issubset(set(config.preference_df.columns))
    assert (config.preference_df[["P", "Q"]] == -1.0).all().all()

    # Cohesion DF: new slate column exists and is -1.0
    assert "slate_3" in config.cohesion_df.columns
    assert (config.cohesion_df["slate_3"] == -1.0).all()

    # Column order in prefs matches config.candidates (A,B,X,Y,P,Q)
    assert list(config.preference_df.columns) == config.candidates


def test_add_slate_rejects_existing_name_and_cross_candidate(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=True)

    # Existing slate name
    with pytest.raises(ValueError, match="already present in configuration"):
        config.add_slate("slate_2", ["Z1", "Z2"])

    # Candidate already present in configuration
    with pytest.raises(ValueError, match="already present in configuration"):
        config.add_slate("slate_3", ["A", "Z"])

    # Empty candidate list
    with pytest.raises(ValueError, match="cannot be empty"):
        config.add_slate("slate_3", [])

    # Non-string candidate type
    with pytest.raises(TypeError, match="candidates must be a 'str'"):
        config.add_slate("slate_3", ["Z", 123])  # type: ignore[list-item]


def test_remove_slate_removes_candidates_and_cohesion_column(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=True)
    config.add_slate("slate_3", ["P", "Q"])

    # Precondition
    assert {"P", "Q"}.issubset(set(config.preference_df.columns))
    assert "slate_3" in config.cohesion_df.columns

    # Remove and verify cleanup
    config.remove_slate("slate_3")
    assert "slate_3" not in config.slates
    assert "P" not in config.preference_df.columns
    assert "Q" not in config.preference_df.columns
    assert "slate_3" not in config.cohesion_df.columns


def test_direct_slate_delete_keeps_frames_in_sync(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=True)

    del config.slate_to_candidates["slate_2"]

    assert config.slates == ["slate_1"]
    assert set(config.candidates) == {"A", "B"}
    assert set(config.preference_df.columns) == {"A", "B"}
    assert list(config.cohesion_df.columns) == ["slate_1"]


def test_remove_candidates_single_then_drop_empty_slate(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=True)

    # Remove one candidate from slate_2 — slate remains
    config.remove_candidates("X")
    assert "X" not in config.candidates
    assert "X" not in config.preference_df.columns
    assert "slate_2" in config.slates  # still has 'Y'

    # Remove the last remaining candidate → slate_2 should disappear
    config.remove_candidates("Y")
    assert "Y" not in config.candidates
    assert "slate_2" not in config.slates
    assert "slate_2" not in config.cohesion_df.columns


# -------- remove_slate: early return when slate not present --------


def test_remove_slate_errors_when_missing(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=True)

    # Non-existent slate → should hit the early return and no-op
    with pytest.raises(KeyError, match="not found in configuration"):
        config.remove_slate("not_a_slate")


# -------- remove_candidates: early return when no overlap --------


def test_remove_candidates_noop_when_missing_single(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=True)

    cands_before = config.candidates[:]  # list copy
    pref_before = config.preference_df.copy()
    coh_before = config.cohesion_df.copy()
    slates_before = config.slate_to_candidates.copy()

    # Non-existent single candidate (string path) → early return
    config.remove_candidates("ZZZ")

    assert config.candidates == cands_before
    assert config.slate_to_candidates == slates_before
    pdt.assert_frame_equal(config.preference_df, pref_before)
    pdt.assert_frame_equal(config.cohesion_df, coh_before)


def test_remove_candidates_noop_when_missing_list(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=True)

    cands_before = config.candidates[:]
    pref_before = config.preference_df.copy()
    coh_before = config.cohesion_df.copy()
    slates_before = config.slate_to_candidates.copy()

    # Non-existent list (list path). This also guards against the internal
    # 'candidate' variable being undefined because we return before the loop.
    config.remove_candidates(["ZZZ", "QQQ"])

    assert config.candidates == cands_before
    assert config.slate_to_candidates == slates_before
    pdt.assert_frame_equal(config.preference_df, pref_before)
    pdt.assert_frame_equal(config.cohesion_df, coh_before)


def test_direct_bloc_delete_keeps_frames_in_sync(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=True)

    del config.bloc_proportions["bloc_2"]

    assert config.blocs == ["bloc_1"]
    assert list(config.preference_df.index) == ["bloc_1"]
    assert list(config.cohesion_df.index) == ["bloc_1"]


# -----------  dirichlet alphas keycheck --------------


def _keycheck(config: BlocSlateConfig, alphas):
    config._BlocSlateConfig__keycheck_dirichlet_alphas(alphas)  # type: ignore[attr-defined]


def test_dirichlet_alphas_df_valid_passes(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=True)
    df = pd.DataFrame(
        {
            "slate_1": {"bloc_1": 1.0, "bloc_2": 2.0},
            "slate_2": {"bloc_1": 3.0, "bloc_2": 4.0},
        }
    ).astype(float)
    # Should not raise
    _keycheck(config, df)


def test_dirichlet_alphas_df_rejects_non_str_index(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=True)
    df = pd.DataFrame({"slate_1": {1: 1.0}, "slate_2": {2: 2.0}}).astype(float)
    with pytest.raises(TypeError, match="index \\(blocs\\) must be a 'str'"):
        _keycheck(config, df)


def test_dirichlet_alphas_df_rejects_non_str_columns(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=True)
    df = pd.DataFrame({1: {"bloc_1": 1.0}, 2: {"bloc_2": 2.0}}).astype(float)
    with pytest.raises(TypeError, match="columns \\(slates\\) must be a 'str'"):
        _keycheck(config, df)


def test_dirichlet_alphas_df_rejects_non_float_dtype(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=True)
    df = pd.DataFrame(
        {"slate_1": {"bloc_1": 1, "bloc_2": 2}, "slate_2": {"bloc_1": 3, "bloc_2": 4}}
    ).astype(int)
    with pytest.raises(TypeError, match="must have float dtypes in every column"):
        _keycheck(config, df)


def test_dirichlet_alphas_df_rejects_non_finite(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=True)
    df = pd.DataFrame(
        {
            "slate_1": {"bloc_1": np.nan, "bloc_2": 1.0},
            "slate_2": {"bloc_1": 1.0, "bloc_2": 1.0},
        }
    ).astype(float)
    with pytest.raises(ValueError, match="contains non-finite values"):
        _keycheck(config, df)


def test_dirichlet_alphas_df_rejects_non_positive(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=True)
    df = pd.DataFrame(
        {
            "slate_1": {"bloc_1": 0.0, "bloc_2": 1.0},
            "slate_2": {"bloc_1": 1.0, "bloc_2": 1.0},
        }
    ).astype(float)
    with pytest.raises(ValueError, match="Dirichlet alphas must be positive finite reals."):
        _keycheck(config, df)


def test_dirichlet_alphas_df_rejects_wrong_blocs_set(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=True)
    # drop bloc_2
    df = pd.DataFrame({"slate_1": {"bloc_1": 1.0}, "slate_2": {"bloc_1": 1.0}}).astype(float)
    with pytest.raises(ValueError, match="must have exactly the blocs"):
        _keycheck(config, df)


def test_dirichlet_alphas_df_rejects_wrong_slates_set(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=True)
    # drop slate_2
    df = pd.DataFrame({"slate_1": {"bloc_1": 1.0, "bloc_2": 1.0}}).astype(float)
    with pytest.raises(ValueError, match="must have exactly the slates"):
        _keycheck(config, df)


def test_dirichlet_alphas_map_rejects_non_str_bloc_key(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=True)
    alphas = {
        1: {"slate_1": 1.0, "slate_2": 1.0},  # non-str bloc key
        "bloc_2": {"slate_1": 1.0, "slate_2": 1.0},
    }
    with pytest.raises(TypeError, match="bloc keys must be a 'str'"):
        _keycheck(config, alphas)


def test_dirichlet_alphas_map_rejects_wrong_blocs_set(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=True)
    # missing one bloc (but keys are strings)
    alphas = {"bloc_1": {"slate_1": 1.0, "slate_2": 1.0}}
    with pytest.raises(ValueError, match="must have exactly the blocs"):
        _keycheck(config, alphas)


def test_dirichlet_alphas_map_rejects_non_mapping_slate_map(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=True)

    class FakeNotMapping:
        # looks like it has keys, but isn't a Mapping
        def __init__(self, d):
            self._d = d

        def keys(self):
            return self._d.keys()

    # Deliberately wrong slate set so the inner TypeError is hit before iterating items()
    alphas = {
        "bloc_1": FakeNotMapping({"slate_1": 1.0}),
        "bloc_2": FakeNotMapping({"slate_1": 1.0}),
    }
    with pytest.raises(TypeError, match="alphas must be a mapping"):
        _keycheck(config, alphas)


def test_dirichlet_alphas_map_rejects_wrong_slates_set(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=True)
    # slate map missing slate_2
    alphas = {
        "bloc_1": {"slate_1": 1.0},
        "bloc_2": {"slate_1": 1.0},
    }
    with pytest.raises(ValueError, match="must have exactly the slates"):
        _keycheck(config, alphas)


def test_dirichlet_alphas_map_rejects_non_str_slate_key_even_if_sets_match(
    valid_config,
):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=True)

    class KeyLike:
        # compares equal & hashes like the underlying string, but is NOT a str
        def __init__(self, s):
            self.s = s

        def __hash__(self):
            return hash(self.s)

        def __eq__(self, other):
            return self.s == other

    # keys "equal" to expected slates so set equality passes, but types are wrong
    alphas = {
        "bloc_1": {KeyLike("slate_1"): 1.0, KeyLike("slate_2"): 1.0},
        "bloc_2": {KeyLike("slate_1"): 1.0, KeyLike("slate_2"): 1.0},
    }
    with pytest.raises(TypeError, match="slate keys must be a 'str'"):
        _keycheck(config, alphas)


def test_dirichlet_alphas_map_rejects_value_not_real(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=True)
    alphas = {
        "bloc_1": {"slate_1": "1.0", "slate_2": 1.0},  # string is not Real
        "bloc_2": {"slate_1": 1.0, "slate_2": 1.0},
    }
    with pytest.raises(TypeError, match="must be a finite real"):
        _keycheck(config, alphas)


@pytest.mark.parametrize("bad", [np.nan, np.inf, 0.0, -1.0, True])
def test_dirichlet_alphas_map_rejects_non_positive_or_non_finite(valid_config, bad):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=True)
    alphas = {
        "bloc_1": {"slate_1": bad, "slate_2": 1.0},
        "bloc_2": {"slate_1": 1.0, "slate_2": 1.0},
    }
    with pytest.raises(ValueError, match="must be a positive\\s+finite real"):
        _keycheck(config, alphas)


def test_dirichlet_alphas_map_valid_passes_and_setter_roundtrip(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=True)
    alphas = {b: {s: 2.0 for s in config.slates} for b in config.blocs}
    # Should not raise; also sets and resamples
    config.set_dirichlet_alphas(alphas)
    rd = config.read_dirichlet_alphas()
    assert rd is not None
    assert set(rd.index) == set(config.blocs)
    assert set(rd.columns) == set(config.slates)


def test_dirichlet_alphas_map_valid_passes_and_setter_roundtrip_with_df(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=True)
    alphas = pd.DataFrame({b: {s: 2.0 for s in config.slates} for b in config.blocs}).T
    # Should not raise; also sets and resamples
    config.set_dirichlet_alphas(alphas)
    rd = config.read_dirichlet_alphas()
    assert rd is not None
    assert set(rd.index) == set(config.blocs)
    assert set(rd.columns) == set(config.slates)


def test_sample_when_dirichlet_alphas_not_set(valid_config):
    config = BlocSlateConfig(**valid_config, n_voters=100, silent=True)
    with pytest.raises(ValueError, match="Dirichlet alphas have not been set"):
        config.resample_preference_intervals_from_dirichlet_alphas()


def test_drops_candidates_when_slate_removed(valid_config):
    """
    Previous mapping has slate_2 = ['X','Y'].
    After removing slate_2 entirely, columns 'X' and 'Y' should be dropped
    from preference_df.
    """
    cfg = BlocSlateConfig(**valid_config, n_voters=100, silent=True)

    # sanity preconditions
    assert set(cfg.candidates) == {"A", "B", "X", "Y"}
    assert set(cfg.preference_df.columns) == {"A", "B", "X", "Y"}

    # remove the entire slate_2 -> should drop X and Y
    cfg.remove_slate("slate_2")

    assert "slate_2" not in cfg.slates
    assert set(cfg.candidates) == {"A", "B"}
    assert set(cfg.preference_df.columns) == {"A", "B"}  # key assertion


def test_drops_candidate_removed_from_existing_slate(valid_config):
    """
    Previous mapping has slate_1 = ['A','B'].
    After changing slate_1 to ['A'], column 'B' should be dropped
    from preference_df.
    """
    cfg = BlocSlateConfig(**valid_config, n_voters=100, silent=True)

    # sanity preconditions
    assert set(cfg.candidates) == {"A", "B", "X", "Y"}
    assert set(cfg.preference_df.columns) == {"A", "B", "X", "Y"}

    # remove 'B' from slate_1 (routes through SlateCandMap and triggers updates)
    cfg.slate_to_candidates["slate_1"] = ["A"]

    assert set(cfg.candidates) == {"A", "X", "Y"}
    assert set(cfg.preference_df.columns) == {"A", "X", "Y"}


# -------- get_preference_interval_for_bloc_and_slate -------------


def test_returns_interval_for_specific_bloc_and_slate(valid_config):
    cfg = BlocSlateConfig(**valid_config, n_voters=100, silent=True)

    bloc_name = "bloc_1"
    slate_name = "slate_1"
    cand_list = list(cfg.slate_to_candidates[slate_name])

    expected = {c: float(cfg.preference_df[c].loc[bloc_name]) for c in cand_list}

    out = cfg.get_preference_interval_for_bloc_and_slate(bloc_name=bloc_name, slate_name=slate_name)
    assert isinstance(out, PreferenceInterval)
    assert set(out.interval.keys()) == set(cand_list)
    for k, v in expected.items():
        assert out.interval[k] == pytest.approx(v, abs=1e-12)
    assert sum(out.interval.values()) == pytest.approx(1.0, abs=1e-12)


def test_raises_keyerror_for_unknown_slate(valid_config):
    cfg = BlocSlateConfig(**valid_config, n_voters=100, silent=True)
    with pytest.raises(KeyError, match="Slate 'not_a_slate' not found"):
        cfg.get_preference_interval_for_bloc_and_slate(bloc_name="bloc_1", slate_name="not_a_slate")


def test_other_missing_bloc_does_not_block_requested_interval(valid_config):
    cfg = BlocSlateConfig(**valid_config, n_voters=100, silent=True)

    # Remove an unrelated bloc row; fetching bloc_1 should still work.
    cfg.preference_df.drop(index=["bloc_2"], inplace=True)

    out = cfg.get_preference_interval_for_bloc_and_slate(bloc_name="bloc_1", slate_name="slate_1")
    assert out.interval == {"A": 0.8, "B": 0.2}


def test_raises_keyerror_when_requested_bloc_missing_from_preference_df(valid_config):
    cfg = BlocSlateConfig(**valid_config, n_voters=100, silent=True)

    cfg.preference_df.drop(index=["bloc_1"], inplace=True)

    with pytest.raises(KeyError, match=r"Bloc 'bloc_1' not found in preference_df index"):
        cfg.get_preference_interval_for_bloc_and_slate(bloc_name="bloc_1", slate_name="slate_1")


def test_raises_valueerror_when_any_candidate_unset_negative_one(valid_config):
    cfg = BlocSlateConfig(**valid_config, n_voters=100, silent=True)

    # Unset 'A' (present in slate_1) for all blocs → sets -1.0
    cfg.unset_candidate_preferences("A")

    with pytest.raises(ValueError, match="have not been set"):
        cfg.get_preference_interval_for_bloc_and_slate(bloc_name="bloc_1", slate_name="slate_1")


def test_raises_valueerror_when_any_candidate_negative_not_unset(valid_config):
    cfg = BlocSlateConfig(**valid_config, n_voters=100, silent=True)
    cfg.preference_df.loc["bloc_1", "A"] = -0.2

    with pytest.raises(ValueError, match="invalid negative values"):
        cfg.get_preference_interval_for_bloc_and_slate(bloc_name="bloc_1", slate_name="slate_1")


def test_get_preference_interval_rejects_explicit_zero_support_by_default(valid_config):
    cfg = BlocSlateConfig(**valid_config, n_voters=100, silent=True)
    cfg.preference_df.loc["bloc_1", ["A", "B"]] = [0.0, 1.0]

    with pytest.raises(
        ValueError,
        match="Support values must be strictly positive for all candidates unless",
    ):
        cfg.get_preference_interval_for_bloc_and_slate(bloc_name="bloc_1", slate_name="slate_1")


def test_get_preference_interval_allows_explicit_zero_support_with_flag(valid_config):
    cfg = BlocSlateConfig(
        **valid_config,
        n_voters=100,
        silent=True,
        allow_zero_support_candidates=True,
    )
    cfg.preference_df.loc["bloc_1", ["A", "B"]] = [0.0, 1.0]

    out = cfg.get_preference_interval_for_bloc_and_slate(bloc_name="bloc_1", slate_name="slate_1")
    assert out.interval["A"] == 0.0
    assert out.interval["B"] == 1.0


def test_raises_valueerror_when_slate_values_do_not_sum_to_one(valid_config):
    cfg = BlocSlateConfig(**valid_config, n_voters=100, silent=True)

    cands = list(cfg.slate_to_candidates["slate_1"])  # e.g., ["A", "B"]
    row = cfg.preference_df.loc["bloc_1", cands]  # type: ignore[union-attr]
    cfg.preference_df.loc["bloc_1", cands] = row * 0.5  # now sums != 1

    with pytest.raises(ValueError, match=r"must\s+sum to 1, got"):
        cfg.get_preference_interval_for_bloc_and_slate(bloc_name="bloc_1", slate_name="slate_1")


def test_get_combined_preference_interval_by_bloc():
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
    cfg = BlocSlateConfig(
        n_voters=100,
        slate_to_candidates=slate_to_candidates,
        bloc_proportions=bloc_proportions,
        preference_mapping=preference_mapping,
        cohesion_mapping=cohesion_mapping,
        silent=True,
    )

    pref_interval_by_bloc = cfg.get_combined_preference_intervals_by_bloc()

    bloc_1_combined = {
        "A": 0.8 * 0.9,
        "B": 0.2 * 0.9,
        "X": 0.1 * 0.1,
        "Y": 0.9 * 0.1,
    }
    bloc_2_combined = {
        "A": 0.5 * 0.2,
        "B": 0.5 * 0.2,
        "X": 0.5 * 0.8,
        "Y": 0.5 * 0.8,
    }

    for bloc_name, expected in zip(["bloc_1", "bloc_2"], [bloc_1_combined, bloc_2_combined]):
        for key, value in expected.items():
            assert math.isclose(
                value,
                pref_interval_by_bloc[bloc_name].interval.get(key, 0.0),
                abs_tol=1e-8,
            )
