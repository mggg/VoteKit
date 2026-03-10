"""Validation and conversion tests for bloc-slate config primitives."""

import math

import numpy as np
import pandas as pd
import pytest

from votekit import PreferenceInterval
from votekit.ballot_generator.bloc_slate_generator.config import (
    convert_bloc_proportion_map_to_series,
    convert_cohesion_map_to_cohesion_df,
    convert_preference_map_to_preference_df,
    typecheck_bloc_proportion_mapping,
    typecheck_cohesion_mapping,
    typecheck_preference,
)

#   typecheck_bloc_proportion_mapping
# =====================================


def test_typecheck_bloc_prop_accepts_series_float_str_index():
    s = pd.Series({"b1": 0.4, "b2": 0.6}, dtype=float)
    typecheck_bloc_proportion_mapping(s)


def test_typecheck_bloc_prop_series_rejects_non_str_index():
    s = pd.Series({1: 0.4, 2: 0.6}, dtype=float)
    with pytest.raises(TypeError, match="Bloc keys must be a 'str'"):
        typecheck_bloc_proportion_mapping(s)


def test_typecheck_bloc_prop_series_rejects_non_numeric_dtype():
    s = pd.Series({"b1": "0.4", "b2": "0.6"}, dtype=object)
    with pytest.raises(TypeError, match="must be numeric"):
        typecheck_bloc_proportion_mapping(s)


def test_typecheck_bloc_prop_series_rejects_nonfinite():
    s = pd.Series({"b1": np.nan, "b2": 1.0}, dtype=float)
    with pytest.raises(ValueError, match="contain non-finite values"):
        typecheck_bloc_proportion_mapping(s)


def test_typecheck_bloc_prop_mapping_valid():
    m = {"b1": 0.25, "b2": 0.75}
    typecheck_bloc_proportion_mapping(m)


def test_typecheck_bloc_prop_mapping_rejects_non_mapping():
    with pytest.raises(TypeError, match="must be a mapping"):
        typecheck_bloc_proportion_mapping(42)  # type: ignore[arg-type]


def test_typecheck_bloc_prop_mapping_rejects_non_str_key():
    with pytest.raises(TypeError, match="must be a 'str'"):
        typecheck_bloc_proportion_mapping({1: 0.2})  # type: ignore[dict-item]


def test_typecheck_bloc_prop_mapping_rejects_nonfinite_or_bool_values():
    with pytest.raises(TypeError, match="must be a finite real"):
        typecheck_bloc_proportion_mapping({"b": True})
    with pytest.raises(TypeError, match="must be a finite real"):
        typecheck_bloc_proportion_mapping({"b": float("inf")})


# =========================================
#   convert_bloc_proportion_map_to_series
# =========================================


def test_convert_bloc_prop_series_passthrough_checks_and_cast():
    s = pd.Series({"b1": 0.4, "b2": 0.6}, dtype=np.float64)
    out = convert_bloc_proportion_map_to_series(s)
    pd.testing.assert_series_equal(out, s.astype(float))
    dup = pd.Series([0.5, 0.5], index=["b1", "b1"], dtype=float)
    with pytest.raises(ValueError, match=r"\(blocs\) contains duplicates."):
        convert_bloc_proportion_map_to_series(dup)
    with pytest.raises(ValueError, match="must be non-negative"):
        convert_bloc_proportion_map_to_series(pd.Series({"b1": -0.1, "b2": 1.1}, dtype=float))


def test_convert_bloc_prop_mapping_requires_sum_one_and_nonneg():
    with pytest.raises(ValueError, match="should sum to 1"):
        convert_bloc_proportion_map_to_series({"b1": 0.2, "b2": 0.2})
    with pytest.raises(ValueError, match="non-negative"):
        convert_bloc_proportion_map_to_series({"b1": -0.1, "b2": 1.1})


def test_convert_bloc_prop_mapping_ok_and_normalizes_fp():
    s = convert_bloc_proportion_map_to_series({"b1": 0.300000004, "b2": 0.699999996})
    assert s.dtype == float
    assert set(s.index) == {"b1", "b2"}
    assert math.isclose(float(s.sum()), 1.0, abs_tol=1e-12)


def test_convert_bloc_prop_series_casts_int_to_float():
    s_int = pd.Series({"b1": 1, "b2": 0}, dtype="int64")
    out = convert_bloc_proportion_map_to_series(s_int)
    assert out.dtype == float
    pd.testing.assert_series_equal(out, s_int.astype(float))


def test_convert_bloc_prop_series_casts_float32_to_float():
    s_f32 = pd.Series({"b1": 0.2, "b2": 0.8}, dtype="float32")
    out = convert_bloc_proportion_map_to_series(s_f32)
    assert out.dtype == float
    pd.testing.assert_series_equal(out, s_f32.astype(float))


def test_convert_bloc_prop_series_rejects_nonfinite_values():
    s_nan = pd.Series({"b1": np.nan, "b2": 1.0}, dtype=float)
    with pytest.raises(ValueError, match="non-finite"):
        convert_bloc_proportion_map_to_series(s_nan)
    s_inf = pd.Series({"b1": np.inf, "b2": 0.0}, dtype=float)
    with pytest.raises(ValueError, match="non-finite"):
        convert_bloc_proportion_map_to_series(s_inf)


def test_convert_bloc_prop_series_rejects_values_greater_than_one():
    s = pd.Series({"b1": 1.2, "b2": 0.8}, dtype=float)
    with pytest.raises(ValueError, match="greater than 1"):
        convert_bloc_proportion_map_to_series(s)


# ==============================
#   typecheck_cohesion_mapping
# ==============================


def test_typecheck_cohesion_df_accepts_float_str_labels():
    df = pd.DataFrame({"s1": {"b1": 0.7, "b2": 0.2}, "s2": {"b1": 0.3, "b2": 0.8}}).astype(float)
    typecheck_cohesion_mapping(df)  # no raise


def test_typecheck_cohesion_df_rejects_non_str_labels_or_nonfloat_dtype():
    df1 = pd.DataFrame({"s1": {1: 0.5}, "s2": {2: 0.5}}).astype(float)
    with pytest.raises(TypeError, match=r"\(slates\) must be a 'str'"):
        typecheck_cohesion_mapping(df1)
    df2 = pd.DataFrame({1: {"b1": 0.5}, 2: {"b2": 0.5}}).astype(float)
    with pytest.raises(TypeError, match=r"\(blocs\) must be a 'str'"):
        typecheck_cohesion_mapping(df2)
    df3 = pd.DataFrame({"s1": {"b1": 1, "b2": 0}, "s2": {"b1": 0, "b2": 1}})
    with pytest.raises(TypeError, match="must have float dtypes in every column"):
        typecheck_cohesion_mapping(df3)
    df4 = pd.DataFrame({"s1": {"b1": np.nan}})
    with pytest.raises(ValueError, match="contains non-finite values"):
        typecheck_cohesion_mapping(df4)


def test_typecheck_cohesion_mapping_nested_dict_happy_and_errors():
    good = {"b1": {"s1": 0.5, "s2": 0.5}, "b2": {"s1": 0.25, "s2": 0.75}}
    typecheck_cohesion_mapping(good)  # no raise
    with pytest.raises(TypeError, match="must be a mapping"):
        typecheck_cohesion_mapping(123)  # not a mapping  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="must be a mapping"):
        typecheck_cohesion_mapping(
            {"b1": ["s1", 0.5]}  # type: ignore[dict-item]
        )  # inner not mapping
    with pytest.raises(TypeError, match="Bloc keys must be a 'str'"):
        typecheck_cohesion_mapping({1: {"s1": 0.5}})  # type: ignore[dict-item]  # bloc key not str
    with pytest.raises(TypeError, match="slate keys must be a 'str'"):
        typecheck_cohesion_mapping({"b1": {1: 0.5}})  # type: ignore[dict-item]  # slate key not str
    with pytest.raises(TypeError, match="must be a finite real"):
        typecheck_cohesion_mapping({"b1": {"s1": float("inf")}})  # non-finite


# -----------------------------
# convert_cohesion_map_to_cohesion_df
# -----------------------------


def test_convert_cohesion_df_passthrough_and_duplicates_check():
    df = pd.DataFrame({"s1": {"b1": 0.7, "b2": 0.3}, "s2": {"b1": 0.3, "b2": 0.7}}).astype(float)
    out = convert_cohesion_map_to_cohesion_df(df)
    pd.testing.assert_frame_equal(out, df)  # copy returns equal content

    # duplicate index
    dup_idx = pd.DataFrame(
        [[0.1, 0.9], [0.2, 0.8]], index=["b1", "b1"], columns=["s1", "s2"]
    ).astype(float)
    with pytest.raises(ValueError):
        convert_cohesion_map_to_cohesion_df(dup_idx)
    # duplicate columns
    dup_cols = pd.DataFrame(
        [[0.1, 0.9], [0.2, 0.8]], index=["b1", "b2"], columns=["s1", "s1"]
    ).astype(float)
    with pytest.raises(ValueError):
        convert_cohesion_map_to_cohesion_df(dup_cols)


def test_convert_cohesion_mapping_to_df_shape_and_fill():
    m = {
        "b1": {"s1": 0.7, "s2": 0.3},
        "b2": {"s1": 0.2},  # s2 missing → should fill -1.0
    }
    out = convert_cohesion_map_to_cohesion_df(m)
    assert list(out.index) == ["b1", "b2"]
    assert set(out.columns) == {"s1", "s2"}
    assert out.loc["b2", "s2"] == -1.0
    assert out.dtypes.eq(float).all()  # type: ignore[union-attr]


# -----------------------------
# typecheck_preference
# -----------------------------


def test_typecheck_preference_df_accepts_numeric_and_str_labels():
    df = pd.DataFrame(
        {"A": {"bloc1": 0.6, "bloc2": 0.3}, "B": {"bloc1": 0.4, "bloc2": 0.7}}
    )  # numeric dtypes inferred
    typecheck_preference(df)  # no raise


def test_typecheck_preference_df_rejects_non_str_labels_or_non_numeric():
    # non-str index
    df1 = pd.DataFrame({"A": {1: 0.5}, "B": {2: 0.5}})
    with pytest.raises(TypeError):
        typecheck_preference(df1)
    # non-str columns
    df2 = pd.DataFrame({1: {"bloc1": 0.5}})
    with pytest.raises(TypeError):
        typecheck_preference(df2)
    # non-numeric dtype
    df3 = pd.DataFrame({"A": {"bloc1": "x"}})
    with pytest.raises(TypeError):
        typecheck_preference(df3)
    # non-finite
    df4 = pd.DataFrame({"A": {"bloc1": np.nan}})
    with pytest.raises(ValueError):
        typecheck_preference(df4)


def test_typecheck_preference_mapping_happy_and_errors():
    good = {
        "bloc1": {"slate1": {"A": 1.0}, "slate2": {"B": 1.0}},
        "bloc2": {"slate1": {"A": 0.2, "B": 0.8}, "slate2": {"B": 1.0}},
    }
    typecheck_preference(good)  # no raise

    with pytest.raises(TypeError):
        typecheck_preference(3.14)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        typecheck_preference({1: {"slate1": {"A": 1.0}}})  # type: ignore[dict-item]  # bloc not str
    with pytest.raises(TypeError):
        typecheck_preference(
            {"bloc": [("slate1", {"A": 1.0})]}  # type: ignore[dict-item]
        )  # inner not mapping
    with pytest.raises(TypeError):
        typecheck_preference({"bloc": {1: {"A": 1.0}}})  # type: ignore[dict-item]  # slate not str
    with pytest.raises(TypeError):
        typecheck_preference(
            {"bloc": {"slate": {1: 1.0}}}  # type: ignore[dict-item]
        )  # candidate name not str
    with pytest.raises(TypeError):
        typecheck_preference({"bloc": {"slate": {"A": float("inf")}}})  # non-finite


def test_typecheck_preference_accepts_real_PreferenceInterval():
    pi = PreferenceInterval.from_dirichlet(candidates=["A", "B"], alpha=1.0)
    pref = {"bloc1": {"slate1": pi}}
    typecheck_preference(pref)


def test_typecheck_preference_accepts_real_preferenceinterval():
    pi = PreferenceInterval({"A": 0.6, "B": 0.4})
    prefs = {"bloc1": {"slate1": pi}}
    typecheck_preference(prefs)  # should not raise


def test_typecheck_preference_PI_non_str_candidate_name_raises():
    pi = PreferenceInterval({1: 0.5, 2: 0.5})  # keys are ints, not str
    prefs = {"bloc1": {"slate1": pi}}
    with pytest.raises(TypeError, match=r"candidate names must be a 'str'"):
        typecheck_preference(prefs)


def test_typecheck_preference_PI_non_finite_score_raises_via_inf_division():
    pi = PreferenceInterval({"A": np.inf, "B": 1.0})
    prefs = {"bloc1": {"slate1": pi}}
    with pytest.raises(TypeError, match=r"score must be a finite real"):
        typecheck_preference(prefs)


def test_typecheck_preference_else_branch_unexpected_item_type():
    prefs = {"bloc1": {"slate1": ["A", 1.0]}}  # list is invalid here
    with pytest.raises(
        TypeError, match=r"expected Mapping\[str, float\|int\] or PreferenceInterval"
    ):
        typecheck_preference(prefs)  # type: ignore[dict-item]


# ===========================================
#   convert_preference_map_to_preference_df
# ===========================================


def test_convert_preference_df_passthrough_and_duplicate_checks():
    df = pd.DataFrame({"A": {"bloc1": 1.0}, "B": {"bloc1": 0.0}})
    out = convert_preference_map_to_preference_df(df)
    # function returns the same object (no copy) for DF input
    assert out is df

    dup_idx = pd.DataFrame([[1.0], [0.0]], index=["bloc1", "bloc1"], columns=["A"])
    with pytest.raises(ValueError):
        convert_preference_map_to_preference_df(dup_idx)

    dup_cols = pd.DataFrame([[1.0, 0.0]], index=["bloc1"], columns=["A", "A"])
    with pytest.raises(ValueError):
        convert_preference_map_to_preference_df(dup_cols)


def test_convert_preference_mapping_unions_candidates_and_fills_unset_with_minus_one():
    pref = {
        "bloc1": {"slate1": {"A": 0.6, "B": 0.4}, "slate2": {"C": 1.0}},
        "bloc2": {"slate1": {"B": 1.0}, "slate2": {"C": 1.0}},
    }
    df = convert_preference_map_to_preference_df(pref)
    assert set(df.index) == {"bloc1", "bloc2"}
    assert set(df.columns) == {"A", "B", "C"}
    assert df.loc["bloc2", "A"] == -1.0
    assert pd.api.types.is_numeric_dtype(df["A"].dtype)
