"""Collection wrapper tests for bloc-slate config."""

import warnings

import numpy as np
import pandas as pd
import pytest

from votekit.ballot_generator.bloc_slate_generator.config import (
    BlocProportions,
    ConfigurationWarning,
    SlateCandMap,
)

#   _CandListProxy
# ==================


# Quick fixture to isolate a SlateCandMap with a parent that has a .candidates property
@pytest.fixture
def slate_map():
    class Parent:
        def __init__(self):
            self._map = None
            self._current_preference_df_slate_cand_mapping = None

        @property
        def candidates(self):
            if self._map is None:
                return []
            return [c for v in self._map._data.values() for c in v]

        def _update_preference_and_cohesion_slates(self):
            pass

    parent = Parent()
    sm = SlateCandMap(parent, {"s1": ["A", "B"], "s2": ["C"]})  # type: ignore[arg-type]
    parent._map = sm
    yield sm


def test_len_and_getitem(slate_map):
    p = slate_map["s1"]
    assert len(p) == 2
    assert p[0] == "A"
    assert p[1] == "B"
    assert p[-1] == "B"
    assert p[:] == ["A", "B"]


def test_setitem_allows_change_and_blocks_cross_slate_clash(slate_map):
    p = slate_map["s1"]
    p[0] = "X"
    assert slate_map["s1"] == ["X", "B"]

    with pytest.raises(ValueError, match="already exist in slate"):
        p[1] = "C"


def test_delitem_updates_via_setitem(slate_map):
    p = slate_map["s1"]
    del p[1]
    assert slate_map["s1"] == ["A"]


def test_insert_type_checks_and_dedup_within_slate(slate_map):
    p = slate_map["s1"]
    with pytest.raises(TypeError, match="Index must be an 'int'"):
        p.insert("1", "Z")
    with pytest.raises(TypeError, match="candidates must be a 'str'"):
        p.insert(1, 5)

    before = slate_map["s1"]
    p.insert(0, "A")
    assert slate_map["s1"] == before

    with pytest.raises(ValueError, match="already exist in slate"):
        p.insert(0, "C")


def test_insert_negative_index_behaves_like_list(slate_map):
    p = slate_map["s1"]
    p.insert(-100, "Z")
    assert slate_map["s1"][0] == "Z"


def test_extend_uses_insert_and_is_non_atomic_on_first_conflict(slate_map):
    p = slate_map["s1"]

    p.extend(["D", "E"])
    assert slate_map["s1"] == ["A", "B", "D", "E"]

    with pytest.raises(ValueError, match="already exist in slate"):
        p.extend(["F", "C", "G"])

    assert slate_map["s1"] == ["A", "B", "D", "E"]
    assert "C" not in slate_map["s1"]


def test_iadd_returns_self_and_modifies(slate_map):
    p = slate_map["s1"]
    ret = p.__iadd__(["Z"])
    assert ret is p
    assert "Z" in slate_map["s1"]


def test_append_dedup_and_cross_slate_validation(slate_map):
    p = slate_map["s1"]
    p.append("A")
    assert slate_map["s1"] == ["A", "B"]
    with pytest.raises(ValueError, match="already exist in slate"):
        p.append("C")


def test_sort_routes_through_owner_and_sorts(slate_map):
    p = slate_map["s1"]
    p.extend(["Z", "D"])
    p.sort()
    assert slate_map["s1"] == sorted(slate_map["s1"])


def test_eq_semantics(slate_map):
    p = slate_map["s1"]
    assert p == ["A", "B"]
    assert not (p == ["B", "A"])
    assert not (p == ["A"])
    assert not (p == 42)


# ================
#   SlateCandMap
# ================


@pytest.fixture
def parent_and_map():
    """
    Minimal parent that exposes `.candidates` the way SlateCandMap expects.
    We keep a strong ref to the parent for the whole test to avoid weakref death.
    """

    class Parent:
        def __init__(self):
            self._map = None
            self._current_preference_df_slate_cand_mapping = None

        @property
        def candidates(self):
            if self._map is None:
                return []
            return [c for v in self._map._data.values() for c in v]

        def _update_preference_and_cohesion_slates(self):
            pass

    parent = Parent()
    smap = SlateCandMap(parent, {"s1": ["A", "B"], "s2": ["C"]})  # type: ignore[arg-type]
    parent._map = smap
    yield parent, smap


@pytest.fixture
def sm(parent_and_map):
    # convenience: just the map
    return parent_and_map[1]


def test_init_accepts_mapping_and_coerces_to_str(parent_and_map):
    parent = parent_and_map[0]
    smap = SlateCandMap(parent, {"x": [1, "2"]})  # type: ignore[arg-type]
    assert smap.to_dict() == {"x": ["1", "2"]}


def test_init_rejects_empty_candidate_list(parent_and_map):
    parent = parent_and_map[0]
    with pytest.raises(ValueError):
        SlateCandMap(parent, {"x": []})


def test_init_raises_attributeerror_if_init_has_no_items(parent_and_map):
    parent = parent_and_map[0]
    with pytest.raises(AttributeError, match=r"does not implement the '\.items\(\)' method"):
        SlateCandMap(
            parent,
            [("x", ["A"])],  # type: ignore[arg-type]
        )  # list has no .items()


def test_len_iter_to_dict(sm):
    assert len(sm) == 2
    assert set(iter(sm)) == {"s1", "s2"}
    d = sm.to_dict()
    assert d == {"s1": ["A", "B"], "s2": ["C"]}
    d["s1"].append("Z")
    assert sm.to_dict()["s1"] == ["A", "B"]


def test_getitem_returns_proxy_and_reads(sm):
    p = sm["s1"]
    assert len(p) == 2
    assert p[0] == "A"


def test_setitem_rollback_on_parent_keyerror_existing_slate():
    """
    If parent._update_preference_and_cohesion_slates() raises KeyError while
    replacing an existing slate's candidate list, SlateCandMap should restore
    the old value (rollback) and re-raise with the friendly message.
    """

    class Parent:
        def __init__(self):
            self._map = None
            self._current_preference_df_slate_cand_mapping = None

        @property
        def candidates(self):
            if self._map is None:
                return []
            return [c for v in self._map._data.values() for c in v]

        def _update_preference_and_cohesion_slates(self):
            # Force the rollback path
            raise KeyError("Preference mapping columns do not match candidates")

    parent = Parent()
    sm = SlateCandMap(parent, {"s1": ["A", "B"], "s2": ["C"]})  # type: ignore[arg-type]
    parent._map = sm

    before = sm.to_dict()["s1"].copy()
    with pytest.raises(KeyError, match="You may have tried to modify the candidate list directly"):
        sm["s1"] = ["Z", "B"]  # non-clashing change; parent blows up

    # Rolled back to original
    assert sm.to_dict()["s1"] == before


def test_setitem_rollback_on_parent_keyerror_new_slate_removes_key():
    """
    If adding a brand-new slate triggers parent KeyError, SlateCandMap should
    delete the newly inserted key (rollback = None branch).
    """

    class Parent:
        def __init__(self):
            self._map = None
            self._current_preference_df_slate_cand_mapping = None

        @property
        def candidates(self):
            if self._map is None:
                return []
            return [c for v in self._map._data.values() for c in v]

        def _update_preference_and_cohesion_slates(self):
            # Force the rollback path
            raise KeyError("Preference mapping columns do not match candidates")

    parent = Parent()
    sm = SlateCandMap(parent, {"s1": ["A", "B"]})  # type: ignore[arg-type]
    parent._map = sm

    with pytest.raises(KeyError, match="You may have tried to modify the candidate list directly"):
        sm["s3"] = ["P", "Q"]  # add new slate; parent blows up

    # New key should have been removed
    assert "s3" not in set(iter(sm))
    assert "s3" not in sm.to_dict()


def test_setitem_replaces_slate_and_coerces_to_str(sm):
    sm["s1"] = ["X", 2]
    assert sm.to_dict()["s1"] == ["X", "2"]


def test_setitem_rejects_non_str_key(sm):
    with pytest.raises(TypeError):
        sm[1] = ["X"]


def test_setitem_rejects_non_sequence_value(sm):
    with pytest.raises(TypeError):
        sm["s1"] = "ABC"  # strings are excluded explicitly


def test_setitem_rejects_empty_sequence(sm):
    with pytest.raises(ValueError):
        sm["s1"] = []


def test_setitem_blocks_cross_slate_clash(sm):
    # "C" exists in s2; assigning it to s1 should raise
    with pytest.raises(ValueError, match=r"already exist in slates"):
        sm["s1"] = ["C"]


def test_delitem_removes_slate(sm):
    del sm["s2"]
    assert "s2" not in set(iter(sm))


def test_update_with_mapping_routes_through_setitem(sm):
    sm.update({"s1": ["X"], "s3": ["Y"]})
    assert sm == {"s1": ["X"], "s2": ["C"], "s3": ["Y"]}


def test_update_with_kwargs_and_pairs_and_right_away_validation(sm):
    sm["s1"] = ["A"]
    with pytest.raises(ValueError):
        sm.update([("s1", ["D"]), ("s2", ["D"])])  # s1 becomes D; s2->D conflicts
    assert sm["s1"] == ["D"]


def test_or_right_bias_and_leaves_original_unchanged(sm, parent_and_map):
    parent, orig = parent_and_map
    left_copy = SlateCandMap(parent, orig.to_dict())
    res = sm | {"s1": ["X"], "s3": ["Y"]}  # right side overwrites s1
    assert res == {"s1": ["X"], "s2": ["C"], "s3": ["Y"]}
    assert sm == left_copy


def test_ror_left_seed_then_self_overwrites(parent_and_map):
    _, sm = parent_and_map
    res = {
        "s1": ["X"],
        "s3": ["Y"],
    } | sm
    assert res["s1"] == ["A", "B"]
    assert res["s3"] == ["Y"]
    assert res["s2"] == ["C"]


def test_ior_in_place(sm):
    sm |= {"s3": ["Y"]}
    assert sm["s3"] == ["Y"]


def test_eq_true_against_equal_dict(sm):
    assert sm == {"s1": ["A", "B"], "s2": ["C"]}


def test_eq_false_when_value_differs_or_missing_keys(sm):
    assert not (sm == {"s1": ["A"], "s2": ["C"]})
    assert not (sm == {"s1": ["A", "B"]})

    bigger = {"s1": ["A", "B"], "s2": ["C"], "extra": ["Z"]}
    assert not (sm == bigger)


def test_eq_non_mapping_is_false(sm):
    assert not (sm == [("s1", ["A", "B"]), ("s2", ["C"])])  # not a MutableMapping


def test_copy_returns_plain_dict_copy_independent(sm):
    d = sm.copy()
    assert isinstance(d, dict)
    assert d == {"s1": ["A", "B"], "s2": ["C"]}
    d["s1"].append("Z")
    assert sm["s1"] == ["A", "B"]


# ===================
#   BlocProportions
# ===================


def _make_parent(silent: bool):
    class Parent:
        def __init__(self, silent):
            self.silent = silent

    return Parent(silent)


@pytest.fixture
def bp_silent_false():
    """BlocProportions with a live parent (silent=False)."""
    parent = _make_parent(silent=False)
    bp = BlocProportions(parent, {"b1": 0.6, "b2": 0.4})
    # keep parent alive for the entire test via the generator frame
    yield bp


@pytest.fixture
def bp_silent_true():
    """BlocProportions with a live parent (silent=True)."""
    parent = _make_parent(silent=True)
    bp = BlocProportions(parent, {"b1": 0.6, "b2": 0.4})
    yield bp


def test_init_from_dict_valid_normalized_to_series_float():
    parent = _make_parent(False)
    bp = BlocProportions(parent, {"b1": 0.3000000004, "b2": 0.699999996})
    s = bp.to_series()
    assert s.dtype == float
    assert set(s.index) == {"b1", "b2"}
    assert pytest.approx(float(s.sum()), rel=0, abs=1e-12) == 1.0


def test_init_from_dict_sum_not_one_raises():
    parent = _make_parent(False)
    with pytest.raises(ValueError, match="sum to 1"):
        BlocProportions(parent, {"b1": 0.2, "b2": 0.2})


def test_init_from_series_casts_to_float_and_accepts_numeric():
    parent = _make_parent(False)
    ser = pd.Series({"b1": 1, "b2": 0}, dtype="int64")
    bp = BlocProportions(parent, ser)
    out = bp.to_series()
    assert out.dtype == float
    pd.testing.assert_series_equal(out, ser.astype(float))


def test_init_from_series_rejects_nonfinite():
    parent = _make_parent(False)
    ser = pd.Series({"b1": np.nan, "b2": 1.0}, dtype=float)
    with pytest.raises(ValueError, match="non-finite"):
        BlocProportions(parent, ser)


def test_len_iter_getitem(bp_silent_false):
    bp = bp_silent_false
    assert len(bp) == 2
    assert set(iter(bp)) == {"b1", "b2"}
    assert bp["b1"] == pytest.approx(0.6)


def test_to_series_and_copy_are_independent(bp_silent_false):
    bp = bp_silent_false
    s = bp.to_series()
    d = bp.copy()
    assert isinstance(d, dict)
    assert s["b2"] == pytest.approx(0.4)
    d["b1"] = 999.0
    assert bp["b1"] != 999.0  # copy is independent


def test_setitem_key_type_and_value_finiteness_errors():
    parent = _make_parent(silent=False)
    with pytest.raises(TypeError, match="Bloc keys must be a 'str'"):
        _ = BlocProportions(parent, {1: 0.5, "b2": 0.5})  # type: ignore[index]
    with pytest.raises(TypeError, match="finite real"):
        _ = BlocProportions(parent, {"b1": float("inf"), "b2": 0.0})
    with pytest.raises(TypeError, match="finite real"):
        _ = BlocProportions(parent, {"b1": True, "b2": 1.0})


def test_setitem_negative_triggers_validate_error():
    parent = _make_parent(silent=False)
    with pytest.raises(ValueError, match="non-negative"):
        _ = BlocProportions(parent, {"b1": -0.1, "b2": 1.1})


def test_delitem_updates_and_warns_on_sum_change_when_silent_false(bp_silent_false):
    bp = bp_silent_false
    with pytest.warns(ConfigurationWarning, match=r"sum to 0\.6"):
        del bp["b2"]  # total now 0.6 -> warn
    assert list(bp) == ["b1"]


def test_delitem_no_warning_when_silent_true(bp_silent_true):
    bp = bp_silent_true
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        del bp["b2"]
        assert not any(isinstance(x.message, ConfigurationWarning) for x in w)
    assert list(bp) == ["b1"]


def test_repr_contains_keys(bp_silent_false):
    r = repr(bp_silent_false)
    assert "b1" in r and "b2" in r
