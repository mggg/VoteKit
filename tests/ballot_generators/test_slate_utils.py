from typing import cast

import numpy as np
import pytest

from votekit.ballot_generator.bloc_slate_generator.config import BlocSlateConfig
from votekit.ballot_generator.bloc_slate_generator.slate_utils import (
    _construct_slate_to_candidate_ordering_arrays,
    _fast_sample_without_replacement,
    _lexicographic_symbol_tuple_iterator,
)
from votekit.pref_interval import PreferenceInterval


@pytest.mark.parametrize("small_weight", [1e-6, 1e-300, 1e-320])
def test_weighted_rankings_preserve_order_for_tiny_weights_and_zero_draw(small_weight):
    class FixedDraws:
        def standard_exponential(self, size):
            assert size == (3, 3)
            return np.array([[1.0, 1.0, 4.0], [1.0, 4.0, 1.0], [1.0, 0.0, 4.0]])

    rankings = _fast_sample_without_replacement(
        np.array([1.0, small_weight, 2 * small_weight]),
        3,
        numpy_rng=cast(np.random.Generator, FixedDraws()),
    )

    np.testing.assert_array_equal(rankings, [[0, 1, 2], [0, 2, 1], [1, 0, 2]])


@pytest.mark.parametrize("scale", [1e-300, 1e300])
def test_weighted_rankings_are_invariant_to_weight_scale(scale):
    weights = np.array([0.2, 0.3, 0.5])
    reference_rng = np.random.default_rng(42)
    sampling_rng = np.random.default_rng(42)
    expected = _fast_sample_without_replacement(weights, 1000, numpy_rng=reference_rng)

    rankings = _fast_sample_without_replacement(weights * scale, 1000, numpy_rng=sampling_rng)

    np.testing.assert_array_equal(rankings, expected)
    assert sampling_rng.bit_generator.state == reference_rng.bit_generator.state


def test_slate_candidate_orderings_shuffle_zero_support_tails():
    config = BlocSlateConfig(
        n_voters=1000,
        allow_zero_support_candidates=True,
        slate_to_candidates={"A": [7, "b", "c"]},
        bloc_proportions={"bloc": 1.0},
        preference_mapping={
            "bloc": {"A": PreferenceInterval({7: 1.0, "b": 0, "c": 0}, allow_zero_support=True)}
        },
        cohesion_mapping={"bloc": {"A": 1.0}},
    )

    rankings = _construct_slate_to_candidate_ordering_arrays(
        config, "bloc", 1000, numpy_rng=np.random.default_rng(42)
    )["A"]

    assert np.all(rankings[:, 0] == 7)
    assert all(set(row) == {"b", "c"} for row in rankings[:, 1:])
    assert np.mean(rankings[:, 1] == "b") == pytest.approx(0.5, abs=0.05)


def test_lexicographic_symbol_tuple_iterator_empty():
    result = list(_lexicographic_symbol_tuple_iterator([]))
    assert result == []


def test_lexicographic_symbol_tuple_iterator_single():
    result = list(_lexicographic_symbol_tuple_iterator(["A"]))
    assert result == [("A",)]


def test_lexicographic_symbol_tuple_iterator_duplicates():
    result = list(_lexicographic_symbol_tuple_iterator(["A", "A", "B"]))
    expected = [("A", "A", "B"), ("A", "B", "A"), ("B", "A", "A")]
    assert result == expected


def test_lexicographic_symbol_tuple_iterator_multiple():
    result = list(_lexicographic_symbol_tuple_iterator(["C", "A", "B"]))
    expected = [
        ("A", "B", "C"),
        ("A", "C", "B"),
        ("B", "A", "C"),
        ("B", "C", "A"),
        ("C", "A", "B"),
        ("C", "B", "A"),
    ]
    assert result == expected


def test_lexicographic_symbol_tuple_iterator_with_more_duplicates():
    result = list(_lexicographic_symbol_tuple_iterator(["A", "B", "A", "C"]))
    expected = [
        ("A", "A", "B", "C"),
        ("A", "A", "C", "B"),
        ("A", "B", "A", "C"),
        ("A", "B", "C", "A"),
        ("A", "C", "A", "B"),
        ("A", "C", "B", "A"),
        ("B", "A", "A", "C"),
        ("B", "A", "C", "A"),
        ("B", "C", "A", "A"),
        ("C", "A", "A", "B"),
        ("C", "A", "B", "A"),
        ("C", "B", "A", "A"),
    ]
    assert result == expected
