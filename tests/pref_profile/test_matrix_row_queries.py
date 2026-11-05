from __future__ import annotations

import numpy as np
import pytest

from votekit.pref_profile.profile_matrix import (
    ProfileVoterWeightData,
    RankMatrix,
    ScoreMatrix,
    WeightKind,
)


def test_score_queries_share_declared_candidate_order_and_missing_value_rules() -> None:
    data = ProfileVoterWeightData(np.array([1.0, 2.0, 1.0, 0.0]), None, WeightKind.FLOAT)
    matrix = ScoreMatrix(
        voter_weight_data=data,
        scores=np.array([[0.1, np.nan, 0.2], [np.nan, np.nan, np.nan], [0, 0, 0], [4, 99, 3]]),
        score_columns=("A", "outside", 1),
        candidates=(1, "A", "unused"),
    )

    assert list(matrix.get_scores(0)) == [1, "A"]
    assert matrix.get_scores(1) == {}
    assert matrix.get_scores(2) == {1: 0.0, "A": 0.0}
    assert matrix.get_scores(-1) == {1: 3.0, "A": 4.0}

    minimum, maximum = matrix.score_bounds_by_ballot()

    np.testing.assert_equal(minimum, [0.1, np.nan, 0.0, 3.0])
    np.testing.assert_equal(maximum, [0.2, np.nan, 0.0, 4.0])

    totals = matrix.sum_scores_by_ballot()

    np.testing.assert_allclose(totals, [0.3, 0.0, 0.0, 7.0])

    totals[:] = 100
    minimum[:] = 100

    assert matrix.get_scores(0) == {1: 0.2, "A": 0.1}


def test_score_totals_accumulate_float32_scores_in_float64() -> None:
    matrix = ScoreMatrix(
        voter_weight_data=ProfileVoterWeightData(np.ones(1), None, WeightKind.FLOAT),
        scores=np.array([[2**24, 1, 1, np.nan]], dtype=np.float32),
        score_columns=("A", "B", "C", "D"),
        candidates=("A", "B", "C", "D"),
    )

    totals = matrix.sum_scores_by_ballot()

    assert totals.dtype == np.float64
    np.testing.assert_equal(totals, [2**24 + 2])
    np.testing.assert_equal(matrix.scores, [[2**24, 1, 1, np.nan]])


@pytest.mark.parametrize("row_count", [0, 2])
def test_score_queries_support_zero_columns(row_count: int) -> None:
    matrix = ScoreMatrix(
        voter_weight_data=ProfileVoterWeightData(np.ones(row_count), None, WeightKind.FLOAT),
        scores=np.empty((row_count, 0)),
        score_columns=(),
        candidates=("unused",),
    )

    minimum, maximum = matrix.score_bounds_by_ballot()

    np.testing.assert_equal(minimum, np.full(row_count, np.nan))
    np.testing.assert_equal(maximum, minimum)
    np.testing.assert_equal(matrix.sum_scores_by_ballot(), np.zeros(row_count))

    if row_count:
        assert matrix.get_scores(0) == {}

    with pytest.raises(IndexError):
        matrix.get_scores(row_count)


def test_score_bounds_preserve_positive_infinity_as_a_score() -> None:
    matrix = ScoreMatrix(
        voter_weight_data=ProfileVoterWeightData(np.ones(1), None, WeightKind.FLOAT),
        scores=np.array([[np.inf]]),
        score_columns=("A",),
        candidates=("A",),
    )

    minimum, maximum = matrix.score_bounds_by_ballot()

    np.testing.assert_equal(minimum, [np.inf])
    np.testing.assert_equal(maximum, [np.inf])
    np.testing.assert_equal(matrix.sum_scores_by_ballot(), [np.inf])


def test_rank_queries_distinguish_padding_empty_sets_and_ties() -> None:
    matrix = RankMatrix(
        voter_weight_data=ProfileVoterWeightData(
            np.array([1.0, 1.0, 0.0, 1.0]), None, WeightKind.FLOAT
        ),
        rankings=np.array([[-1, -1, -1], [-2, -1, -1], [2, -1, -1], [0, -2, 1]]),
        cand_sets=(
            frozenset(),
            frozenset({"~"}),
            frozenset({"A"}),
            frozenset({"B"}),
            frozenset({"A", "B"}),
        ),
        ranking_columns=("Ranking_1", "Ranking_2", "Ranking_3"),
        candidates=("A", "B"),
    )

    np.testing.assert_equal(matrix.has_rankings(), [False, True, True, True])
    np.testing.assert_equal(matrix.has_ranked_candidates(), [False, False, True, True])
    np.testing.assert_equal(matrix.has_ties(), [False, False, True, False])

    assert matrix.get_ranking(0) == ()
    assert matrix.get_ranking(1) == (frozenset(),)
    assert matrix.get_ranking(2) == (frozenset({"A", "B"}),)
    assert matrix.get_ranking(-1) == (
        frozenset({"A"}),
        frozenset(),
        frozenset({"B"}),
    )

    with pytest.raises(IndexError):
        matrix.get_ranking(4)


@pytest.mark.parametrize("row_count", [0, 2])
def test_rank_queries_support_zero_columns(row_count: int) -> None:
    matrix = RankMatrix(
        voter_weight_data=ProfileVoterWeightData(np.ones(row_count), None, WeightKind.FLOAT),
        rankings=np.empty((row_count, 0), dtype=np.intp),
        cand_sets=(frozenset(), frozenset({"~"})),
        ranking_columns=(),
        candidates=(),
    )

    np.testing.assert_equal(matrix.has_rankings(), np.zeros(row_count, dtype=bool))
    np.testing.assert_equal(matrix.has_ranked_candidates(), np.zeros(row_count, dtype=bool))
    np.testing.assert_equal(matrix.has_ties(), np.zeros(row_count, dtype=bool))

    if row_count:
        assert matrix.get_ranking(0) == ()
