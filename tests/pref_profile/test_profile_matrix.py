from __future__ import annotations

from fractions import Fraction
from typing import cast

import numpy as np
import pandas as pd
import pytest

from votekit.ballot import RankBallot, ScoreBallot
from votekit.exceptions import ProfileError
from votekit.pref_profile import (
    CleanedRankProfile,
    CleanedScoreProfile,
    PreferenceProfile,
    RankProfile,
    ScoreProfile,
)
from votekit.pref_profile.profile_matrix import (
    ProfileMatrix,
    ProfileVoterWeightData,
    RankMatrix,
    ScoreMatrix,
    WeightKind,
)
from votekit.types import Candidate

# —————————————————— OWNERSHIP MODEL TESTS ———————————————————————


def test_constructor_isolates_weights_from_input_array():
    weights = np.array([1.0, 2.0])
    data = ProfileVoterWeightData(
        weights=weights,
        voter_sets=None,
        weight_kind=WeightKind.FLOAT,
    )

    weights[0] = 99.0

    np.testing.assert_array_equal(data.weights, [1.0, 2.0])


def test_constructor_isolates_voter_sets_from_input_set():
    voters = {"Alice"}
    voter_sets = np.empty(1, dtype=object)
    voter_sets[0] = voters
    data = ProfileVoterWeightData(
        weights=np.array([1.0]),
        voter_sets=voter_sets,
        weight_kind=WeightKind.FLOAT,
    )

    voters.add("Bob")

    assert data.voter_sets is not None
    assert data.voter_sets[0] == {"Alice"}
    assert data.voter_sets[0] == frozenset({"Alice"})
    assert isinstance(data.voter_sets[0], frozenset)


def test_constructor_isolates_voter_array_from_input_array():
    voter_sets = np.empty(1, dtype=object)
    voter_sets[0] = frozenset({"Alice"})

    data = ProfileVoterWeightData(
        weights=np.array([1.0]),
        voter_sets=voter_sets,
        weight_kind=WeightKind.FLOAT,
    )

    voter_sets[0] = frozenset({"Bob"})

    assert data.voter_sets is not None
    assert data.voter_sets[0] == frozenset({"Alice"})


def test_constructor_isolates_index_labels_from_input_array():
    labels = np.array([10, 42])
    data = ProfileVoterWeightData(
        weights=np.array([1.0, 2.0]),
        voter_sets=None,
        weight_kind=WeightKind.FLOAT,
        ballot_index=labels,
    )

    labels[0] = 99

    np.testing.assert_array_equal(data.ballot_index, [10, 42])


def _df_with_weights(weights) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "Ranking_1": [frozenset({"A"}), frozenset({"B"})],
            "Voter Set": [frozenset(), frozenset()],
            "Weight": weights,
        }
    )
    df.index.name = "Ballot Index"

    return df


def _profile_from_df(df: pd.DataFrame) -> RankProfile:
    return RankProfile(df=df, candidates=["A", "B"], max_ranking_length=1)


def test_profile_isolates_weights_from_input_dataframe():
    df = _df_with_weights([1.0, 2.0])
    profile = _profile_from_df(df)

    df.loc[0, "Weight"] = 99.0

    np.testing.assert_array_equal(
        profile._matrix.voter_weight_data.weights,
        [1.0, 2.0],
    )
    assert profile.to_df()["Weight"].to_list() == [1.0, 2.0]
    assert profile.total_ballot_wt == 3.0


@pytest.mark.parametrize("shape", [(), (1, 1), (1, 1, 1)], ids=["scalar", "2d", "3d"])
def test_shared_data_constructor_rejects_non_vector_weights(shape: tuple[int, ...]):
    with pytest.raises(ValueError, match=r"weights.*1D"):
        ProfileVoterWeightData(
            weights=np.ones(shape),
            voter_sets=None,
            weight_kind=WeightKind.FLOAT,
        )


@pytest.mark.parametrize(
    ("voter_sets", "ballot_index", "message"),
    [
        pytest.param(np.empty((), dtype=object), None, r"Voter Set.*1D", id="scalar-voters"),
        pytest.param(np.empty((1, 1), dtype=object), None, r"Voter Set.*1D", id="2d-voters"),
        pytest.param(None, np.array(1), r"Ballot Index.*1D", id="scalar-index"),
        pytest.param(None, np.array([[1]]), r"Ballot Index.*1D", id="2d-index"),
    ],
)
def test_shared_data_constructor_rejects_non_vector_optional_columns(
    voter_sets, ballot_index, message: str
):
    with pytest.raises(ValueError, match=message):
        ProfileVoterWeightData(
            weights=np.array([1.0]),
            voter_sets=voter_sets,
            weight_kind=WeightKind.FLOAT,
            ballot_index=ballot_index,
        )


@pytest.mark.parametrize(
    ("voter_sets", "ballot_index", "message"),
    [
        pytest.param(np.empty(0, dtype=object), None, r"Voter Set.*length", id="short-voters"),
        pytest.param(np.empty(2, dtype=object), None, r"Voter Set.*length", id="long-voters"),
        pytest.param(None, np.array([], dtype=int), r"Ballot Index.*length", id="short-index"),
        pytest.param(None, np.array([1, 2]), r"Ballot Index.*length", id="long-index"),
    ],
)
def test_shared_data_constructor_rejects_optional_column_length_mismatch(
    voter_sets, ballot_index, message: str
):
    with pytest.raises(ValueError, match=message):
        ProfileVoterWeightData(
            weights=np.array([1.0]),
            voter_sets=voter_sets,
            weight_kind=WeightKind.FLOAT,
            ballot_index=ballot_index,
        )


@pytest.mark.parametrize("dtype", [np.float32, np.int64, np.bool_, np.object_, np.complex128])
def test_shared_data_constructor_rejects_non_float64_dtype_for_float_weights(dtype):
    with pytest.raises(ValueError, match=r"float64.*FLOAT"):
        ProfileVoterWeightData(
            weights=np.array([1], dtype=dtype),
            voter_sets=None,
            weight_kind=WeightKind.FLOAT,
        )


@pytest.mark.parametrize("dtype", [np.float64, np.int64])
def test_shared_data_constructor_rejects_non_object_dtype_for_fraction_weights(dtype):
    with pytest.raises(ValueError, match=r"object.*FRACTION"):
        ProfileVoterWeightData(
            weights=np.array([1], dtype=dtype),
            voter_sets=None,
            weight_kind=WeightKind.FRACTION,
        )


@pytest.mark.parametrize("invalid_weight", [1, 1.0, "1", None], ids=["int", "float", "str", "none"])
def test_shared_data_constructor_rejects_non_fraction_entries(invalid_weight):
    with pytest.raises(ValueError, match="Fractions"):
        ProfileVoterWeightData(
            weights=np.array([Fraction(1, 3), invalid_weight], dtype=object),
            voter_sets=None,
            weight_kind=WeightKind.FRACTION,
        )


@pytest.mark.parametrize(
    ("invalid_weight", "message"),
    [(np.nan, "NaN"), (np.inf, "infinite"), (-np.inf, "infinite"), (-0.5, "negative")],
    ids=["nan", "positive-infinity", "negative-infinity", "negative"],
)
def test_shared_data_constructor_rejects_invalid_float_values(invalid_weight: float, message: str):
    with pytest.raises(ProfileError, match=message):
        ProfileVoterWeightData(
            weights=np.array([invalid_weight], dtype=np.float64),
            voter_sets=None,
            weight_kind=WeightKind.FLOAT,
        )


def test_shared_data_constructor_rejects_negative_fraction():
    with pytest.raises(ValueError, match="negative"):
        ProfileVoterWeightData(
            weights=np.array([Fraction(1, 3), Fraction(-1, 2)], dtype=object),
            voter_sets=None,
            weight_kind=WeightKind.FRACTION,
        )


def test_shared_data_constructor_rejects_unknown_weight_kind():
    with pytest.raises(ValueError, match="weight_kind"):
        ProfileVoterWeightData(
            weights=np.array([1.0]),
            voter_sets=None,
            weight_kind=cast(WeightKind, "FLOAT"),
        )


def test_shared_data_constructor_preserves_and_owns_nonnegative_fractions():
    weights = np.array([Fraction(0), Fraction(1, 3), Fraction(2)], dtype=object)
    data = ProfileVoterWeightData(
        weights=weights,
        voter_sets=None,
        weight_kind=WeightKind.FRACTION,
    )

    weights[1] = Fraction(99)

    assert data.weights.tolist() == [Fraction(0), Fraction(1, 3), Fraction(2)]
    assert all(isinstance(weight, Fraction) for weight in data.weights)
    assert data.has_fraction_weights


@pytest.mark.parametrize("weight_kind", [WeightKind.FLOAT, WeightKind.FRACTION])
@pytest.mark.parametrize("include_optional_columns", [False, True], ids=["absent", "empty"])
def test_shared_data_constructor_accepts_empty_arrays(
    weight_kind: WeightKind, include_optional_columns: bool
):
    dtype = np.float64 if weight_kind is WeightKind.FLOAT else np.object_
    voter_sets = np.empty(0, dtype=object) if include_optional_columns else None
    ballot_index = np.empty(0, dtype=int) if include_optional_columns else None
    data = ProfileVoterWeightData(
        weights=(
            np.empty(0, dtype=np.float64)
            if weight_kind is WeightKind.FLOAT
            else np.empty(0, dtype=object)
        ),
        voter_sets=voter_sets,
        weight_kind=weight_kind,
        ballot_index=ballot_index,
    )

    assert len(data) == 0
    assert data.weights.shape == (0,)
    assert data.weights.dtype == dtype
    assert data.weight_kind is weight_kind

    if include_optional_columns:
        assert data.voter_sets is not None
        assert data.voter_sets.shape == (0,)
        assert data.ballot_index is not None
        assert data.ballot_index.shape == (0,)
    else:
        assert data.voter_sets is None
        assert data.ballot_index is None


def _make_voter_weight_data(num_ballots: int = 1) -> ProfileVoterWeightData:
    return ProfileVoterWeightData(
        weights=np.ones(num_ballots, dtype=np.float64),
        voter_sets=None,
        weight_kind=WeightKind.FLOAT,
    )


def test_rank_matrix_constructor_owns_rankings():
    rankings = np.array([[0]], dtype=np.int32)
    matrix = RankMatrix(
        voter_weight_data=_make_voter_weight_data(),
        rankings=rankings,
        cand_sets=(frozenset(), frozenset({"~"}), frozenset({"A"})),
        ranking_columns=("Ranking_1",),
        candidates=("A",),
    )

    rankings[0, 0] = -2

    np.testing.assert_array_equal(matrix.rankings, [[0]])
    assert matrix.to_df()["Ranking_1"].tolist() == [frozenset({"A"})]


def test_score_matrix_constructor_owns_scores():
    scores = np.array([[1.0]])
    matrix = ScoreMatrix(
        voter_weight_data=_make_voter_weight_data(),
        scores=scores,
        score_columns=("A",),
        candidates=("A",),
    )

    scores[0, 0] = 99.0

    np.testing.assert_array_equal(matrix.scores, [[1.0]])
    assert matrix.to_df()["A"].tolist() == [1.0]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_score_matrix_constructor_canonicalizes_zeros_without_mutating_input(dtype):
    scores = np.array([[-0.0, 0.0, np.nan, 1.5]], dtype=dtype)
    candidates = ("A", "B", "C", "D")
    matrix = ScoreMatrix(_make_voter_weight_data(), scores, candidates, candidates)

    np.testing.assert_array_equal(matrix.scores, scores)
    assert not np.signbit(matrix.scores[0, :2]).any()
    assert np.signbit(scores[0, 0])
    assert matrix.scores.dtype == dtype
    assert not matrix.scores.flags.writeable
    assert not np.shares_memory(matrix.scores, scores)


def test_rank_matrix_constructor_freezes_candidate_sets():
    candidate_set = {"A"}
    matrix = RankMatrix(
        voter_weight_data=_make_voter_weight_data(),
        rankings=np.array([[0]]),
        cand_sets=cast(
            tuple[frozenset[Candidate], ...],
            (frozenset(), frozenset({"~"}), candidate_set),
        ),
        ranking_columns=("Ranking_1",),
        candidates=("A", "B"),
    )

    candidate_set.add("B")

    assert matrix.get_cand_set_from_id(0) == frozenset({"A"})
    assert isinstance(matrix.get_cand_set_from_id(0), frozenset)


@pytest.mark.parametrize("shape", [(), (1,), (1, 1, 1)], ids=["scalar", "vector", "3d"])
def test_rank_matrix_constructor_rejects_non_matrix_rankings(shape: tuple[int, ...]):
    with pytest.raises(ValueError, match=r"(?i)rankings"):
        RankMatrix(
            voter_weight_data=_make_voter_weight_data(),
            rankings=np.zeros(shape, dtype=np.int32),
            cand_sets=(frozenset(), frozenset({"~"}), frozenset({"A"})),
            ranking_columns=("Ranking_1",),
            candidates=("A",),
        )


@pytest.mark.parametrize("shape", [(), (1,), (1, 1, 1)], ids=["scalar", "vector", "3d"])
def test_score_matrix_constructor_rejects_non_matrix_scores(shape: tuple[int, ...]):
    with pytest.raises(ValueError, match=r"(?i)scores"):
        ScoreMatrix(
            voter_weight_data=_make_voter_weight_data(),
            scores=np.zeros(shape),
            score_columns=("A",),
            candidates=("A",),
        )


@pytest.mark.parametrize("num_rows", [0, 2])
def test_rank_matrix_constructor_rejects_row_count_mismatch(num_rows: int):
    with pytest.raises(ValueError, match=r"(?i)row|ballot|weight"):
        RankMatrix(
            voter_weight_data=_make_voter_weight_data(),
            rankings=np.zeros((num_rows, 1), dtype=np.int32),
            cand_sets=(frozenset(), frozenset({"~"}), frozenset({"A"})),
            ranking_columns=("Ranking_1",),
            candidates=("A",),
        )


@pytest.mark.parametrize("num_rows", [0, 2])
def test_score_matrix_constructor_rejects_row_count_mismatch(num_rows: int):
    with pytest.raises(ValueError, match=r"(?i)row|ballot|weight"):
        ScoreMatrix(
            voter_weight_data=_make_voter_weight_data(),
            scores=np.zeros((num_rows, 1)),
            score_columns=("A",),
            candidates=("A",),
        )


@pytest.mark.parametrize("num_columns", [0, 2])
def test_rank_matrix_constructor_rejects_column_count_mismatch(num_columns: int):
    with pytest.raises(ValueError, match=r"(?i)column"):
        RankMatrix(
            voter_weight_data=_make_voter_weight_data(),
            rankings=np.zeros((1, num_columns), dtype=np.int32),
            cand_sets=(frozenset(), frozenset({"~"}), frozenset({"A"})),
            ranking_columns=("Ranking_1",),
            candidates=("A",),
        )


@pytest.mark.parametrize("num_columns", [0, 2])
def test_score_matrix_constructor_rejects_column_count_mismatch(num_columns: int):
    with pytest.raises(ValueError, match=r"(?i)column"):
        ScoreMatrix(
            voter_weight_data=_make_voter_weight_data(),
            scores=np.zeros((1, num_columns)),
            score_columns=("A",),
            candidates=("A",),
        )


@pytest.mark.parametrize("dtype", [np.float64, np.bool_, np.object_])
def test_rank_matrix_constructor_rejects_non_integer_ids(dtype):
    with pytest.raises(ValueError, match=r"(?i)integer|dtype"):
        RankMatrix(
            voter_weight_data=_make_voter_weight_data(),
            rankings=np.array([[0]], dtype=dtype),
            cand_sets=(frozenset(), frozenset({"~"}), frozenset({"A"})),
            ranking_columns=("Ranking_1",),
            candidates=("A",),
        )


@pytest.mark.parametrize("ranking", [[-1, 0], [0, -1, 0], [-1, -2], [0, -1, -2]])
@pytest.mark.parametrize("weight", [0.0, 1.0])
def test_rank_matrix_rejects_positions_after_end_sentinel(
    ranking: list[int], weight: float
) -> None:
    with pytest.raises(ValueError, match="can only trail"):
        RankMatrix(
            voter_weight_data=ProfileVoterWeightData(np.array([weight]), None, WeightKind.FLOAT),
            rankings=np.array([ranking]),
            cand_sets=(frozenset(), frozenset({"~"}), frozenset({"A"})),
            ranking_columns=tuple(f"Ranking_{i + 1}" for i in range(len(ranking))),
            candidates=("A",),
        )


@pytest.mark.parametrize("candidate_set_id", [-3, 1], ids=["below-empty", "past-last-set"])
def test_rank_matrix_constructor_rejects_out_of_bounds_ids(candidate_set_id: int):
    with pytest.raises(ValueError, match=r"(?i)id"):
        RankMatrix(
            voter_weight_data=_make_voter_weight_data(),
            rankings=np.array([[candidate_set_id]], dtype=np.int32),
            cand_sets=(frozenset(), frozenset({"~"}), frozenset({"A"})),
            ranking_columns=("Ranking_1",),
            candidates=("A",),
        )


@pytest.mark.parametrize(
    "candidate_sets",
    [
        (),
        (frozenset(),),
        (frozenset({"~"}), frozenset()),
        (frozenset(), frozenset()),
    ],
    ids=["missing-both", "missing-tilde", "swapped", "duplicate-empty"],
)
def test_rank_matrix_constructor_rejects_invalid_reserved_sets(
    candidate_sets: tuple[frozenset[Candidate], ...],
):
    with pytest.raises(ValueError, match=r"(?i)reserved|empty|tilde|cand.*set"):
        RankMatrix(
            voter_weight_data=_make_voter_weight_data(0),
            rankings=np.empty((0, 1), dtype=np.int32),
            cand_sets=candidate_sets,
            ranking_columns=("Ranking_1",),
            candidates=("A",),
        )


def test_rank_matrix_constructor_rejects_undeclared_candidate():
    with pytest.raises(ValueError, match=r"(?i)candidate"):
        RankMatrix(
            voter_weight_data=_make_voter_weight_data(),
            rankings=np.array([[0]]),
            cand_sets=(frozenset(), frozenset({"~"}), frozenset({"B"})),
            ranking_columns=("Ranking_1",),
            candidates=("A",),
        )


@pytest.mark.parametrize(
    "score_columns", [("B",), ("A", "A")], ids=["undeclared-candidate", "duplicate-candidate"]
)
def test_score_matrix_constructor_rejects_invalid_column_mapping(
    score_columns: tuple[Candidate, ...],
):
    with pytest.raises(ValueError, match=r"(?i)candidate|column"):
        ScoreMatrix(
            voter_weight_data=_make_voter_weight_data(),
            scores=np.ones((1, len(score_columns))),
            score_columns=score_columns,
            candidates=("A",),
        )


def test_rank_matrix_constructor_preserves_reserved_ids_ties_and_mixed_candidates():
    matrix = RankMatrix(
        voter_weight_data=_make_voter_weight_data(),
        rankings=np.array([[-2, 0, -1]], dtype=np.int32),
        cand_sets=(frozenset(), frozenset({"~"}), frozenset({1, "1"})),
        ranking_columns=("Ranking_1", "Ranking_2", "Ranking_3"),
        candidates=(1, "1", "unranked"),
    )

    assert list(matrix.iter_candidate_sets()) == [
        (-2, frozenset()),
        (-1, frozenset({"~"})),
        (0, frozenset({1, "1"})),
    ]
    assert matrix.get_cand_set_from_id(-2) == frozenset()
    assert matrix.get_cand_set_from_id(-1) == frozenset({"~"})
    assert matrix.get_cand_set_from_id(0) == frozenset({1, "1"})
    assert matrix.max_candidates_ranked() == 2
    assert matrix.to_df()[["Ranking_1", "Ranking_2", "Ranking_3"]].iloc[0].tolist() == [
        frozenset(),
        frozenset({1, "1"}),
        frozenset({"~"}),
    ]


def test_score_matrix_constructor_preserves_column_order_and_unscored_values():
    matrix = ScoreMatrix(
        voter_weight_data=_make_voter_weight_data(),
        scores=np.array([[2.0, np.nan]]),
        score_columns=("1", 1),
        candidates=(1, "1", "unscored"),
    )

    exported = matrix.to_df()

    assert exported.columns.tolist() == ["1", 1, "Voter Set", "Weight"]
    assert exported["1"].iloc[0] == 2.0
    assert np.isnan(exported[1].iloc[0])
    assert matrix.candidates == (1, "1", "unscored")


@pytest.mark.parametrize("shape", [(0, 0), (0, 2), (2, 0)])
def test_rank_matrix_constructor_accepts_empty_dimensions(shape: tuple[int, int]):
    num_ballots, num_columns = shape
    matrix = RankMatrix(
        voter_weight_data=_make_voter_weight_data(num_ballots),
        rankings=np.empty(shape, dtype=np.int32),
        cand_sets=(frozenset(), frozenset({"~"})),
        ranking_columns=tuple(f"Ranking_{i + 1}" for i in range(num_columns)),
        candidates=(),
    )

    assert len(matrix) == num_ballots
    assert matrix.num_ranking_positions == num_columns
    assert matrix.max_candidates_ranked() == 0
    assert matrix.to_df().shape == (num_ballots, num_columns + 2)


@pytest.mark.parametrize("shape", [(0, 0), (0, 2), (2, 0)])
def test_score_matrix_constructor_accepts_empty_dimensions(shape: tuple[int, int]):
    num_ballots, num_columns = shape
    score_columns = ("A", "B")[:num_columns]
    matrix = ScoreMatrix(
        voter_weight_data=_make_voter_weight_data(num_ballots),
        scores=np.empty(shape),
        score_columns=score_columns,
        candidates=("A", "B"),
    )

    assert len(matrix) == num_ballots
    assert matrix.scores.shape == shape
    assert matrix.to_df().shape == (num_ballots, num_columns + 2)


@pytest.mark.parametrize(
    "ranking_columns",
    [
        ("Ranking_1extra",),
        ("Ranking_1\n",),
        (1,),
        ("Ranking_1", "Ranking_1"),
        ("Ranking_10", "Ranking_2"),
    ],
    ids=["trailing-text", "trailing-newline", "non-string", "duplicate", "out-of-order"],
)
def test_rank_matrix_constructor_rejects_invalid_ranking_columns(ranking_columns):
    with pytest.raises(ValueError, match="ranking_columns"):
        RankMatrix(
            voter_weight_data=_make_voter_weight_data(),
            rankings=np.zeros((1, len(ranking_columns)), dtype=np.int32),
            cand_sets=(frozenset(), frozenset({"~"}), frozenset({"A"})),
            ranking_columns=ranking_columns,
            candidates=("A",),
        )


def test_rank_matrix_constructor_owns_metadata_sequences():
    ranking_columns = ["Ranking_2", "Ranking_10"]
    candidates = ["A"]
    candidate_sets = [frozenset(), frozenset({"~"}), frozenset({"A"})]
    matrix = RankMatrix(
        voter_weight_data=_make_voter_weight_data(),
        rankings=np.array([[0, -1]]),
        cand_sets=cast(tuple[frozenset[Candidate], ...], candidate_sets),
        ranking_columns=cast(tuple[str, ...], ranking_columns),
        candidates=cast(tuple[Candidate, ...], candidates),
    )

    ranking_columns[0] = "Ranking_99"
    candidates[0] = "B"
    candidate_sets[2] = frozenset({"B"})

    assert matrix.ranking_columns == ("Ranking_2", "Ranking_10")
    assert matrix.candidates == ("A",)
    assert matrix.get_cand_set_from_id(0) == frozenset({"A"})


def test_rank_matrix_constructor_widens_ids_before_candidate_set_lookup():
    matrix = RankMatrix(
        voter_weight_data=_make_voter_weight_data(),
        rankings=np.array([[127]], dtype=np.int8),
        cand_sets=(frozenset(), frozenset({"~"}))
        + tuple(frozenset({candidate}) for candidate in range(128)),
        ranking_columns=("Ranking_1",),
        candidates=tuple(range(128)),
    )

    assert matrix.to_df()["Ranking_1"].tolist() == [frozenset({127})]


def test_rank_matrix_constructor_rejects_unsigned_id_before_narrowing():
    with pytest.raises(ValueError, match="IDs"):
        RankMatrix(
            voter_weight_data=_make_voter_weight_data(),
            rankings=np.array([[np.iinfo(np.uint64).max]], dtype=np.uint64),
            cand_sets=(frozenset(), frozenset({"~"})),
            ranking_columns=("Ranking_1",),
            candidates=(),
        )


@pytest.mark.parametrize("candidate_set", [frozenset(), frozenset({"~", "A"})])
def test_rank_matrix_constructor_rejects_reserved_values_in_real_sets(candidate_set):
    with pytest.raises(ValueError, match="Real candidate sets"):
        RankMatrix(
            voter_weight_data=_make_voter_weight_data(),
            rankings=np.array([[0]]),
            cand_sets=(frozenset(), frozenset({"~"}), candidate_set),
            ranking_columns=("Ranking_1",),
            candidates=("A",),
        )


@pytest.mark.parametrize("candidate_set", ["AB", frozenset({1.5}), frozenset({True})])
def test_rank_matrix_constructor_rejects_invalid_candidate_set_types(candidate_set):
    with pytest.raises(TypeError, match="Candidate sets"):
        RankMatrix(
            voter_weight_data=_make_voter_weight_data(),
            rankings=np.array([[0]]),
            cand_sets=(frozenset(), frozenset({"~"}), candidate_set),
            ranking_columns=("Ranking_1",),
            candidates=("A", "B", 1),
        )


def test_rank_matrix_constructor_protects_owned_rankings_from_accidental_writes():
    matrix = RankMatrix(
        voter_weight_data=_make_voter_weight_data(),
        rankings=np.array([[0]]),
        cand_sets=(frozenset(), frozenset({"~"}), frozenset({"A"})),
        ranking_columns=("Ranking_1",),
        candidates=("A",),
    )

    with pytest.raises(ValueError, match="read-only"):
        matrix.rankings[0, 0] = -2


def test_rank_matrix_constructor_retains_undeclared_candidates_on_zero_weight_ballots():
    data = ProfileVoterWeightData(
        weights=np.array([0.0, 1.0]), voter_sets=None, weight_kind=WeightKind.FLOAT
    )
    matrix = RankMatrix(
        voter_weight_data=data,
        rankings=np.array([[0], [1]]),
        cand_sets=(frozenset(), frozenset({"~"}), frozenset({"A"}), frozenset({"B"})),
        ranking_columns=("Ranking_1",),
        candidates=("B",),
    )

    assert matrix.candidates == ("B",)
    assert matrix.to_df()["Ranking_1"].tolist() == [frozenset({"A"}), frozenset({"B"})]


@pytest.mark.parametrize("dtype", [np.int64, np.bool_, np.object_, np.complex128])
def test_score_matrix_constructor_rejects_non_floating_scores(dtype):
    with pytest.raises(ValueError, match="floating-point dtype"):
        ScoreMatrix(_make_voter_weight_data(), np.ones((1, 1), dtype=dtype), ("A",), ("A",))


def test_score_matrix_constructor_rejects_non_array_scores():
    with pytest.raises(TypeError, match="scores must be a numpy array"):
        ScoreMatrix(_make_voter_weight_data(), cast(np.ndarray, [[1.0]]), ("A",), ("A",))


@pytest.mark.parametrize("candidate", [True, 1.5, ("A",)])
def test_score_matrix_constructor_rejects_unsupported_column_names(candidate):
    with pytest.raises(TypeError, match="string or integer"):
        ScoreMatrix(_make_voter_weight_data(), np.ones((1, 1)), (candidate,), ("A",))


@pytest.mark.parametrize("candidate", ["Weight", "Voter Set"])
def test_score_matrix_constructor_rejects_reserved_column_names(candidate: str):
    with pytest.raises(ValueError, match="reserved"):
        ScoreMatrix(_make_voter_weight_data(), np.ones((1, 1)), (candidate,), (candidate,))


def test_score_matrix_constructor_owns_metadata_and_to_df_returns_writable_scores():
    score_columns = ["A"]
    candidates = ["A", "B"]
    matrix = ScoreMatrix(
        _make_voter_weight_data(),
        np.array([[1.0]]),
        cast(tuple[Candidate, ...], score_columns),
        cast(tuple[Candidate, ...], candidates),
    )

    score_columns[0] = "B"
    candidates.clear()

    assert matrix.score_columns == ("A",)
    assert matrix.candidates == ("A", "B")

    with pytest.raises(ValueError, match="read-only"):
        matrix.scores[0, 0] = 99.0

    exported = matrix.to_df()
    exported.loc[0, "A"] = 99.0

    assert matrix.scores[0, 0] == 1.0
    assert matrix.to_df().loc[0, "A"] == 1.0


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_score_matrix_constructor_preserves_floating_score_values(dtype):
    scores = np.array([[np.nan, 0.0, 1.0, np.inf, 2.0]], dtype=dtype)
    candidates = ("A", "B", "C", "D", "E")
    matrix = ScoreMatrix(_make_voter_weight_data(), scores, candidates, candidates)

    assert matrix.scores.dtype == dtype
    np.testing.assert_array_equal(matrix.scores, scores)


def test_score_matrix_constructor_preserves_undeclared_candidates_on_zero_weight_ballots():
    data = ProfileVoterWeightData(np.array([1.0, 0.0]), None, WeightKind.FLOAT)
    scores = np.array([[1.0, np.nan], [np.nan, 2.0]])
    matrix = ScoreMatrix(data, scores, ("A", "B"), ("A",))

    assert matrix.candidates == ("A",)
    np.testing.assert_array_equal(matrix.to_df()[["A", "B"]].to_numpy(), scores)


@pytest.mark.parametrize("score", [0.0, np.nan])
def test_score_matrix_constructor_accepts_unused_undeclared_column(score: float):
    matrix = ScoreMatrix(_make_voter_weight_data(), np.array([[score]]), ("B",), ("A",))

    np.testing.assert_array_equal(matrix.scores, [[score]])


@pytest.mark.parametrize("score", [-1.0, -np.inf])
@pytest.mark.parametrize("weight", [0.0, 1.0])
def test_score_matrix_rejects_negative_scores_even_on_zero_weight_rows(
    score: float, weight: float
) -> None:
    with pytest.raises(ValueError, match="Scores must be non-negative"):
        ScoreMatrix(
            voter_weight_data=ProfileVoterWeightData(np.array([weight]), None, WeightKind.FLOAT),
            scores=np.array([[score]]),
            score_columns=("A",),
            candidates=("A",),
        )


def test_score_matrix_constructor_rejects_undeclared_negative_score():
    with pytest.raises(ValueError, match="non-negative"):
        ScoreMatrix(_make_voter_weight_data(), np.array([[-1.0]]), ("B",), ("A",))


def test_score_matrix_from_df_owns_score_values():
    df = pd.DataFrame({"A": [1.0], "Voter Set": [frozenset()], "Weight": [1.0]})
    matrix = ScoreMatrix.from_df(df, ("A",))
    df.loc[0, "A"] = 99.0

    assert matrix.scores[0, 0] == 1.0


@pytest.mark.parametrize("matrix_type", [RankMatrix, ScoreMatrix])
@pytest.mark.parametrize("labels", [None, [7, 7]])
@pytest.mark.parametrize("rows", [[-1, 0, 1], []])
@pytest.mark.parametrize("with_voters", [False, True])
def test_selected_rows_to_df_matches_full_dataframe_and_owns_storage(
    matrix_type: type[RankMatrix] | type[ScoreMatrix],
    labels: list[int] | None,
    rows: list[int],
    with_voters: bool,
) -> None:
    voter_sets = None

    if with_voters:
        voter_sets = np.empty(2, dtype=object)
        voter_sets[:] = [frozenset({"v"}), frozenset()]

    data = ProfileVoterWeightData(
        weights=np.array([Fraction(1, 3), Fraction(2, 3)], dtype=object),
        voter_sets=voter_sets,
        weight_kind=WeightKind.FRACTION,
        ballot_index=None if labels is None else np.array(labels),
    )
    matrix: RankMatrix | ScoreMatrix

    if matrix_type is RankMatrix:
        matrix = RankMatrix(
            data,
            np.array([[0], [-1]]),
            (frozenset(), frozenset({"~"}), frozenset({"A"})),
            ("Ranking_1",),
            ("A",),
        )
    else:
        matrix = ScoreMatrix(data, np.array([[0.0], [np.nan]]), ("A",), ("A",))

    before = matrix.to_df()
    selected = matrix.to_df(rows=np.array(rows, dtype=np.intp))

    pd.testing.assert_frame_equal(selected, before.iloc[rows])

    if rows:
        selected.iloc[0, -1] = 99

    pd.testing.assert_frame_equal(matrix.to_df(), before)


@pytest.mark.parametrize("weight", [2.0, Fraction(2, 3)])
@pytest.mark.parametrize(
    "profile_type, ballot_type, payload, column, replacement",
    [
        (RankProfile, RankBallot, {"ranking": [{"A"}]}, "Ranking_1", frozenset({"B"})),
        (ScoreProfile, ScoreBallot, {"scores": {"A": 3}}, "A", 99.0),
    ],
)
def test_profile_to_df_returns_detached_dataframes(
    profile_type, ballot_type, payload, column, replacement, weight
):
    ballot = ballot_type(**payload, weight=weight, voter_set={"voter"})
    source = profile_type(ballots=[ballot]).to_df()
    source["Weight"] = pd.Series([weight], dtype=object)
    source.index = pd.Index([37], name="Ballot Index")
    profile = profile_type(df=source, candidates=["A"], max_ranking_length=1)
    expected = profile.to_df()

    exported = profile.to_df()
    exported.at[37, "Weight"] = 9
    exported.at[37, "Voter Set"] = frozenset({"other voter"})
    exported.at[37, column] = replacement

    assert not np.shares_memory(
        exported.index.to_numpy(), profile._matrix.voter_weight_data.ballot_index
    )

    exported.index = pd.Index([99], name="Ballot Index")

    pd.testing.assert_frame_equal(profile.to_df(), expected)
    pd.testing.assert_frame_equal(profile._matrix.to_df(), expected)
    pd.testing.assert_frame_equal(profile.copy().to_df(), expected)
    assert profile.total_ballot_wt == weight
    assert profile.group_ballots().total_ballot_wt == weight
    assert profile.ballots == (ballot,)


def test_frozen_matrices_implement_profile_matrix_protocol():
    rank = RankProfile(ballots=[RankBallot(ranking=[{"A"}])] * 2)
    score = ScoreProfile(ballots=[ScoreBallot(scores={"A": 1})] * 2)
    matrices: tuple[ProfileMatrix, ...] = (rank._matrix, score._matrix)

    for matrix in matrices:
        assert isinstance(matrix, ProfileMatrix)
        assert len(matrix) == 2
        assert matrix.candidates == ("A",)
        assert matrix.to_df()["Weight"].tolist() == [1.0, 1.0]

        grouped = matrix.group_ballots()

        assert isinstance(grouped, ProfileMatrix)
        assert type(grouped) is type(matrix)
        assert len(grouped) == 1
        assert grouped.candidates == matrix.candidates
        assert grouped.to_df()["Weight"].tolist() == [2.0]
        assert len(matrix) == 2


def test_fraction_weight_type_error_summarizes_invalid_entries():
    with pytest.raises(ValueError, match=r"0 \(int\), 1 \(int\), 2 \(int\), and 2 more"):
        ProfileVoterWeightData(
            weights=np.array([0, 1, 2, 3, 4], dtype=object),
            voter_sets=None,
            weight_kind=WeightKind.FRACTION,
        )


@pytest.mark.parametrize(
    "profile_type", [RankProfile, ScoreProfile, CleanedRankProfile, CleanedScoreProfile]
)
def test_df_warns_and_delegates_to_to_df(
    profile_type: type[PreferenceProfile], monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = profile_type()
    snapshot = pd.DataFrame({"Weight": [2.0]})

    monkeypatch.setattr(profile_type, "to_df", lambda self: snapshot)

    with pytest.warns(
        DeprecationWarning,
        match=r"deprecated as of VoteKit 3\.6\.0.*next major release.*to_df\(\).*fresh",
    ) as recorded:
        exported = profile.df

    assert exported is snapshot
    assert len(recorded) == 1
    assert recorded[0].filename == __file__


def test_trusted_rank_constructor_shares_storage_and_derives_metadata():
    source = RankProfile(
        ballots=[RankBallot(ranking=[{"B"}, {"A"}], weight=Fraction(2, 3))],
        candidates=("B", "A", "unused"),
        max_ranking_length=3,
    )

    result = RankProfile._from_matrix(source._matrix, max_ranking_length=3)

    assert type(result) is RankProfile
    assert result._matrix is source._matrix
    assert result.candidates == ("B", "A", "unused")
    assert result.candidates_cast == ("A", "B")
    assert result.total_ballot_wt == Fraction(2, 3)
    assert result.num_ballots == 1
    assert result.max_ranking_length == 3
    assert result.max_candidates_ranked == 2

    with pytest.raises(AttributeError, match="frozen"):
        result.candidates = ()

    exported = result.to_df()
    exported["Weight"] = pd.Series([Fraction(99)])

    assert result.total_ballot_wt == source.total_ballot_wt == Fraction(2, 3)


@pytest.mark.parametrize("rankings", [np.array([[0], [1]]), np.array([[0], [0]])])
def test_rank_matrix_constructor_rejects_duplicate_candidate_sets(rankings):
    with pytest.raises(ValueError, match="Candidate sets must be unique"):
        RankMatrix(
            voter_weight_data=_make_voter_weight_data(2),
            rankings=rankings,
            cand_sets=(frozenset(), frozenset({"~"}), frozenset({"A"}), frozenset({"A"})),
            ranking_columns=("Ranking_1",),
            candidates=("A",),
        )


def test_rank_matrix_weight_replacement_owns_weights_and_checks_candidates() -> None:
    profile = RankProfile(ballots=[RankBallot(ranking=[{"A"}], weight=0)])
    matrix = profile._matrix
    weights = np.array([0.5])

    replacement = matrix.with_weights(weights)

    weights[0] = 99

    assert replacement.voter_weight_data.weights[0] == 0.5
    assert not replacement.voter_weight_data.weights.flags.writeable
    assert replacement.rankings is matrix.rankings
    assert replacement.voter_weight_data.voter_sets is matrix.voter_weight_data.voter_sets
    assert replacement.candidates == ("A",)
    assert matrix.candidates == ()
    assert matrix.voter_weight_data.weights[0] == 0

    restricted = RankMatrix(
        voter_weight_data=matrix.voter_weight_data,
        rankings=matrix.rankings,
        cand_sets=matrix.cand_sets,
        ranking_columns=matrix.ranking_columns,
        candidates=("B",),
    )

    with pytest.raises(ProfileError, match="not a subset"):
        restricted.with_weights(np.array([1.0]))


@pytest.mark.parametrize("weights", [np.array([1.0, 2.0]), np.array([[1.0]])])
def test_replacement_weights_reject_wrong_shape_or_count(weights):
    data = ProfileVoterWeightData(np.array([1.0]), None, WeightKind.FLOAT)

    with pytest.raises(ProfileError, match="same length as existing weights|1D"):
        data.with_weights(weights)


def test_replacement_weights_own_input_buffer():
    data = ProfileVoterWeightData(np.array([1.0]), None, WeightKind.FLOAT)
    weights = np.array([2.0])

    replaced = data.with_weights(weights)

    weights[0] = 99.0

    assert replaced.weights.tolist() == [2.0]
    assert data.weights.tolist() == [1.0]
    assert not replaced.weights.flags.writeable


@pytest.mark.parametrize("labels", [None, [8, 8, 20]])
def test_rank_matrix_apply_mask_selects_rows_and_keeps_labels(labels: list[int] | None) -> None:
    df = RankProfile(
        ballots=[RankBallot(ranking=[{c}], weight=w) for c, w in [("A", 1), ("B", 2), ("A", 3)]]
    ).to_df()
    if labels is not None:
        df.index = pd.Index(labels, name="Ballot Index")
    matrix = RankProfile(df=df, candidates=("A", "B"), max_ranking_length=1)._matrix
    mask = np.array([True, False, True])

    masked = matrix.apply_mask(mask)

    pd.testing.assert_frame_equal(masked.to_df(), matrix.to_df()[mask])


def test_rank_matrix_apply_mask_owns_rows_and_shares_metadata() -> None:
    profile = RankProfile(
        ballots=[RankBallot(ranking=[{"A"}], weight=1), RankBallot(ranking=[{"B"}], weight=2)]
    )
    matrix = profile._matrix

    masked = matrix.apply_mask(np.array([False, True]))

    assert masked.voter_weight_data.weights.tolist() == [2.0]
    assert masked.voter_weight_data.ballot_index is not None
    assert masked.voter_weight_data.ballot_index.tolist() == [1]
    assert not masked.rankings.flags.writeable
    assert not masked.voter_weight_data.weights.flags.writeable
    assert masked.cand_sets is matrix.cand_sets
    assert masked.candidates is matrix.candidates
    assert len(matrix) == 2


@pytest.mark.parametrize("mask", [np.ones(1, dtype=bool), np.ones((2, 1), dtype=bool)])
def test_rank_matrix_mask_rejects_wrong_shape(mask):
    matrix = RankProfile(ballots=[RankBallot(ranking=[{"A"}]), RankBallot(ranking=[{"B"}])])._matrix

    with pytest.raises(ValueError, match="mask must have shape"):
        matrix.apply_mask(mask)


@pytest.mark.parametrize("fraction_weights", [False, True])
def test_score_matrix_grouping_canonicalizes_nan_keys_but_preserves_source_scores(fraction_weights):
    nan_values = np.array([0x7FF8000000000001, 0xFFF8000000000002], dtype=np.uint64).view(
        np.float64
    )
    scores = np.array([[nan_values[0], -0.0], [1.0, 2.0], [nan_values[1], 0.0], [0.0, 0.0]])
    weights = np.array(
        [Fraction(n, 3) if fraction_weights else n / 4 for n in (1, 2, 3, 4)],
        dtype=object if fraction_weights else float,
    )
    voters = np.empty(4, dtype=object)
    voters[:] = [frozenset({"Alice"}), frozenset({"Bob"}), frozenset({"Carol"}), frozenset()]
    data = ProfileVoterWeightData(
        weights,
        voters,
        WeightKind.FRACTION if fraction_weights else WeightKind.FLOAT,
        ballot_index=np.array([10, 10, 20, 30]),
    )
    matrix = ScoreMatrix(data, scores, ("B", "A"), ("unused", "A", "B"))
    source_bits = matrix.scores.view(np.uint64).copy()

    grouped = matrix.group_ballots()

    np.testing.assert_array_equal(grouped.scores.view(np.uint64), source_bits[[0, 1, 3]])
    np.testing.assert_array_equal(matrix.scores.view(np.uint64), source_bits)
    assert grouped.voter_weight_data.weights.tolist() == [
        weights[0] + weights[2],
        weights[1],
        weights[3],
    ]
    assert grouped.voter_weight_data.voter_sets is not None
    assert grouped.voter_weight_data.voter_sets.tolist() == [
        frozenset({"Alice", "Carol"}),
        frozenset({"Bob"}),
        frozenset(),
    ]
    assert grouped.voter_weight_data.ballot_index is None
    assert grouped.score_columns == matrix.score_columns
    assert grouped.candidates == matrix.candidates
    assert not np.shares_memory(grouped.scores, matrix.scores)
    assert not np.shares_memory(grouped.voter_weight_data.weights, data.weights)
    assert not grouped.scores.flags.writeable


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.longdouble])
def test_score_matrix_grouping_preserves_distinct_representable_scores(dtype):
    one = dtype(1)
    next_value = np.nextafter(one, dtype(2))
    scores = np.array([[one], [next_value], [one]], dtype=dtype)
    matrix = ScoreMatrix(_make_voter_weight_data(3), scores, ("A",), ("A",))

    grouped = matrix.group_ballots()

    assert grouped.scores.dtype == scores.dtype
    np.testing.assert_array_equal(grouped.scores, scores[:2])
    np.testing.assert_array_equal(grouped.voter_weight_data.weights, [2.0, 1.0])


def test_trusted_score_constructor_derives_metadata_and_shares_immutable_storage():
    source = ScoreProfile(
        ballots=[ScoreBallot(scores={"B": 2, "A": 1}, weight=Fraction(2, 3))],
        candidates=("unused", "B", "A"),
    )

    result = ScoreProfile._from_matrix(source._matrix)

    assert type(result) is ScoreProfile
    assert result._matrix is source._matrix
    assert result.candidates == source.candidates
    assert result.candidates_cast == ("A", "B")
    assert result.total_ballot_wt == Fraction(2, 3)
    assert result.num_ballots == 1
    assert result.copy() == result

    with pytest.raises(AttributeError, match="frozen"):
        result.candidates = ()

    exported = result.to_df()
    exported.loc[0, "A"] = 99.0

    assert result.to_df().loc[0, "A"] == 1.0
