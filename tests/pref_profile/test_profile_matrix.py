from fractions import Fraction
from typing import cast

import numpy as np
import pandas as pd
import pytest

from votekit.exceptions import ProfileError
from votekit.pref_profile import RankProfile
from votekit.pref_profile.profile_matrix import (
    RankMatrix,
    ScoreMatrix,
    WeightKind,
    _ProfileVoterWeightData,
)
from votekit.types import Candidate

# —————————————————— OWNERSHIP MODEL TESTS ———————————————————————


def test_constructor_isolates_weights_from_input_array():
    weights = np.array([1.0, 2.0])
    data = _ProfileVoterWeightData(
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
    data = _ProfileVoterWeightData(
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

    data = _ProfileVoterWeightData(
        weights=np.array([1.0]),
        voter_sets=voter_sets,
        weight_kind=WeightKind.FLOAT,
    )

    voter_sets[0] = frozenset({"Bob"})

    assert data.voter_sets is not None
    assert data.voter_sets[0] == frozenset({"Alice"})


def test_constructor_isolates_index_labels_from_input_array():
    labels = np.array([10, 42])
    data = _ProfileVoterWeightData(
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
    return cast(RankProfile, RankProfile(df=df, candidates=["A", "B"], max_ranking_length=1))


def test_profile_isolates_weights_from_input_dataframe():
    df = _df_with_weights([1.0, 2.0])
    profile = _profile_from_df(df)

    df.loc[0, "Weight"] = 99.0

    np.testing.assert_array_equal(
        profile._matrix.voter_weight_data.weights,
        [1.0, 2.0],
    )
    assert profile.df["Weight"].to_list() == [1.0, 2.0]
    assert profile.total_ballot_wt == 3.0


@pytest.mark.parametrize("shape", [(), (1, 1), (1, 1, 1)], ids=["scalar", "2d", "3d"])
def test_shared_data_constructor_rejects_non_vector_weights(shape: tuple[int, ...]):
    with pytest.raises(ValueError, match=r"weights.*1D"):
        _ProfileVoterWeightData(
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
        _ProfileVoterWeightData(
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
        _ProfileVoterWeightData(
            weights=np.array([1.0]),
            voter_sets=voter_sets,
            weight_kind=WeightKind.FLOAT,
            ballot_index=ballot_index,
        )


@pytest.mark.parametrize("dtype", [np.float32, np.int64, np.bool_, np.object_, np.complex128])
def test_shared_data_constructor_rejects_non_float64_dtype_for_float_weights(dtype):
    with pytest.raises(ValueError, match=r"float64.*FLOAT"):
        _ProfileVoterWeightData(
            weights=np.array([1], dtype=dtype),
            voter_sets=None,
            weight_kind=WeightKind.FLOAT,
        )


@pytest.mark.parametrize("dtype", [np.float64, np.int64])
def test_shared_data_constructor_rejects_non_object_dtype_for_fraction_weights(dtype):
    with pytest.raises(ValueError, match=r"object.*FRACTION"):
        _ProfileVoterWeightData(
            weights=np.array([1], dtype=dtype),
            voter_sets=None,
            weight_kind=WeightKind.FRACTION,
        )


@pytest.mark.parametrize("invalid_weight", [1, 1.0, "1", None], ids=["int", "float", "str", "none"])
def test_shared_data_constructor_rejects_non_fraction_entries(invalid_weight):
    with pytest.raises(ValueError, match="Fractions"):
        _ProfileVoterWeightData(
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
        _ProfileVoterWeightData(
            weights=np.array([invalid_weight], dtype=np.float64),
            voter_sets=None,
            weight_kind=WeightKind.FLOAT,
        )


def test_shared_data_constructor_rejects_negative_fraction():
    with pytest.raises(ValueError, match="negative"):
        _ProfileVoterWeightData(
            weights=np.array([Fraction(1, 3), Fraction(-1, 2)], dtype=object),
            voter_sets=None,
            weight_kind=WeightKind.FRACTION,
        )


def test_shared_data_constructor_rejects_unknown_weight_kind():
    with pytest.raises(ValueError, match="weight_kind"):
        _ProfileVoterWeightData(
            weights=np.array([1.0]),
            voter_sets=None,
            weight_kind=cast(WeightKind, "FLOAT"),
        )


def test_shared_data_constructor_preserves_and_owns_nonnegative_fractions():
    weights = np.array([Fraction(0), Fraction(1, 3), Fraction(2)], dtype=object)
    data = _ProfileVoterWeightData(
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
    data = _ProfileVoterWeightData(
        weights=np.empty(0, dtype=dtype),
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


def _make_voter_weight_data(num_ballots: int = 1) -> _ProfileVoterWeightData:
    return _ProfileVoterWeightData(
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
        rankings=np.array([[-2, -1, 0]], dtype=np.int32),
        cand_sets=(frozenset(), frozenset({"~"}), frozenset({1, "1"})),
        ranking_columns=("Ranking_1", "Ranking_2", "Ranking_3"),
        candidates=(1, "1", "unranked"),
    )

    assert matrix.get_cand_set_from_id(-2) == frozenset()
    assert matrix.get_cand_set_from_id(-1) == frozenset({"~"})
    assert matrix.get_cand_set_from_id(0) == frozenset({1, "1"})
    assert matrix.max_candidates_ranked() == 2
    assert matrix.to_df()[["Ranking_1", "Ranking_2", "Ranking_3"]].iloc[0].tolist() == [
        frozenset(),
        frozenset({"~"}),
        frozenset({1, "1"}),
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
    assert matrix.max_ranking_length == num_columns
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
    data = _ProfileVoterWeightData(
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


def test_score_matrix_constructor_owns_metadata_and_exports_writable_scores():
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
    scores = np.array([[np.nan, 0.0, -1.0, np.inf, -np.inf]], dtype=dtype)
    candidates = ("A", "B", "C", "D", "E")
    matrix = ScoreMatrix(_make_voter_weight_data(), scores, candidates, candidates)

    assert matrix.scores.dtype == dtype
    np.testing.assert_array_equal(matrix.scores, scores)


def test_score_matrix_constructor_preserves_undeclared_candidates_on_zero_weight_ballots():
    data = _ProfileVoterWeightData(np.array([1.0, 0.0]), None, WeightKind.FLOAT)
    scores = np.array([[1.0, np.nan], [np.nan, 2.0]])
    matrix = ScoreMatrix(data, scores, ("A", "B"), ("A",))

    assert matrix.candidates == ("A",)
    np.testing.assert_array_equal(matrix.to_df()[["A", "B"]].to_numpy(), scores)


@pytest.mark.parametrize("score", [0.0, np.nan])
def test_score_matrix_constructor_accepts_unused_undeclared_column(score: float):
    matrix = ScoreMatrix(_make_voter_weight_data(), np.array([[score]]), ("B",), ("A",))

    np.testing.assert_array_equal(matrix.scores, [[score]])


def test_score_matrix_constructor_rejects_undeclared_negative_score():
    with pytest.raises(ValueError, match="outside the declared candidates"):
        ScoreMatrix(_make_voter_weight_data(), np.array([[-1.0]]), ("B",), ("A",))


def test_score_matrix_from_df_owns_score_values():
    df = pd.DataFrame({"A": [1.0], "Voter Set": [frozenset()], "Weight": [1.0]})
    matrix = ScoreMatrix.from_df(df, ("A",))
    df.loc[0, "A"] = 99.0

    assert matrix.scores[0, 0] == 1.0
