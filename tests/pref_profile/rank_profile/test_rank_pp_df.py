from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from votekit.ballot import RankBallot
from votekit.cleaning import remove_cand_rank_profile
from votekit.pref_profile import ProfileError, RankProfile

ballots_rankings = [
    RankBallot(ranking=({"A"}, {"B"}, {"C"}), weight=2),
    RankBallot(ranking=({"A", "B"}, frozenset(), {"D"}), voter_set={"Chris"}),
    RankBallot(),
    RankBallot(weight=0),
]
mixed_ballot_rankings = [
    RankBallot(ranking=({"A"}, {1}, {2}), weight=2),
    RankBallot(ranking=({"A", "B"}, frozenset(), {"C"}), voter_set={"Chris"}),
    RankBallot(ranking=(2, 1)),
    RankBallot(),
    RankBallot(weight=0),
]


def test_pp_df_rankings():
    pp = RankProfile(ballots=ballots_rankings)
    data = {
        "Ranking_1": [
            frozenset({"A"}),
            frozenset({"A", "B"}),
            frozenset("~"),
            frozenset("~"),
        ],
        "Ranking_2": [frozenset({"B"}), frozenset(), frozenset("~"), frozenset("~")],
        "Ranking_3": [
            frozenset({"C"}),
            frozenset({"D"}),
            frozenset("~"),
            frozenset("~"),
        ],
        "Voter Set": [set(), {"Chris"}, set(), set()],
        "Weight": [2.0, 1.0, 1.0, 0.0],
    }
    true_df = pd.DataFrame(data)
    true_df.index.name = "Ballot Index"
    assert pp.to_df().equals(true_df)


def test_pp_df_rankings_args():
    pp = RankProfile(
        ballots=ballots_rankings,
        candidates=["A", "B", "C", "D", "E"],
        max_ranking_length=4,
    )
    data = {
        "Ranking_1": [
            frozenset({"A"}),
            frozenset({"A", "B"}),
            frozenset("~"),
            frozenset("~"),
        ],
        "Ranking_2": [frozenset({"B"}), frozenset(), frozenset("~"), frozenset("~")],
        "Ranking_3": [
            frozenset({"C"}),
            frozenset({"D"}),
            frozenset("~"),
            frozenset("~"),
        ],
        "Ranking_4": [frozenset("~"), frozenset("~"), frozenset("~"), frozenset("~")],
        "Voter Set": [set(), {"Chris"}, set(), set()],
        "Weight": [2.0, 1.0, 1.0, 0.0],
    }
    true_df = pd.DataFrame(data)
    true_df.index.name = "Ballot Index"
    assert pp.to_df().equals(true_df)


def test_df_with_mixed_cand_types_as_ranking_values():
    rank_profile = RankProfile(
        ballots=mixed_ballot_rankings,
        candidates=["A", "B", "C", 1, 2, 3],
        max_ranking_length=4,
    )

    data = {
        "Ranking_1": [
            frozenset({"A"}),
            frozenset({"A", "B"}),
            frozenset({2}),
            frozenset("~"),
            frozenset("~"),
        ],
        "Ranking_2": [frozenset({1}), frozenset(), frozenset({1}), frozenset("~"), frozenset("~")],
        "Ranking_3": [
            frozenset({2}),
            frozenset({"C"}),
            frozenset("~"),
            frozenset("~"),
            frozenset("~"),
        ],
        "Ranking_4": [
            frozenset("~"),
            frozenset("~"),
            frozenset("~"),
            frozenset("~"),
            frozenset("~"),
        ],
        "Voter Set": [set(), {"Chris"}, set(), set(), set()],
        "Weight": [2.0, 1.0, 1.0, 1.0, 0.0],
    }
    true_df = pd.DataFrame(data)
    true_df.index.name = "Ballot Index"
    assert rank_profile.to_df().equals(true_df)


def test_internal_matrix_with_cand_ids_as_ranking_values():
    rank_profile = RankProfile(
        ballots=ballots_rankings,
        candidates=["A", "B", "C", "D", "E"],
        max_ranking_length=4,
    )
    matrix = rank_profile._matrix
    candidate_id_map = {
        cand_set: cand_set_id for cand_set_id, cand_set in matrix.iter_candidate_sets()
    }

    id_A = candidate_id_map[frozenset({"A"})]
    id_B = candidate_id_map[frozenset({"B"})]
    id_C = candidate_id_map[frozenset({"C"})]
    id_D = candidate_id_map[frozenset({"D"})]
    id_AB_tie = candidate_id_map[frozenset({"A", "B"})]
    id_tilde = candidate_id_map[frozenset({"~"})]
    id_empty = candidate_id_map[frozenset()]

    true_rankings = np.array(
        [
            [id_A, id_B, id_C, id_tilde],
            [id_AB_tie, id_empty, id_D, id_tilde],
            [id_tilde, id_tilde, id_tilde, id_tilde],
            [id_tilde, id_tilde, id_tilde, id_tilde],
        ]
    )
    assert np.array_equal(matrix.rankings, true_rankings)
    assert matrix.ranking_columns == ("Ranking_1", "Ranking_2", "Ranking_3", "Ranking_4")

    assert np.array_equal(matrix.voter_weight_data.weights, np.array([2.0, 1.0, 1.0, 0.0]))
    assert not matrix.voter_weight_data.has_fraction_weights

    # One ballot carries a voter set, so the vector is stored rather than left as None.
    assert matrix.voter_weight_data.voter_sets is not None
    assert list(matrix.voter_weight_data.voter_sets) == [set(), {"Chris"}, set(), set()]


def test_internal_matrix_voter_sets_none_when_all_empty():
    rank_profile = RankProfile(
        ballots=[RankBallot(ranking=[{"A"}, {"B"}], weight=1)],
        candidates=["A", "B"],
        max_ranking_length=2,
    )
    assert rank_profile._matrix.voter_weight_data.voter_sets is None
    assert rank_profile.to_df()["Voter Set"].tolist() == [frozenset()]


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


def _profile_from(df: pd.DataFrame) -> RankProfile:
    return RankProfile(df=df, candidates=["A", "B"], max_ranking_length=1)


def test_df_rejects_nan_float64_weight():
    with pytest.raises(ProfileError, match="cannot be NaN"):
        _profile_from(_df_with_weights(np.array([np.nan, 1.0], dtype=np.float64)))


def test_df_rejects_infinite_and_negative_weights():
    with pytest.raises(ProfileError, match="cannot be infinite"):
        _profile_from(_df_with_weights([np.inf, 1.0]))

    with pytest.raises(ProfileError, match="cannot be negative"):
        _profile_from(_df_with_weights([-1.0, 1.0]))

    with pytest.raises(ProfileError, match="cannot be NaN or infinity"):
        _profile_from(_df_with_weights(pd.Series([np.nan, Fraction(1)], dtype=object)))


@pytest.mark.parametrize("other", [np.float64(2), 2, 2.0])
def test_df_integral_reals_mixed_with_fraction_become_fractions(other):
    profile = _profile_from(_df_with_weights(pd.Series([other, Fraction(1, 3)], dtype=object)))

    assert profile._matrix.voter_weight_data.has_fraction_weights
    assert list(profile.to_df()["Weight"]) == [Fraction(2), Fraction(1, 3)]


def test_df_rejects_non_integral_float_mixed_with_fraction():
    with pytest.raises(ProfileError, match="Cannot mix Fraction weights with non-integral"):
        _profile_from(_df_with_weights(pd.Series([0.1, Fraction(1)], dtype=object)))


def test_df_object_column_without_fractions_becomes_float64():
    profile = _profile_from(_df_with_weights(pd.Series([np.float64(1.5), 2], dtype=object)))

    assert profile.to_df()["Weight"].dtype == np.float64
    assert list(profile.to_df()["Weight"]) == [1.5, 2.0]


def test_df_ballot_index_round_trips():
    df = _df_with_weights([1.0, 2.0])
    df.index = pd.Index([10, 42], name="Ballot Index")
    profile = _profile_from(df)

    assert list(profile.to_df().index) == [10, 42]
    assert list(remove_cand_rank_profile("A", profile).to_df().index) == [10, 42]
    assert list(RankProfile(ballots=profile.ballots).to_df().index) == [0, 1]
