import re

import numpy as np
import pandas as pd
import pytest

from votekit.ballot import RankBallot
from votekit.pref_profile import RankProfile, ScoreProfile
from votekit.utils import search_profile_for_rank_pattern
from votekit.utils.search_utils import (
    _boolean_matrix,
    _compare_candidate_pair_ranks,
    _compare_query_ranks,
    _get_candidate_pair_min_distance,
    _get_next_true_ballot_with_cand,
    _shift_idx_to_next_ballot,
    _validate_max_cand_pair_dist,
    _validate_ranking_query,
)

# --- ranking_query searchs -----------------------------------


def test_search_profile_for_rank_pattern():
    profile = RankProfile(
        ballots=(
            RankBallot(ranking=[{"A", "B"}, {"C"}, 1], weight=1.0),
            RankBallot(ranking=[{"B", "D"}, {"A"}, {"C"}], weight=1.0),
            RankBallot(ranking=[1, 2, "B", "C"], weight=1.0),
            RankBallot(ranking=[2, "C", "B"], weight=1.0),
        ),
        candidates=("A", "B", "C", "D", 1, 2),
        max_ranking_length=4,
    )
    profile_match = RankProfile(
        ballots=(
            RankBallot(ranking=[{"A", "B"}, {"C"}, 1], weight=1.0),
            RankBallot(ranking=[{"B", "D"}, {"A"}, {"C"}], weight=1.0),
            RankBallot(ranking=[1, 2, "B", "C"], weight=1.0),
        ),
        candidates=("A", "B", "C", "D", 1, 2),
        max_ranking_length=4,
    )
    assert (
        search_profile_for_rank_pattern(profile, ranking_query=["B", "C"])
        .reset_index(drop=True)
        .equals(profile_match.df.reset_index(drop=True))
    )


def test_search_for_set_rank_pattern():
    profile = RankProfile(
        ballots=(
            RankBallot(ranking=[{"A", "B"}, {"C"}, 1], weight=1.0),
            RankBallot(ranking=[{"B", "D"}, {"A"}, {"C"}], weight=1.0),
            RankBallot(ranking=[1, 2, "B", "C"], weight=1.0),
            RankBallot(ranking=[2, "C", "B"], weight=1.0),
        ),
        candidates=("A", "B", "C", "D", 1, 2),
        max_ranking_length=4,
    )
    profile_match = RankProfile(
        ballots=(RankBallot(ranking=[1, 2, "B", "C"], weight=1.0),),
        candidates=("A", "B", "C", "D", 1, 2),
        max_ranking_length=4,
    )
    assert (
        search_profile_for_rank_pattern(profile, ranking_query=[{"B"}, "C"])
        .reset_index(drop=True)
        .equals(profile_match.df.reset_index(drop=True))
    )


def test_search_for_tuple_rank_pattern():
    profile = RankProfile(
        ballots=(
            RankBallot(ranking=[{"A", "B"}, {"C"}, 1], weight=1.0),
            RankBallot(ranking=[{"B", "D"}, {"A"}, {"C"}], weight=1.0),
            RankBallot(ranking=[1, 2, "B", "C"], weight=1.0),
            RankBallot(ranking=[2, "C", "B"], weight=1.0),
        ),
        candidates=("A", "B", "C", "D", 1, 2),
        max_ranking_length=4,
    )
    profile_match = RankProfile(
        ballots=(
            RankBallot(ranking=[{"A", "B"}, {"C"}, 1], weight=1.0),
            RankBallot(ranking=[{"B", "D"}, {"A"}, {"C"}], weight=1.0),
            RankBallot(ranking=[1, 2, "B", "C"], weight=1.0),
            RankBallot(ranking=[2, "C", "B"], weight=1.0),
        ),
        candidates=("A", "B", "C", "D", 1, 2),
        max_ranking_length=4,
    )
    assert (
        search_profile_for_rank_pattern(profile, ranking_query=[("A", 2), "C"])
        .reset_index(drop=True)
        .equals(profile_match.df.reset_index(drop=True))
    )


def test_search_for_singleton_rank_pattern():
    profile = RankProfile(
        ballots=(
            RankBallot(ranking=[{"A", "B"}, {"C"}, 1], weight=1.0),
            RankBallot(ranking=[{"B", "D"}, {"A"}, {"C"}], weight=1.0),
            RankBallot(ranking=[1, 2, "B", "C"], weight=1.0),
            RankBallot(ranking=[2, "C", "B"], weight=1.0),
        ),
        candidates=("A", "B", "C", "D", 1, 2),
        max_ranking_length=4,
    )
    profile_match = RankProfile(
        ballots=(
            RankBallot(ranking=[{"A", "B"}, {"C"}, 1], weight=1.0),
            RankBallot(ranking=[{"B", "D"}, {"A"}, {"C"}], weight=1.0),
        ),
        candidates=("A", "B", "C", "D", 1, 2),
        max_ranking_length=4,
    )
    assert (
        search_profile_for_rank_pattern(profile, ranking_query=["A"])
        .reset_index(drop=True)
        .equals(profile_match.df.reset_index(drop=True))
    )


def test_search_for_multiple_elements_rank_pattern():
    profile = RankProfile(
        ballots=(
            RankBallot(ranking=[{"A", "B"}, {"C"}, 1], weight=1.0),
            RankBallot(ranking=[{"B", "D"}, {"A"}, {"C"}], weight=1.0),
            RankBallot(ranking=[1, 2, "B", "C"], weight=1.0),
            RankBallot(ranking=[2, "C", "B"], weight=1.0),
        ),
        candidates=("A", "B", "C", "D", 1, 2),
        max_ranking_length=4,
    )
    profile_match = RankProfile(
        ballots=(RankBallot(ranking=[{"A", "B"}, {"C"}, 1], weight=1.0),),
        candidates=("A", "B", "C", "D", 1, 2),
        max_ranking_length=4,
    )
    assert (
        search_profile_for_rank_pattern(profile, ranking_query=["A", "C", 1])
        .reset_index(drop=True)
        .equals(profile_match.df.reset_index(drop=True))
    )


def test_search_returns_empty_df():
    profile = RankProfile(
        ballots=(
            RankBallot(ranking=[{"A", "B"}, {"C"}, 1], weight=1.0),
            RankBallot(ranking=[{"B", "D"}, {"A"}, {"C"}], weight=1.0),
            RankBallot(ranking=[1, 2, "B", "C"], weight=1.0),
            RankBallot(ranking=[2, "C", "B"], weight=1.0),
        ),
        candidates=("A", "B", "C", "D", 1, 2),
        max_ranking_length=4,
    )
    empty_profile = RankProfile(
        ballots=(),
        candidates=("A", "B", "C", "D", 1, 2),
        max_ranking_length=4,
    )

    pd.testing.assert_frame_equal(
        search_profile_for_rank_pattern(profile, ranking_query=["B", "D"]).reset_index(drop=True),
        empty_profile.df.reset_index(drop=True),
        check_dtype=False,
    )


def test_search_with_ranking_query_with_duplicate_candidate_rankings():
    profile = RankProfile(
        ballots=(
            RankBallot(ranking=["A", 1, "B", "A", "C", "A"], weight=1.0),
            RankBallot(ranking=[1, 2, {"A", "B"}, "D", "C", "A"], weight=1.0),
            RankBallot(ranking=[1, 2, "A", "D", "C", "B"], weight=1.0),
            RankBallot(ranking=[1, 2, "B", "D", "C"], weight=1.0),
        ),
        candidates=("A", "B", "C", "D", 1, 2),
        max_ranking_length=6,
    )

    true_profile = RankProfile(
        ballots=(
            RankBallot(ranking=["A", 1, "B", "A", "C", "A"], weight=1.0),
            RankBallot(ranking=[1, 2, {"A", "B"}, "D", "C", "A"], weight=1.0),
        ),
        candidates=("A", "B", "C", "D", 1, 2),
        max_ranking_length=6,
    )

    pd.testing.assert_frame_equal(
        search_profile_for_rank_pattern(profile, ranking_query=["B", "A"]).reset_index(drop=True),
        true_profile.df.reset_index(drop=True),
        check_dtype=False,
    )


def test_search_with_ranking_query_with_duplicate_candidate_rankings_for_multiple_rank_elements():
    profile = RankProfile(
        ballots=(
            RankBallot(ranking=["B", "A", "C", "B"], weight=1.0),
            RankBallot(ranking=["B", "A", "B", "C"], weight=1.0),
        ),
        candidates=("A", "B", "C", "D"),
        max_ranking_length=4,
    )

    true_profile = RankProfile(
        ballots=(RankBallot(ranking=["B", "A", "B", "C"], weight=1.0),),
        candidates=("A", "B", "C", "D"),
        max_ranking_length=4,
    )

    pd.testing.assert_frame_equal(
        search_profile_for_rank_pattern(profile, ranking_query=["A", "B", "C"]).reset_index(
            drop=True
        ),
        true_profile.df.reset_index(drop=True),
        check_dtype=False,
    )


# --- max_cand_pair_dist searchs -----------------------------------


def test_search_with_max_cand_pair_dist():
    profile = RankProfile(
        ballots=(
            RankBallot(ranking=[{"A", "B"}, {"C"}, 1], weight=1.0),
            RankBallot(ranking=[{"B", "D"}, {"A"}, {"C"}], weight=1.0),
            RankBallot(ranking=[1, 2, "B", "C"], weight=1.0),
            RankBallot(ranking=[2, "C", "B"], weight=1.0),
        ),
        candidates=("A", "B", "C", "D", 1, 2),
        max_ranking_length=4,
    )
    profile_match_dist_0 = RankProfile(
        ballots=(
            RankBallot(ranking=[{"A", "B"}, {"C"}, 1], weight=1.0),
            RankBallot(ranking=[1, 2, "B", "C"], weight=1.0),
            RankBallot(ranking=[2, "C", "B"], weight=1.0),
        ),
        candidates=("A", "B", "C", "D", 1, 2),
        max_ranking_length=4,
    )
    profile_match_dist_1 = RankProfile(
        ballots=(
            RankBallot(ranking=[{"A", "B"}, {"C"}, 1], weight=1.0),
            RankBallot(ranking=[{"B", "D"}, {"A"}, {"C"}], weight=1.0),
            RankBallot(ranking=[1, 2, "B", "C"], weight=1.0),
            RankBallot(ranking=[2, "C", "B"], weight=1.0),
        ),
        candidates=("A", "B", "C", "D", 1, 2),
        max_ranking_length=4,
    )

    max_sep_dist_0 = {("B", "C"): 0}
    assert (
        search_profile_for_rank_pattern(profile, max_cand_pair_dist=max_sep_dist_0)
        .reset_index(drop=True)
        .equals(profile_match_dist_0.df.reset_index(drop=True))
    )

    max_sep_dist_1 = {("B", "C"): 1}
    assert (
        search_profile_for_rank_pattern(profile, max_cand_pair_dist=max_sep_dist_1)
        .reset_index(drop=True)
        .equals(profile_match_dist_1.df.reset_index(drop=True))
    )


def test_search_with_max_cand_pair_dist_with_multiple_cand_pairs():
    profile = RankProfile(
        ballots=(
            RankBallot(ranking=[{"A", "B"}, {"C"}, 1], weight=1.0),
            RankBallot(ranking=[{"B", "D"}, {"A"}, {"C"}], weight=1.0),
            RankBallot(ranking=[1, 2, "B", "C"], weight=1.0),
            RankBallot(ranking=[2, "C", "B"], weight=1.0),
        ),
        candidates=("A", "B", "C", "D", 1, 2),
        max_ranking_length=4,
    )
    profile_match_dist_0 = RankProfile(
        ballots=(RankBallot(ranking=[{"A", "B"}, {"C"}, 1], weight=1.0),),
        candidates=("A", "B", "C", "D", 1, 2),
        max_ranking_length=4,
    )

    max_sep_dist_0 = {("A", "B"): 0, ("B", "C"): 0}
    print(search_profile_for_rank_pattern(profile, max_cand_pair_dist=max_sep_dist_0))
    assert (
        search_profile_for_rank_pattern(profile, max_cand_pair_dist=max_sep_dist_0)
        .reset_index(drop=True)
        .equals(profile_match_dist_0.df.reset_index(drop=True))
    )


def test_search_with_max_cand_pair_dist_with_duplicate_candidate_rankings():
    profile = RankProfile(
        ballots=(
            RankBallot(ranking=["A", 1, "B", "A", "C", "A"], weight=1.0),
            RankBallot(ranking=[1, 2, "A", "D", "C", {"A", "B"}], weight=1.0),
            RankBallot(ranking=[1, 2, "A", "D", "C", "B"], weight=1.0),
            RankBallot(ranking=[1, 2, "B", "D", "C"], weight=1.0),
        ),
        candidates=("A", "B", "C", "D", 1, 2),
        max_ranking_length=6,
    )

    true_profile = RankProfile(
        ballots=(
            RankBallot(ranking=["A", 1, "B", "A", "C", "A"], weight=1.0),
            RankBallot(ranking=[1, 2, "A", "D", "C", {"A", "B"}], weight=1.0),
        ),
        candidates=("A", "B", "C", "D", 1, 2),
        max_ranking_length=6,
    )
    cand_pair_dict = {("A", "B"): 0}
    pd.testing.assert_frame_equal(
        search_profile_for_rank_pattern(profile, max_cand_pair_dist=cand_pair_dict).reset_index(
            drop=True
        ),
        true_profile.df.reset_index(drop=True),
        check_dtype=False,
    )


# --- ranking_query + max_cand_pair_dist search -----------------------------------


def test_search_with_matching_ranking_query_and_max_cand_pair_dist():
    profile = RankProfile(
        ballots=(
            RankBallot(ranking=[{"A", "B"}, {"C"}, 1], weight=1.0),
            RankBallot(ranking=[{"B", "D"}, {"A"}, {"C"}], weight=1.0),
            RankBallot(ranking=[1, 2, "B", "C"], weight=1.0),
            RankBallot(ranking=[2, "C", "B"], weight=1.0),
        ),
        candidates=("A", "B", "C", "D", 1, 2),
        max_ranking_length=4,
    )
    profile_match = RankProfile(
        ballots=(RankBallot(ranking=[2, "C", "B"], weight=1.0),),
        candidates=("A", "B", "C", "D", 1, 2),
        max_ranking_length=4,
    )
    max_sep_dist = {(2, "C"): 0}
    assert (
        search_profile_for_rank_pattern(
            profile, ranking_query=[2, "C"], max_cand_pair_dist=max_sep_dist
        )
        .reset_index(drop=True)
        .equals(profile_match.df.reset_index(drop=True))
    )


# --- include_unranked searchs -----------------------------------


def test_search_for_ranking_query_with_include_unranked():
    profile = RankProfile(
        ballots=(
            RankBallot(ranking=[{"A", "B"}, {"C"}, 1], weight=1.0),
            RankBallot(ranking=[{"B", "D"}, {"A"}, {"C"}], weight=1.0),
            RankBallot(ranking=[1, 2, "B", "C"], weight=1.0),
            RankBallot(ranking=[2, "C", "B"], weight=1.0),
        ),
        candidates=("A", "B", "C", "D", 1, 2),
        max_ranking_length=4,
    )
    profile_match = RankProfile(
        ballots=(
            RankBallot(ranking=[{"A", "B"}, {"C"}, 1], weight=1.0),
            RankBallot(ranking=[{"B", "D"}, {"A"}, {"C"}], weight=1.0),
        ),
        candidates=("A", "B", "C", "D", 1, 2),
        max_ranking_length=4,
    )
    assert (
        search_profile_for_rank_pattern(profile, ranking_query=["A", 1], include_unranked=True)
        .reset_index(drop=True)
        .equals(profile_match.df.reset_index(drop=True))
    )


# --- Search inputs validation -----------------------------------


def test_search_with_non_rank_profile_raises_error():
    with pytest.raises(TypeError, match="Profile must be a RankProfile"):
        search_profile_for_rank_pattern(ScoreProfile(), ranking_query=["A", "B"])  # type: ignore[arg-type]


def test_validate_ranking_query_raises_errors():
    invalid_set = frozenset({"A"})
    with pytest.raises(
        TypeError,
        match=re.escape(f"Use set for {invalid_set}, not frozenset within ranking_query."),
    ):
        _validate_ranking_query([invalid_set], RankProfile())  # type: ignore[arg-type]

    invalid_non_candidate_set = {1.0}
    with pytest.raises(TypeError, match="Set items must be 'str' or 'int' candidates"):
        _validate_ranking_query([invalid_non_candidate_set], RankProfile())

    invalid_tuple = (invalid_set,)
    with pytest.raises(
        TypeError, match=re.escape(f"Use set for {invalid_set}, not frozenset inside tuple")
    ):
        _validate_ranking_query([invalid_tuple], RankProfile())

    invalid_non_cand_set_in_tuple = (invalid_non_candidate_set,)
    with pytest.raises(
        TypeError, match=r"Set items must be 'str' or 'int' candidates inside tuple"
    ):
        _validate_ranking_query([invalid_non_cand_set_in_tuple], RankProfile())

    invalid_non_cand_tuple = (1.0,)
    with pytest.raises(TypeError, match="Tuple elements must be 'str'/'int' candidates or sets"):
        _validate_ranking_query([invalid_non_cand_tuple], RankProfile())

    profile = RankProfile(
        ballots=[
            RankBallot(ranking=({"A"},), weight=2),
        ]
    )
    mismatch_candidate = "B"
    with pytest.raises(ValueError, match="from ranking_query not in profile."):
        _validate_ranking_query(["A", mismatch_candidate], profile)


def test_validate_max_cand_pair_dist_raises_errors():
    profile = RankProfile(
        ballots=(
            RankBallot(ranking=[{"A", "C"}, {"B"}], weight=1.0),
            RankBallot(ranking=[{"A"}, {"B"}, {"D"}, {"C"}], weight=1.0),
            RankBallot(ranking=[1, 2, "C", "A"], weight=1.0),
        ),
        candidates=("A", "B", "C", "D", 1, 2),
        max_ranking_length=4,
    )

    invalid_dict = {("A", frozenset({"B"})): 0}
    with pytest.raises(
        TypeError, match="max_cand_pair_dist keys must be a tuple of candidate pairs"
    ):
        _validate_max_cand_pair_dist(invalid_dict, profile)

    invalid_dict = {("A", "B", "C"): 0}
    with pytest.raises(
        TypeError, match="max_cand_pair_dist keys must be a tuple of candidate pairs"
    ):
        _validate_max_cand_pair_dist(invalid_dict, profile)

    invalid_dict = {"AB": 0}
    with pytest.raises(
        TypeError, match="max_cand_pair_dist keys must be a tuple of candidate pairs"
    ):
        _validate_max_cand_pair_dist(invalid_dict, profile)  # type: ignore[arg-type]

    invalid_dict = {("A", "B"): 1.0}
    with pytest.raises(TypeError, match="max distances of max_cand_pair_dist must be integers"):
        _validate_max_cand_pair_dist(invalid_dict, profile)  # type: ignore[arg-type]

    invalid_dict = {("A", "B"): -1}
    with pytest.raises(
        ValueError, match="max distances of max_cand_pair_dist must be non-negative integers"
    ):
        _validate_max_cand_pair_dist(invalid_dict, profile)

    invalid_dict = {("X", "Y"): 0}
    with pytest.raises(ValueError, match=r"contain candidate\(s\) not in the profile"):
        _validate_max_cand_pair_dist(invalid_dict, profile)


def test_search_with_invalid_ranking_query_raises_error():
    invalid_set = frozenset({"A"})
    with pytest.raises(
        TypeError,
        match=re.escape(f"Use set for {invalid_set}, not frozenset within ranking_query."),
    ):
        search_profile_for_rank_pattern(RankProfile(), ranking_query=[invalid_set])  # type: ignore[arg-type]


def test_search_with_invalid_max_cand_pair_dist_raises_error():
    invalid_dict = {("A", "B", "C"): 0}
    with pytest.raises(
        TypeError, match="max_cand_pair_dist keys must be a tuple of candidate pairs"
    ):
        search_profile_for_rank_pattern(RankProfile(), max_cand_pair_dist=invalid_dict)


# --- _boolean_matrix -----------------------------------


def test_boolean_matrix_with_duplicates():
    profile = RankProfile(
        ballots=(
            RankBallot(ranking=[{"A", "B"}, {"C"}, 1], weight=1.0),
            RankBallot(ranking=[{"B", "D"}, {"A"}, {"C"}], weight=1.0),
            RankBallot(ranking=["A", "A", "B", "C"], weight=1.0),
            RankBallot(ranking=[2, "C", "B"], weight=1.0),
        ),
        candidates=("A", "B", "C", "D", 1, 2),
        max_ranking_length=4,
    )

    true_boolean_matrix = np.array(
        [
            [True, False, False, False],
            [False, True, False, False],
            [True, True, False, False],
            [False, False, False, False],
        ]
    )
    np.testing.assert_array_equal(_boolean_matrix(profile, "A"), true_boolean_matrix)


def test_boolean_matrix_with_strict_cand_set():
    profile = RankProfile(
        ballots=(
            RankBallot(ranking=[{"A", "B"}, {"C"}, 1], weight=1.0),
            RankBallot(ranking=[{"B", "D"}, {"A"}, {"C"}], weight=1.0),
            RankBallot(ranking=[1, 2, "B", "C"], weight=1.0),
            RankBallot(ranking=[2, "C", "B"], weight=1.0),
        ),
        candidates=("A", "B", "C", "D", 1, 2),
        max_ranking_length=4,
    )

    true_boolean_matrix = np.array(
        [
            [False, False, False, False],
            [False, True, False, False],
            [False, False, False, False],
            [False, False, False, False],
        ]
    )
    np.testing.assert_array_equal(
        _boolean_matrix(
            profile,
            {
                "A",
            },
        ),
        true_boolean_matrix,
    )


def test_boolean_matrix_include_unranked():
    profile = RankProfile(
        ballots=(
            RankBallot(ranking=[{"A", "B"}, {"C"}, 1], weight=1.0),
            RankBallot(ranking=[{"B", "D"}, {"A"}, {"C"}], weight=1.0),
            RankBallot(ranking=[1, 2, "B", "C"], weight=1.0),
            RankBallot(ranking=[2, "C"], weight=1.0),
        ),
        candidates=("A", "B", "C", "D", 1, 2),
        max_ranking_length=4,
    )

    true_boolean_matrix = np.array(
        [
            [True, False, False, False, False],
            [False, True, False, False, False],
            [False, False, False, False, True],
            [False, False, True, False, False],
        ]
    )
    np.testing.assert_array_equal(
        _boolean_matrix(profile, "A", include_unranked=True), true_boolean_matrix
    )


def test_boolean_matrix_include_unranked_with_strict_set():
    profile = RankProfile(
        ballots=(
            RankBallot(ranking=[{"A", "B"}, {"C"}, 1], weight=1.0),
            RankBallot(ranking=[{"B", "D"}, {"A"}, {"C"}], weight=1.0),
            RankBallot(ranking=[1, 2, "B", "C"], weight=1.0),
            RankBallot(ranking=[2, "C", "B"], weight=1.0),
        ),
        candidates=("A", "B", "C", "D", 1, 2),
        max_ranking_length=4,
    )

    true_boolean_matrix = np.array(
        [
            [False, False, False, False, False],
            [False, True, False, False, False],
            [False, False, False, False, False],
            [False, False, False, False, False],
        ]
    )
    np.testing.assert_array_equal(
        _boolean_matrix(
            profile,
            {
                "A",
            },
            include_unranked=True,
        ),
        true_boolean_matrix,
    )


# --- _compare_query_ranks -----------------------------------


def test_compare_query_ranks_for_ties():
    matrix_a_locations = np.array(
        [
            [False, False, False, False, False],
            [False, True, False, False, False],
            [False, False, False, False, False],
            [False, False, False, False, False],
        ]
    )
    matrix_b_locations = np.array(
        [
            [False, False, False, False, False],
            [False, True, False, False, False],
            [False, False, False, False, False],
            [False, False, False, False, False],
        ]
    )
    tied_query_a_b_location_matrices = [matrix_a_locations, matrix_b_locations]
    mask = np.ones(len(matrix_a_locations), dtype=bool)

    true_mask = np.zeros(len(matrix_a_locations), dtype=bool)

    np.testing.assert_array_equal(
        _compare_query_ranks(tied_query_a_b_location_matrices, mask), true_mask
    )


def test_compare_query_ranks_for_duplicates():
    matrix_a_locations_with_dups = np.array(
        [
            [False, False, False, False, False],
            [True, True, False, False, False],
            [False, False, False, False, False],
            [False, False, False, False, False],
        ]
    )
    matrix_b_locations = np.array(
        [
            [False, False, False, False, False],
            [False, True, False, False, False],
            [False, False, False, False, False],
            [False, False, False, False, False],
        ]
    )
    query_a_b_location_matrices = [matrix_a_locations_with_dups, matrix_b_locations]
    mask = np.ones(len(matrix_a_locations_with_dups), dtype=bool)

    true_mask = np.zeros(len(matrix_a_locations_with_dups), dtype=bool)
    true_mask[1] = True

    np.testing.assert_array_equal(
        _compare_query_ranks(query_a_b_location_matrices, mask), true_mask
    )


# --- _compare_candidate_pair_ranks -----------------------------------


def test_compare_candidate_pair_ranks_for_ties():
    matrix_a_locations = np.array(
        [
            [False, False, False, False, False],
            [False, True, False, False, False],
            [False, False, False, False, False],
            [False, False, False, False, False],
        ]
    )
    matrix_b_locations = np.array(
        [
            [False, False, False, False, False],
            [False, True, False, False, False],
            [False, False, False, False, False],
            [False, False, False, False, False],
        ]
    )
    tied_cand_a_b_location_matrices = [matrix_a_locations, matrix_b_locations]
    max_dist_cand_a_b = [0]
    mask = np.ones(len(matrix_a_locations), dtype=bool)

    true_mask = np.zeros(len(matrix_a_locations), dtype=bool)
    true_mask[1] = True

    np.testing.assert_array_equal(
        _compare_candidate_pair_ranks(tied_cand_a_b_location_matrices, max_dist_cand_a_b, mask),
        true_mask,
    )


def test_compare_candidate_pair_ranks_for_duplicates():
    matrix_a_locations_with_dups = np.array(
        [
            [False, False, False, False, False],
            [True, True, False, False, False],
            [False, False, False, False, False],
            [False, False, False, False, False],
        ]
    )
    matrix_b_locations = np.array(
        [
            [False, False, False, False, False],
            [False, True, False, False, False],
            [False, False, False, False, False],
            [False, False, False, False, False],
        ]
    )
    cand_a_b_location_matrices = [matrix_a_locations_with_dups, matrix_b_locations]
    max_dist_cand_a_b = [0]
    mask = np.ones(len(matrix_a_locations_with_dups), dtype=bool)

    true_mask = np.zeros(len(matrix_a_locations_with_dups), dtype=bool)
    true_mask[1] = True

    np.testing.assert_array_equal(
        _compare_candidate_pair_ranks(cand_a_b_location_matrices, max_dist_cand_a_b, mask),
        true_mask,
    )


# --- _shift_idx_to_next_ballot -----------------------------------


def test_shift_idx_to_next_ballot():
    curr_where_idx = 3
    ballot_indices = np.array([0, 0, 0, 1, 1, 1, 4, 5])
    true_next_ballot_idx = 6
    assert true_next_ballot_idx == _shift_idx_to_next_ballot(curr_where_idx, ballot_indices)
    assert ballot_indices[true_next_ballot_idx] != ballot_indices[curr_where_idx]
    assert ballot_indices[true_next_ballot_idx - 1] == ballot_indices[curr_where_idx]


# --- _get_next_true_ballot_with_cand -----------------------------------


def test_get_next_true_ballot_with_cand():
    ballots_mask = np.array([False, False, True, False, True])
    ballots_where_true = np.where(ballots_mask)[0]
    ballot_indices_where_cand = np.array([0, 2, 4])
    # next true ballot with candidate is at ballot index 2, which is at index 1 in where array
    true_next_ballot_where_idx = 1
    assert true_next_ballot_where_idx == _get_next_true_ballot_with_cand(
        ballots_where_true, 0, ballot_indices_where_cand
    )

    ballot_indices_where_cand = np.array([0, 1, 3])
    true_next_ballot_where_idx = None  # no true ballots with candidate present
    assert true_next_ballot_where_idx == _get_next_true_ballot_with_cand(
        ballots_where_true, 0, ballot_indices_where_cand
    )


# --- _get_candidate_pair_min_distance -----------------------------------


def test_get_candidate_pair_min_distance_with_duplicates():
    cand_a_ballots = np.array([0, 1, 1, 1, 4, 6])
    cand_a_ranks = np.array([1, 1, 3, 6, 0, 3])
    cand_b_ballots = np.array([1, 1, 2, 3])
    cand_b_ranks = np.array([3, 5, 1, 2])

    # get the minimum distance of cand a and b pair for ballot index 1
    cand_a_where_idx = 1
    cand_b_where_idx = 0
    true_min_distance = 0  # cand a and b are tied at index 3
    min_distance = _get_candidate_pair_min_distance(
        cand_a_where_idx,
        cand_b_where_idx,
        cand_a_ballots,
        cand_b_ballots,
        cand_a_ranks,
        cand_b_ranks,
    )
    assert min_distance == true_min_distance
