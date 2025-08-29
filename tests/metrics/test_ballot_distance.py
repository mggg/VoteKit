from votekit.metrics.distances import compute_ranking_distance_on_ballot_graph
import pytest


def test_ballot_distance_disjoint_rankings_partition_candidates():
    assert compute_ranking_distance_on_ballot_graph((1, 2), (3, 4, 5), 5) == 7.5


def test_ballot_distance_disjoint_rankings_donot_partition_candidates():
    assert compute_ranking_distance_on_ballot_graph((1, 2), (3, 4, 5), 8) == 8.5


def test_ballot_distance_subset_rankings_subset_candidates():
    assert compute_ranking_distance_on_ballot_graph((1, 2), (3, 4, 2, 5, 1), 8) == 7.5


def test_ballot_distance_subset_rankings_all_candidates():
    assert compute_ranking_distance_on_ballot_graph((1, 2), (3, 4, 2, 5, 1), 5) == 7.0


def test_ballot_distance_same_rankings_is_zero():
    assert compute_ranking_distance_on_ballot_graph((1, 2), (1, 2), 5) == 0.0


def test_ballot_distance_same_rankings_in_disguise_is_zero():
    assert (
        compute_ranking_distance_on_ballot_graph((1, 2, 3, 4), (1, 2, 3, 4, 5), 5)
        == 0.0
    )


def test_ballot_distance_bullet_vs_full_ranking():
    assert compute_ranking_distance_on_ballot_graph((1, 2, 3, 4, 5), (5,), 5) == 5.5


def test_ballot_distance_bullet_vs_full_ranking_in_disguise():
    assert compute_ranking_distance_on_ballot_graph((1, 2, 3, 4), (5,), 5) == 5.5


def test_ballot_distance_bullet_vs_not_full_ranking():
    assert compute_ranking_distance_on_ballot_graph((1, 2, 3, 4, 5), (5,), 8) == 6.0


def test_ballot_distance_bullet_and_partial_ranking():
    assert compute_ranking_distance_on_ballot_graph((1, 2, 3), (5,), 5) == 5.0


def test_ballot_distance_bullet_and_empty_ranking():
    assert compute_ranking_distance_on_ballot_graph(tuple(), (5,), 5) == 0.5


def test_ballot_distance_emptry_ranking_and_partial_ranking():
    assert compute_ranking_distance_on_ballot_graph(tuple(), (1, 2, 3), 5) == 1.5


def test_ballot_distance_bullet_and_full_ranking():
    assert compute_ranking_distance_on_ballot_graph((1,), (1, 2, 3, 4, 5), 5) == 1.5


def test_ballot_distance_bullet_and_not_full_ranking():
    assert compute_ranking_distance_on_ballot_graph((1,), (1, 2, 3, 4, 5), 6) == 2.0


def test_ballot_distance_empty_ballot_and_full_ranking():
    assert compute_ranking_distance_on_ballot_graph(tuple(), (1, 2, 3, 4, 5), 5) == 2.0


def test_ballot_distance_two_bullet_ballots():
    assert compute_ranking_distance_on_ballot_graph((1,), (2,), 5) == 2.0


def test_ballot_distance_two_bullet_ballots_same():
    assert compute_ranking_distance_on_ballot_graph((1,), (1,), 5) == 0.0


def test_ballot_distance_two_bullet_ballots_partition_candidates():
    assert compute_ranking_distance_on_ballot_graph((0,), (1,), 2) == 1.0


def test_ballot_distance_2_candidates_with_transposition():
    assert compute_ranking_distance_on_ballot_graph((0, 1), (1, 0), 2) == 1.0


def test_ballot_distance_full_ranking_transposition_at_end():
    assert (
        compute_ranking_distance_on_ballot_graph((0, 1, 2, 3), (0, 1, 3, 2), 4) == 1.0
    )


def test_ballot_distance_errors():
    with pytest.raises(
        ValueError, match="The number of candidates must be greater than zero."
    ):
        compute_ranking_distance_on_ballot_graph((1, 2), (3, 4, 5), 0)

    with pytest.raises(
        ValueError, match="The number of candidates must be greater than zero."
    ):
        compute_ranking_distance_on_ballot_graph((1, 2), (3, 4, 5), -1)
