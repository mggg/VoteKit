from votekit import PreferenceProfile, Ballot
import numpy as np
import pytest
from scipy.stats import wasserstein_distance

from votekit.metrics.distances import (
    earth_mover_dist,
    emd_via_scipy_linear_program,
    __vaildate_ranking_distance_inputs,
)


def test_lp_perfect_match():
    source = np.array([0.5, 0.5])
    target = np.array([0.5, 0.5])
    cost = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert emd_via_scipy_linear_program(source, target, cost) == 0.0


def test_lp_one_to_one_transfer():
    source = np.array([1.0, 0.0])
    target = np.array([0.0, 1.0])
    cost = np.array([[0.0, 2.0], [2.0, 0.0]])
    assert emd_via_scipy_linear_program(source, target, cost) == 2.0


def test_lp_infeasible_transfer():
    source = np.array([1.0, 0.0])
    target = np.array([1.0, 1.0])
    cost = np.array([[0.0, 2.0], [2.0, 0.0]])
    with pytest.raises(RuntimeError, match="linprog failed"):
        emd_via_scipy_linear_program(source, target, cost)


def test_lp_unequal_sizes_simple_distances():
    source = np.array([0.5, 0.5, 0.0])
    target = np.array([0.0, 0.3, 0.7])
    cost = np.array([[0.0, 2.0, 3.0], [4.0, 0.0, 6.0], [7.0, 8.0, 0.0]])
    # Optimal transport plan:
    # [[0.0, 0.0, 0.5],
    #  [0.0, 0.3, 0.2],
    #  [0.0, 0.0, 0.0]]
    assert emd_via_scipy_linear_program(source, target, cost) == 2.7


def test_lp_unequal_sizes_simple_distances_movement_urged():
    source = np.array([0.5, 0.5, 0.0])
    target = np.array([0.0, 0.3, 0.7])
    # Note: staying still costs you!
    # Optimal transport plan:
    # [[0.0, 0.3, 0.2],
    #  [0.0, 0.0, 0.5],
    #  [0.0, 0.0, 0.0]]
    cost = np.array([[100.0, 2.0, 3.0], [4.0, 100.0, 6.0], [7.0, 8.0, 100.0]])
    assert emd_via_scipy_linear_program(source, target, cost) == 4.2


def test_against_scipy_wasserstein():

    vector_length = 50

    for _ in range(100):
        source = np.random.choice(np.arange(vector_length), size=vector_length).astype(
            np.float64
        )
        source /= np.sum(source)  # Normalize to sum to 1
        target = np.random.choice(np.arange(vector_length), size=vector_length).astype(
            np.float64
        )
        target /= np.sum(target)  # Normalize to sum to 1
        bins = np.arange(vector_length)
        cost = np.abs(bins[:, None] - bins[None, :])

        assert np.isclose(
            emd_via_scipy_linear_program(source, target, cost),
            wasserstein_distance(bins, bins, source, target),
            atol=1e-10,
        )


def make_random_profile(n_voters: int, cand_list: list[str]) -> PreferenceProfile:

    weights = np.unique_counts(list(map(int, np.random.gamma(5, 1, n_voters))))[1]

    n_cands = len(cand_list)
    all_cand_set = set(map(lambda x: frozenset({x}), cand_list))
    ballot_list = []
    for wt in weights:
        ranking = list(
            map(
                lambda x: frozenset({str(x)}),
                np.random.choice(
                    cand_list,
                    size=np.random.randint(1, len(cand_list)),
                    replace=False,
                ),
            )
        )
        if len(ranking) == n_cands - 1:
            ranking.append(*(all_cand_set - set(ranking)))

        ballot_list.append(
            Ballot(
                ranking=tuple(ranking),
                weight=wt,
            )
        )

    return PreferenceProfile(
        ballots=tuple(ballot_list),
        candidates=tuple(cand_list),
        max_ranking_length=n_cands,
    )


def _ballot(candidate_list: list[str], wt: int = 1) -> Ballot:
    """
    Helper function to create a ballot with a ranking of candidates.
    This makes it so mypy stops complaining about the ballot types
    """
    return Ballot(
        ranking=tuple(map(lambda x: frozenset({x}), candidate_list)),
        weight=wt,
    )


def test_earth_mover_dist_same_profile_is_zero():
    profile1 = PreferenceProfile(
        ballots=(_ballot(["A", "B"]), _ballot(["B", "A"])),
    )

    assert earth_mover_dist(profile1, profile1) == 0.0


def test_earth_mover_dist_transposition():
    profile1 = PreferenceProfile(
        ballots=(_ballot(["A", "B"]),),
    )
    profile2 = PreferenceProfile(
        ballots=(_ballot(["B", "A"]),),
    )

    assert earth_mover_dist(profile1, profile2) == 1.0


def test_earth_mover_dist_transposition_short_ballots():
    profile1 = PreferenceProfile(ballots=(_ballot(["A"]),), candidates=("A", "B"))
    profile2 = PreferenceProfile(ballots=(_ballot(["B"]),), candidates=("A", "B"))

    assert earth_mover_dist(profile1, profile2) == 1.0


def test_earth_mover_dist_move_one_ballot():
    profile1 = PreferenceProfile(
        ballots=(_ballot(["A", "B"], 2), _ballot(["B", "C"], 1))
    )
    profile2 = PreferenceProfile(
        ballots=(_ballot(["A", "B"], 1), _ballot(["B", "C"], 2))
    )

    assert earth_mover_dist(profile1, profile2) == 2 * (1 / 3)


def test_earth_mover_dist_move_several_ballots():
    profile1 = PreferenceProfile(
        ballots=(_ballot(["A", "B"], 2), _ballot(["B", "C"], 1), _ballot(["C"], 1))
    )
    profile2 = PreferenceProfile(
        ballots=(_ballot(["A", "B"], 1), _ballot(["B", "C"], 2), _ballot(["A"], 1))
    )

    assert earth_mover_dist(profile1, profile2) == (0.5 + 1.5) * (1 / 4)


def test_earth_mover_dist_readjust_weights():
    profile1 = PreferenceProfile(
        ballots=(_ballot(["A", "B"], 2), _ballot(["B", "C"], 1))
    )
    profile2 = PreferenceProfile(
        ballots=(_ballot(["A", "B"], 1), _ballot(["B", "C"], 1))
    )

    assert abs(earth_mover_dist(profile1, profile2) - 2 * (1 / 6)) < 1e-8


def test_earth_mover_dist_secretly_equivalent_profiles():
    profile1 = PreferenceProfile(
        ballots=(_ballot(["A", "B"]),), candidates=("A", "B"), max_ranking_length=2
    )
    profile2 = PreferenceProfile(
        ballots=(_ballot(["A"]),), candidates=("A", "B"), max_ranking_length=2
    )

    assert earth_mover_dist(profile1, profile2) == 0


def test_emd_validate_errors():
    with pytest.raises(ValueError, match="contains duplicates"):
        __vaildate_ranking_distance_inputs(
            ranking1=(1, 1, 3, 4, 5),
            ranking2=(1, 2, 3, 4, 5),
            n_candidates=5,
        )
    with pytest.raises(ValueError, match="contains duplicates"):
        __vaildate_ranking_distance_inputs(
            ranking1=(1, 2, 3, 4, 5),
            ranking2=(1, 1, 3, 4, 5),
            n_candidates=5,
        )

    with pytest.raises(ValueError, match="exceeds the total number of candidates"):
        __vaildate_ranking_distance_inputs(
            ranking1=(1, 2, 3, 4, 5),
            ranking2=(1, 2, 3, 4, 5),
            n_candidates=4,
        )


def test_emd_profile_errors():
    with pytest.raises(ValueError, match="contains duplicates"):
        earth_mover_dist(
            PreferenceProfile(ballots=(_ballot(["A", "B"]),), candidates=("A", "B")),
            PreferenceProfile(ballots=(_ballot(["A", "A"]),), candidates=("A", "B")),
        )

    with pytest.raises(
        ValueError, match="The two profiles must have the same candidates"
    ):
        earth_mover_dist(
            PreferenceProfile(ballots=(_ballot(["A", "B"]),)),
            PreferenceProfile(ballots=(_ballot(["A"]),)),
        )

    with pytest.raises(ValueError, match="Both profiles must contain rankings"):
        earth_mover_dist(
            PreferenceProfile(ballots=(_ballot(["A", "B"]),), candidates=("A", "B")),
            PreferenceProfile(
                ballots=(Ballot(scores={"A": 1, "B": 1}),),
                candidates=("A", "B"),
            ),
        )

    with pytest.raises(
        ValueError, match="Both profiles must have the same maximum ranking length"
    ):
        earth_mover_dist(
            PreferenceProfile(ballots=(_ballot(["A", "B"]),)),
            PreferenceProfile(ballots=(_ballot(["A", "B", "B"]),)),
        )
