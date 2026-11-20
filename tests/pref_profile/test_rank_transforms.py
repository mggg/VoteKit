import pytest

from votekit.ballot import RankBallot
from votekit.pref_profile import RankProfile
from votekit.pref_profile.profile_transform import SwapAdjacentRankPositions, SwapCandidates
from votekit.types import Candidate


def ranking(*positions: str | set[str]) -> tuple[frozenset[Candidate], ...]:
    """A ranking from its positions: a candidate, a set of tied candidates, or set() if empty."""
    return tuple(
        frozenset({position}) if isinstance(position, str) else frozenset(position)
        for position in positions
    )


def ranking_weights(profile: RankProfile) -> dict[tuple[frozenset[Candidate], ...], float]:
    """Total weight of each ranking, with unranked padding dropped."""
    matrix = profile._matrix
    weights: dict[tuple[frozenset[Candidate], ...], float] = {}
    for row_idx in range(len(matrix)):
        ranked = matrix.get_ranking(row_idx)
        weights[ranked] = weights.get(ranked, 0.0) + float(
            matrix.voter_weight_data.weights[row_idx]
        )
    return weights


def test_swap_candidates_swaps_their_positions():
    profile = RankProfile(ballots=[RankBallot(ranking=[{"A"}, {"C"}, {"B"}])])

    perturbed = profile.perturb(SwapCandidates("A", "B"))

    assert ranking_weights(perturbed) == {ranking("B", "C", "A"): 1.0}


def test_swap_candidates_moves_a_tie_and_frees_a_singleton():
    profile = RankProfile(ballots=[RankBallot(ranking=[{"A", "C"}, {"B"}])], max_ranking_length=3)

    perturbed = profile.perturb(SwapCandidates("A", "B"))

    assert ranking_weights(perturbed) == {ranking({"B", "C"}, "A"): 1.0}


def test_swap_candidates_leaves_the_two_tied_together_unchanged():
    profile = RankProfile(ballots=[RankBallot(ranking=[{"A", "B"}, {"C"}])], max_ranking_length=3)

    perturbed = profile.perturb(SwapCandidates("A", "B"))

    assert ranking_weights(perturbed) == {ranking({"A", "B"}, "C"): 1.0}


def test_swap_candidates_with_only_one_of_them_on_the_ballot():
    profile = RankProfile(ballots=[RankBallot(ranking=[{"A"}, {"C"}])], candidates=["A", "B", "C"])

    perturbed = profile.perturb(SwapCandidates("A", "B"))

    assert ranking_weights(perturbed) == {ranking("B", "C"): 1.0}


def test_swap_candidates_keeps_empty_positions():
    profile = RankProfile(ballots=[RankBallot(ranking=[{"A"}, set(), {"B"}])])

    perturbed = profile.perturb(SwapCandidates("A", "B"))

    assert ranking_weights(perturbed) == {ranking("B", set(), "A"): 1.0}


def test_swap_candidates_on_a_mix_of_short_ballots():
    profile = RankProfile(
        ballots=[
            RankBallot(ranking=[{"A"}]),
            RankBallot(ranking=[{"C"}, {"B"}]),
            RankBallot(ranking=[{"A"}, {"B"}, {"C"}]),
        ]
    )

    perturbed = profile.perturb(SwapCandidates("A", "B"))

    assert ranking_weights(perturbed) == {
        ranking("B"): 1.0,
        ranking("C", "A"): 1.0,
        ranking("B", "A", "C"): 1.0,
    }


def test_swap_candidates_rejects_a_repeated_candidate():
    profile = RankProfile(
        ballots=[RankBallot(ranking=[{"C"}]), RankBallot(ranking=[{"B"}, {"C"}, {"B"}])],
        candidates=["A", "B", "C"],
    )

    with pytest.raises(ValueError, match="Ballot Index 1 ranks one of them in more than one"):
        profile.perturb(SwapCandidates("A", "B"))


def test_swap_candidates_rejects_an_undeclared_candidate():
    profile = RankProfile(ballots=[RankBallot(ranking=[{"A"}, {"B"}])])

    with pytest.raises(ValueError, match=r"\{'Z'\} not in the profile's candidates"):
        profile.perturb(SwapCandidates("A", "Z"))


def test_swap_candidates_rejects_two_misspelled_candidates_instead_of_changing_nothing():
    profile = RankProfile(ballots=[RankBallot(ranking=[{"A"}, {"B"}])])

    with pytest.raises(ValueError, match="not in the profile's candidates") as error:
        profile.perturb(SwapCandidates("X", "Y"))
    assert "'X'" in str(error.value) and "'Y'" in str(error.value)


def test_swap_candidates_rejects_an_undeclared_candidate_even_when_no_voter_is_selected():
    profile = RankProfile(ballots=[RankBallot(ranking=[{"A"}, {"B"}])])

    with pytest.raises(ValueError, match="not in the profile's candidates"):
        profile.perturb(SwapCandidates("A", "Z"), 0.0)


def test_swap_candidates_with_the_same_candidate_twice_changes_nothing():
    profile = RankProfile(ballots=[RankBallot(ranking=[{"A"}, {"B"}])])

    perturbed = profile.perturb(SwapCandidates("A", "A"))

    assert ranking_weights(perturbed) == {ranking("A", "B"): 1.0}


def test_adjacent_swap_only_swaps_neighbors():
    profile = RankProfile(ballots=[RankBallot(ranking=[{"A"}, {"B"}, {"C"}], weight=300)])

    perturbed = profile.perturb(SwapAdjacentRankPositions(rng_seed=0))

    weights = ranking_weights(perturbed)
    assert set(weights) == {ranking("A", "B", "C"), ranking("B", "A", "C"), ranking("A", "C", "B")}
    assert sum(weights.values()) == 300


def test_adjacent_swap_never_swaps_into_padding():
    profile = RankProfile(
        ballots=[RankBallot(ranking=[{"A"}, {"B"}], weight=300)],
        candidates=["A", "B", "C", "D"],
        max_ranking_length=4,
    )

    perturbed = profile.perturb(SwapAdjacentRankPositions(rng_seed=0))

    assert set(ranking_weights(perturbed)) == {ranking("A", "B"), ranking("B", "A")}


def test_adjacent_swap_swaps_an_empty_position_like_any_other():
    profile = RankProfile(ballots=[RankBallot(ranking=[{"A"}, set(), {"B"}], weight=300)])

    perturbed = profile.perturb(SwapAdjacentRankPositions(rng_seed=0))

    assert set(ranking_weights(perturbed)) == {
        ranking("A", set(), "B"),
        ranking(set(), "A", "B"),
        ranking("A", "B", set()),
    }


def test_adjacent_swap_leaves_a_one_position_ranking_unchanged():
    profile = RankProfile(ballots=[RankBallot(ranking=[{"A"}], weight=20)], max_ranking_length=1)

    perturbed = profile.perturb(SwapAdjacentRankPositions(rng_seed=0))

    assert ranking_weights(perturbed) == {ranking("A"): 20.0}


def test_adjacent_swap_draws_each_swap_with_equal_probability():
    profile = RankProfile(ballots=[RankBallot(ranking=[{"A"}, {"B"}, {"C"}], weight=1200)])

    perturbed = profile.perturb(SwapAdjacentRankPositions(rng_seed=0))

    # position 0 only swaps right, position 2 only swaps left, position 1 swaps either way,
    # so each adjacent pair and no swap each occur with probability 1/3
    weights = ranking_weights(perturbed)
    for ranked in [ranking("A", "B", "C"), ranking("B", "A", "C"), ranking("A", "C", "B")]:
        assert weights[ranked] / 1200 == pytest.approx(1 / 3, abs=0.05)


def test_adjacent_swap_is_reproducible_and_a_reused_transform_continues_its_stream():
    profile = RankProfile(ballots=[RankBallot(ranking=[{"A"}, {"B"}, {"C"}], weight=50)])

    first = ranking_weights(profile.perturb(SwapAdjacentRankPositions(rng_seed=1)))
    again = ranking_weights(profile.perturb(SwapAdjacentRankPositions(rng_seed=1)))
    assert first == again

    reused = SwapAdjacentRankPositions(rng_seed=1)
    assert ranking_weights(profile.perturb(reused)) != ranking_weights(profile.perturb(reused))
