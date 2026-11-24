import re

import numpy as np
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

    transformed = profile.transform(SwapCandidates("A", "B"))

    assert ranking_weights(transformed) == {ranking("B", "C", "A"): 1.0}


def test_swap_candidates_moves_a_tie_and_frees_a_singleton():
    profile = RankProfile(ballots=[RankBallot(ranking=[{"A", "C"}, {"B"}])], max_ranking_length=3)

    transformed = profile.transform(SwapCandidates("A", "B"))

    assert ranking_weights(transformed) == {ranking({"B", "C"}, "A"): 1.0}


def test_swap_candidates_leaves_the_two_tied_together_unchanged():
    profile = RankProfile(ballots=[RankBallot(ranking=[{"A", "B"}, {"C"}])], max_ranking_length=3)

    transformed = profile.transform(SwapCandidates("A", "B"))

    assert ranking_weights(transformed) == {ranking({"A", "B"}, "C"): 1.0}


def test_swap_candidates_with_only_one_of_them_on_the_ballot():
    profile = RankProfile(ballots=[RankBallot(ranking=[{"A"}, {"C"}])], candidates=["A", "B", "C"])

    transformed = profile.transform(SwapCandidates("A", "B"))

    assert ranking_weights(transformed) == {ranking("B", "C"): 1.0}


def test_swap_candidates_keeps_empty_positions():
    profile = RankProfile(ballots=[RankBallot(ranking=[{"A"}, set(), {"B"}])])

    transformed = profile.transform(SwapCandidates("A", "B"))

    assert ranking_weights(transformed) == {ranking("B", set(), "A"): 1.0}


def test_swap_candidates_on_a_mix_of_short_ballots():
    profile = RankProfile(
        ballots=[
            RankBallot(ranking=[{"A"}]),
            RankBallot(ranking=[{"C"}, {"B"}]),
            RankBallot(ranking=[{"A"}, {"B"}, {"C"}]),
        ]
    )

    transformed = profile.transform(SwapCandidates("A", "B"))

    assert ranking_weights(transformed) == {
        ranking("B"): 1.0,
        ranking("C", "A"): 1.0,
        ranking("B", "A", "C"): 1.0,
    }


def test_swap_candidates_rejects_a_repeated_candidate():
    profile = RankProfile(
        ballots=[RankBallot(ranking=[{"C"}]), RankBallot(ranking=[{"B"}, {"C"}, {"B"}])],
        candidates=["A", "B", "C"],
    )

    ballot = re.escape("the ballot (frozenset({'B'}), frozenset({'C'}), frozenset({'B'}))")
    with pytest.raises(ValueError, match=f"{ballot} ranks one of them in more than one"):
        profile.transform(SwapCandidates("A", "B"))


def test_swap_candidates_rejects_an_undeclared_candidate():
    profile = RankProfile(ballots=[RankBallot(ranking=[{"A"}, {"B"}])])

    with pytest.raises(ValueError, match=r"\{'Z'\} not in the profile's candidates"):
        profile.transform(SwapCandidates("A", "Z"))


def test_swap_candidates_rejects_two_misspelled_candidates_instead_of_changing_nothing():
    profile = RankProfile(ballots=[RankBallot(ranking=[{"A"}, {"B"}])])

    with pytest.raises(ValueError, match="not in the profile's candidates") as error:
        profile.transform(SwapCandidates("X", "Y"))
    assert "'X'" in str(error.value) and "'Y'" in str(error.value)


def test_swap_candidates_with_the_same_candidate_twice_changes_nothing():
    profile = RankProfile(ballots=[RankBallot(ranking=[{"A"}, {"B"}])])

    transformed = profile.transform(SwapCandidates("A", "A"))

    assert ranking_weights(transformed) == {ranking("A", "B"): 1.0}


def test_adjacent_swap_only_swaps_neighbors():
    profile = RankProfile(ballots=[RankBallot(ranking=[{"A"}, {"B"}, {"C"}], weight=300)])

    transformed = profile.transform(SwapAdjacentRankPositions(), rng=0)

    weights = ranking_weights(transformed)
    assert set(weights) == {ranking("A", "B", "C"), ranking("B", "A", "C"), ranking("A", "C", "B")}
    assert sum(weights.values()) == 300


def test_adjacent_swap_never_swaps_into_padding():
    profile = RankProfile(
        ballots=[RankBallot(ranking=[{"A"}, {"B"}], weight=300)],
        candidates=["A", "B", "C", "D"],
        max_ranking_length=4,
    )

    transformed = profile.transform(SwapAdjacentRankPositions(), rng=0)

    assert set(ranking_weights(transformed)) == {ranking("A", "B"), ranking("B", "A")}


def test_adjacent_swap_swaps_an_empty_position_like_any_other():
    profile = RankProfile(ballots=[RankBallot(ranking=[{"A"}, set(), {"B"}], weight=300)])

    transformed = profile.transform(SwapAdjacentRankPositions(), rng=0)

    assert set(ranking_weights(transformed)) == {
        ranking("A", set(), "B"),
        ranking(set(), "A", "B"),
        ranking("A", "B", set()),
    }


def test_adjacent_swap_leaves_a_one_position_ranking_unchanged():
    profile = RankProfile(ballots=[RankBallot(ranking=[{"A"}], weight=20)], max_ranking_length=1)

    transformed = profile.transform(SwapAdjacentRankPositions(), rng=0)

    assert ranking_weights(transformed) == {ranking("A"): 20.0}


def test_adjacent_swap_draws_each_swap_with_equal_probability():
    profile = RankProfile(ballots=[RankBallot(ranking=[{"A"}, {"B"}, {"C"}], weight=1200)])

    transformed = profile.transform(SwapAdjacentRankPositions(), rng=0)

    # position 0 only swaps right, position 2 only swaps left, position 1 swaps either way,
    # so each adjacent pair and no swap each occur with probability 1/3
    weights = ranking_weights(transformed)
    for ranked in [ranking("A", "B", "C"), ranking("B", "A", "C"), ranking("A", "C", "B")]:
        assert weights[ranked] / 1200 == pytest.approx(1 / 3, abs=0.05)


def test_adjacent_swap_is_reproducible_and_a_shared_generator_continues_its_stream():
    profile = RankProfile(ballots=[RankBallot(ranking=[{"A"}, {"B"}, {"C"}], weight=50)])

    first = ranking_weights(profile.transform(SwapAdjacentRankPositions(), rng=1))
    again = ranking_weights(profile.transform(SwapAdjacentRankPositions(), rng=1))
    assert first == again

    generator = np.random.default_rng(1)
    first_draw = ranking_weights(profile.transform(SwapAdjacentRankPositions(), rng=generator))
    next_draw = ranking_weights(profile.transform(SwapAdjacentRankPositions(), rng=generator))
    assert first_draw == first
    assert next_draw != first_draw


def test_adjacent_swap_gives_each_outcome_one_row():
    profile = RankProfile(ballots=[RankBallot(ranking=[{"A"}, {"B"}, {"C"}], weight=300)])

    df = profile.transform(SwapAdjacentRankPositions(), rng=0, group_ballots=False).to_df()

    assert len(df) == 3
    assert df["Weight"].sum() == 300


def test_adjacent_swap_splits_voters_between_outcomes():
    voters = {f"v{i}" for i in range(30)}
    profile = RankProfile(ballots=[RankBallot(ranking=[{"A"}, {"B"}], weight=30, voter_set=voters)])

    df = profile.transform(SwapAdjacentRankPositions(), rng=0).to_df()

    assert set().union(*df["Voter Set"]) == voters
    assert sum(len(voter_set) for voter_set in df["Voter Set"]) == 30
    assert [len(voter_set) for voter_set in df["Voter Set"]] == df["Weight"].tolist()


def test_adjacent_swap_gives_each_voter_their_ballot_or_one_adjacent_swap():
    # many small rows, so some rows run out of voters partway through the pairs
    rng = np.random.default_rng(5)
    original_ballots = {}
    ballots = []
    for row_idx in range(200):
        num_ranked = int(rng.integers(1, 6))
        ranked = ranking(*rng.permutation(list("ABCDE"))[:num_ranked])
        voters = {f"v{row_idx}_{unit}" for unit in range(int(rng.integers(0, 4)))}
        original_ballots |= dict.fromkeys(voters, ranked)
        ballots.append(RankBallot(ranking=list(ranked), weight=len(voters), voter_set=voters))
    profile = RankProfile(ballots=ballots, candidates=list("ABCDE"), max_ranking_length=5)

    transformed = profile.transform(SwapAdjacentRankPositions(), rng=1, group_ballots=False)

    transformed_ballots = {
        voter: transformed._matrix.get_ranking(row_idx)
        for row_idx, voter_set in enumerate(transformed.to_df()["Voter Set"])
        for voter in voter_set
    }
    assert transformed_ballots.keys() == original_ballots.keys()
    for voter, ballot in transformed_ballots.items():
        before = original_ballots[voter]
        changed = [pos for pos in range(len(before)) if before[pos] != ballot[pos]]
        assert changed == [] or (
            len(changed) == 2
            and changed[1] == changed[0] + 1
            and before[changed[0]] == ballot[changed[1]]
        )
