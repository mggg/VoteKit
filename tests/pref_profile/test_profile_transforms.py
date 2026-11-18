from dataclasses import dataclass, replace
from fractions import Fraction
from functools import partial

import numpy as np
import pandas as pd
import pytest

from votekit.ballot import RankBallot, ScoreBallot
from votekit.exceptions import TransformContractError
from votekit.pref_profile import ProfileError, RankProfile, ScoreProfile
from votekit.pref_profile.profile_matrix import ScoreMatrix
from votekit.pref_profile.transform import (
    SwapAdjacentRankPositions,
    SwapCandidates,
    perturb,
)


def make_profile() -> RankProfile:
    return RankProfile(
        ballots=[
            RankBallot(ranking=[{"A"}, {"B"}, {"C"}], weight=10),
            RankBallot(ranking=[{"A"}, {"C"}, {"B"}], weight=10),
        ]
    )


def ranking_weights(profile: RankProfile) -> dict[tuple[str, ...], float]:
    """Map each ranking to its total weight, ignoring row order."""
    weights: dict[tuple[str, ...], float] = {}
    ranking_cols = [col for col in profile.to_df().columns if "Ranking_" in col]
    for _, row in profile.to_df().iterrows():
        ranking = tuple(next(iter(row[col]), "") for col in ranking_cols if row[col] != {"~"})
        weights[ranking] = weights.get(ranking, 0.0) + row["Weight"]
    return weights


def reverse(ranking):
    """Reverses the ranked positions in place, leaving trailing padding at the end."""
    num_ranked = sum(cand_set != {"~"} for cand_set in ranking)
    ranking[:num_ranked] = ranking[:num_ranked][::-1]
    return ranking


@dataclass
class SwapRandomAdjacentPositions:
    """Swaps a random pair of adjacent ranking positions on each voter's ballot."""

    rng: np.random.Generator

    def transform_matrix(self, matrix):
        ranking_arr = matrix.rankings.copy()
        rows = np.arange(len(ranking_arr))
        positions = self.rng.integers(0, matrix.num_ranking_positions - 1, size=len(ranking_arr))
        left = ranking_arr[rows, positions].copy()
        ranking_arr[rows, positions] = ranking_arr[rows, positions + 1]
        ranking_arr[rows, positions + 1] = left
        return replace(matrix, rankings=ranking_arr)


def test_swap_candidates():
    perturbed = perturb(make_profile(), SwapCandidates("A", "B"))

    assert ranking_weights(perturbed) == {("B", "A", "C"): 10.0, ("B", "C", "A"): 10.0}


def test_swap_candidates_with_probability():
    perturbed = perturb(make_profile(), SwapCandidates("A", "B"), 0.5, rng_seed=42)
    weights = ranking_weights(perturbed)

    assert perturbed.total_ballot_wt == 20.0
    assert weights[("A", "B", "C")] + weights[("B", "A", "C")] == 10.0
    assert weights[("A", "C", "B")] + weights[("B", "C", "A")] == 10.0
    assert 0 < weights[("B", "A", "C")] < 10


def swapped_rankings(ballots, candidates, candidate_a="A", candidate_b="B"):
    """Rankings after swapping two candidates directly on the profile's matrix."""
    profile = RankProfile(ballots=ballots, candidates=candidates, max_ranking_length=3)
    matrix = SwapCandidates(candidate_a, candidate_b).transform_matrix(profile._matrix)
    return [matrix.get_ranking(row) for row in range(len(matrix))]


@pytest.mark.parametrize(
    "ranking, expected",
    [
        # a tie with one candidate becomes a tie with the other, freeing a singleton
        ([{"A", "C"}, {"B"}], (frozenset({"B", "C"}), frozenset({"A"}))),
        # a freed singleton that was not in the table is added
        ([{"A", "C"}, {"D"}], (frozenset({"B", "C"}), frozenset({"D"}))),
        # a ballot with only one of the two candidates still swaps
        ([{"A"}, {"C"}], (frozenset({"B"}), frozenset({"C"}))),
        # the two candidates tied together are unchanged
        ([{"A", "B"}, {"C"}], (frozenset({"A", "B"}), frozenset({"C"}))),
    ],
)
def test_swap_candidates_with_ties(ranking, expected):
    rankings = swapped_rankings([RankBallot(ranking=ranking)], ["A", "B", "C", "D"])

    assert rankings == [expected]


def test_swap_candidates_repeated_candidate_raises():
    ballots = [RankBallot(ranking=[{"C"}]), RankBallot(ranking=[{"B"}, {"C"}, {"B"}])]

    with pytest.raises(ValueError, match="Ballot Index 1 ranks one of them in more than one"):
        swapped_rankings(ballots, ["A", "B", "C"])


def test_seed_is_reproducible():
    first = perturb(make_profile(), reverse, 0.5, rng_seed=7)
    second = perturb(make_profile(), reverse, 0.5, rng_seed=7)

    assert ranking_weights(first) == ranking_weights(second)


def test_random_profile_transform_owns_its_randomness():
    def run():
        transform = SwapRandomAdjacentPositions(np.random.default_rng(11))
        return perturb(make_profile(), transform, 0.5, rng_seed=7)

    assert ranking_weights(run()) == ranking_weights(run())


def test_zero_probability_leaves_profile_unchanged():
    perturbed = perturb(make_profile(), reverse, 0.0, rng_seed=1)

    assert ranking_weights(perturbed) == ranking_weights(make_profile())


def test_partial_selection_keeps_profile_transform_rows_grouped():
    perturbed = perturb(make_profile(), SwapCandidates("A", "B"), 0.5, rng_seed=42)
    first_choices = [next(iter(cand_set)) for cand_set in perturbed.to_df()["Ranking_1"]]

    # each original row splits into one unperturbed row and one grouped perturbed row
    assert first_choices.count("A") == 2
    assert first_choices.count("B") == 2


def test_profile_transform_receives_rows_weighted_by_selected_voters():
    received = []

    class Record:
        def transform_matrix(self, matrix):
            received.append(list(matrix.voter_weight_data.weights))
            return matrix

    perturb(make_profile(), Record())

    assert received == [[10.0, 10.0]]


def test_per_voter_function_splits_selected_voters_into_unit_rows():
    perturbed = perturb(make_profile(), reverse, 0.5, rng_seed=42)
    first_choices = [next(iter(cand_set)) for cand_set in perturbed.to_df()["Ranking_1"]]

    # each original row keeps one unperturbed row, every perturbed voter is its own row
    assert first_choices.count("A") == 2
    assert all(
        weight == 1
        for first, weight in zip(first_choices, perturbed.to_df()["Weight"])
        if first != "A"
    )


@pytest.mark.parametrize("profile", [RankProfile(), ScoreProfile()])
def test_empty_profile(profile):
    perturbed = perturb(profile, reverse)

    assert isinstance(perturbed, type(profile))
    assert perturbed.num_ballots == 0


def test_zero_weight_rows_are_kept():
    profile = RankProfile(
        ballots=[
            RankBallot(ranking=[{"A"}, {"B"}], weight=2),
            RankBallot(ranking=[{"B"}, {"A"}], weight=0),
        ]
    )
    perturbed = perturb(profile, reverse)

    # the exhausted weight 2 row is dropped, the zero weight row is kept
    assert sorted(perturbed.to_df()["Weight"]) == [0.0, 1.0, 1.0]
    assert ranking_weights(perturbed) == {("B", "A"): 2.0}


def test_group_ballots_regroups_output():
    perturbed = perturb(make_profile(), SwapCandidates("A", "B"), group_ballots=True)

    assert len(perturbed.to_df()) == 2
    assert ranking_weights(perturbed) == {("B", "A", "C"): 10.0, ("B", "C", "A"): 10.0}


def test_ranking_function_is_called_once_per_selected_voter():
    calls = []

    def record(ranking):
        calls.append(ranking)
        return ranking

    perturb(make_profile(), record)

    assert len(calls) == 20


def test_ranking_function_receives_padded_array():
    received = []

    def record(ranking):
        received.append(ranking.copy())
        return reverse(ranking)

    profile = RankProfile(
        ballots=[RankBallot(ranking=[{"A"}, {"B"}], weight=1)],
        candidates=["A", "B", "C"],
        max_ranking_length=3,
    )
    perturbed = perturb(profile, record)

    assert isinstance(received[0], np.ndarray)
    assert list(received[0]) == [frozenset({"A"}), frozenset({"B"}), frozenset({"~"})]
    assert ranking_weights(perturbed) == {("B", "A"): 1.0}


@pytest.mark.parametrize("container", [list, tuple, np.array])
def test_ranking_function_may_return_any_1d_container(container):
    def swap_first_two(ranking):
        return container([ranking[1], ranking[0], ranking[2]])

    perturbed = perturb(make_profile(), swap_first_two, group_ballots=True)

    assert ranking_weights(perturbed) == {("B", "A", "C"): 10.0, ("C", "A", "B"): 10.0}


def test_ranking_function_may_return_sets():
    perturbed = perturb(
        make_profile(), lambda ranking: [{"C"}, {"B"}, {"A"}, *ranking[3:]], group_ballots=True
    )

    assert ranking_weights(perturbed) == {("C", "B", "A"): 20.0}
    assert all(
        isinstance(cand_set, frozenset)
        for col in ["Ranking_1", "Ranking_2", "Ranking_3"]
        for cand_set in perturbed.to_df()[col]
    )


def test_ranking_function_can_add_declared_spoiler():
    def add_spoiler(ranking):
        ranking[2] = frozenset({"Spoiler"})
        return ranking

    profile = RankProfile(
        ballots=[RankBallot(ranking=[{"A"}, {"B"}], weight=2)],
        candidates=["A", "B", "Spoiler"],
        max_ranking_length=3,
    )
    perturbed = perturb(profile, add_spoiler, group_ballots=True)

    assert ranking_weights(perturbed) == {("A", "B", "Spoiler"): 2.0}


def test_curried_ranking_function():
    def truncate(ranking, length):
        ranking[length:] = frozenset({"~"})
        return ranking

    perturbed = perturb(make_profile(), partial(truncate, length=1), group_ballots=True)

    assert ranking_weights(perturbed) == {("A",): 20.0}


def test_probability_function_receives_read_only_array():
    def write(ranking):
        ranking[0] = frozenset({"B"})
        return 1.0

    with pytest.raises(ValueError, match="read-only"):
        perturb(make_profile(), reverse, write)


def test_probability_function_filters_rows():
    perturbed = perturb(
        make_profile(),
        reverse,
        lambda ranking: 1.0 if ranking[1] == {"B"} else 0.0,
        group_ballots=True,
    )

    assert ranking_weights(perturbed) == {("C", "B", "A"): 10.0, ("A", "C", "B"): 10.0}


@pytest.mark.parametrize("probability", [-0.1, 1.5, float("nan"), lambda ranking: 2.0])
def test_probability_out_of_range_raises(probability):
    with pytest.raises(ValueError, match="must lie in"):
        perturb(make_profile(), reverse, probability)


def test_probability_not_numeric_raises():
    with pytest.raises(TypeError, match="real number"):
        perturb(make_profile(), reverse, lambda ranking: "high")


def test_non_integral_weights_raise():
    profile = RankProfile(ballots=[RankBallot(ranking=[{"A"}, {"B"}], weight=1.5)])

    with pytest.raises(ValueError, match="non-integral weights"):
        perturb(profile, reverse)


def test_class_instead_of_instance_raises():
    with pytest.raises(TransformContractError, match="not an instance"):
        perturb(make_profile(), SwapCandidates)


def test_non_callable_transform_raises():
    with pytest.raises(TransformContractError, match="per-voter transform function"):
        perturb(make_profile(), "reverse")


@pytest.mark.parametrize(
    "ranking_fn, match",
    [
        (lambda ranking: "AB", "1D array of length 3"),
        (lambda ranking: None, "1D array of length 3"),
        (lambda ranking: np.stack([ranking] * 3), "1D array of length 3"),
        (lambda ranking: np.concatenate([ranking, ranking]), "1D array of length 3"),
        (lambda ranking: ranking[:2], "1D array of length 3"),
        (lambda ranking: ["A", "B", "C"], "sets or frozensets of candidates"),
        (lambda ranking: [frozenset({"Z"}), *ranking[1:]], "Declare them"),
    ],
)
def test_invalid_ranking_raises(ranking_fn, match):
    with pytest.raises(TransformContractError, match=match):
        perturb(make_profile(), ranking_fn)


def test_invalid_ranking_error_names_ballot_index():
    def break_second_ballot(ranking):
        if ranking[1] == {"C"}:
            ranking[0] = "A"
        return ranking

    def select_second_ballot(ranking):
        return 1.0 if ranking[1] == {"C"} else 0.0

    # only Ballot Index 1 is selected, so it is row 0 of the matrix the transform receives
    with pytest.raises(TransformContractError, match="for Ballot Index 1"):
        perturb(make_profile(), break_second_ballot, select_second_ballot)


def test_undeclared_candidate_error_names_ballot_index():
    def add_z_to_second_ballot(ranking):
        if ranking[1] == {"C"}:
            ranking[2] = frozenset({"Z"})
        return ranking

    with pytest.raises(TransformContractError, match=r"\{'Z'\}.*for Ballot Index 1"):
        perturb(make_profile(), add_z_to_second_ballot)


def test_profile_transform_must_conserve_weight():
    class DropVoters:
        def transform_matrix(self, matrix):
            keep_first = np.zeros(len(matrix), dtype=bool)
            keep_first[0] = True
            return matrix.apply_mask(keep_first)

    with pytest.raises(TransformContractError, match="conserve weight"):
        perturb(make_profile(), DropVoters())


def test_voter_sets_are_split_between_perturbed_and_unperturbed():
    voters = {f"v{i}" for i in range(10)}
    profile = RankProfile(ballots=[RankBallot(ranking=[{"A"}, {"B"}], weight=10, voter_set=voters)])
    perturbed = perturb(profile, reverse, 0.5, rng_seed=3)
    df = perturbed.to_df()

    unit_rows = df[df["Weight"] == 1]
    assert all(len(voter_set) == 1 for voter_set in unit_rows["Voter Set"])
    assert all(len(vs) == w for vs, w in zip(df["Voter Set"], df["Weight"]))
    assert set().union(*df["Voter Set"]) == voters


@pytest.mark.parametrize("probability", [0.0, 1.0])
def test_voter_sets_when_no_or_all_weight_is_selected(probability):
    voters = {"v1", "v2", "v3"}
    profile = RankProfile(ballots=[RankBallot(ranking=[{"A"}, {"B"}], weight=3, voter_set=voters)])
    df = perturb(profile, reverse, probability, rng_seed=0).to_df()

    if probability == 0.0:
        # nothing selected: the row and its voters are untouched
        assert list(df["Weight"]) == [3.0]
        assert df["Voter Set"].iloc[0] == voters
    else:
        # everything selected: no unperturbed row, one voter per unit row
        assert list(df["Weight"]) == [1.0, 1.0, 1.0]
        assert set().union(*df["Voter Set"]) == voters


@pytest.mark.parametrize("probability", [0.0, 0.5, 1.0])
def test_voter_set_size_mismatch_raises_for_any_probability(probability):
    profile = RankProfile(
        ballots=[RankBallot(ranking=[{"A"}, {"B"}], weight=3, voter_set={"v1", "v2"})]
    )

    with pytest.raises(ProfileError, match="voter set of size 2 but weight 3"):
        perturb(profile, reverse, probability, rng_seed=0)


def make_score_profile() -> ScoreProfile:
    df = pd.DataFrame(
        {"A": [2.0, np.nan], "B": [0.0, 1.0], "Voter Set": [set(), set()], "Weight": [4.0, 2.0]}
    )
    df.index.name = "Ballot Index"
    return ScoreProfile(df=df, candidates=["A", "B"])


def test_score_function_can_modify_array_in_place():
    def double(scores):
        assert isinstance(scores, np.ndarray)
        scores *= 2
        return scores

    profile = make_score_profile()
    perturbed = perturb(profile, double, group_ballots=True)

    assert perturbed.to_df().set_index("Weight").loc[4.0, "A"] == 4.0
    assert profile.to_df().set_index("Weight").loc[4.0, "A"] == 2.0


def test_score_function_preserves_zero_and_nan():
    perturbed = perturb(
        make_score_profile(), lambda scores: tuple(s * 2 for s in scores), group_ballots=True
    )
    df = perturbed.to_df().set_index("Weight")

    assert df.loc[4.0, "A"] == 4.0
    assert df.loc[4.0, "B"] == 0.0
    assert np.isnan(df.loc[2.0, "A"])
    assert df.loc[2.0, "B"] == 2.0


@pytest.mark.parametrize(
    "scores_fn, match",
    [
        (lambda scores: scores[:1], "1D array of length 2"),
        (lambda scores: np.stack([scores, scores]), "1D array of length 2"),
        (lambda scores: (-1.0, 1.0), "negative score"),
        (lambda scores: ("high", 1.0), "non-numeric score"),
        (lambda scores: np.array([True, False]), "non-numeric score"),
    ],
)
def test_invalid_scores_raise(scores_fn, match):
    with pytest.raises(TransformContractError, match=match):
        perturb(make_score_profile(), scores_fn)


@pytest.mark.parametrize("width_change", [-1, 1])
def test_profile_transform_must_keep_ranking_columns(width_change):
    class ChangeWidth:
        def transform_matrix(self, matrix):
            width = matrix.num_ranking_positions + width_change
            rankings = np.full((len(matrix), width), -1, dtype=np.intp)
            keep = min(width, matrix.num_ranking_positions)
            rankings[:, :keep] = matrix.rankings[:, :keep]
            return replace(
                matrix,
                rankings=rankings,
                ranking_columns=tuple(f"Ranking_{i + 1}" for i in range(width)),
            )

    with pytest.raises(TransformContractError, match="must keep the ranking columns"):
        perturb(make_profile(), ChangeWidth())


def test_profile_transform_must_keep_score_columns():
    class ReorderColumns:
        def transform_matrix(self, matrix):
            return ScoreMatrix(
                voter_weight_data=matrix.voter_weight_data,
                scores=matrix.scores[:, ::-1],
                score_columns=matrix.score_columns[::-1],
                candidates=matrix.candidates,
            )

    with pytest.raises(TransformContractError, match="must keep the score columns"):
        perturb(make_score_profile(), ReorderColumns())


def test_score_profile_from_ballots():
    profile = ScoreProfile(
        ballots=[ScoreBallot(scores={"A": 1, "B": 3}, weight=5)], candidates=["A", "B"]
    )
    perturbed = perturb(profile, lambda scores: (scores[1], scores[0]), 0.4, rng_seed=0)

    assert perturbed.total_ballot_wt == 5.0
    assert set(map(tuple, perturbed.to_df()[["A", "B"]].to_numpy())) <= {(1.0, 3.0), (3.0, 1.0)}


@pytest.mark.parametrize("probability", [0.0, 0.5, 1.0])
def test_fraction_weights_stay_exact(probability):
    profile = RankProfile(
        ballots=[RankBallot(ranking=[{"A"}, {"B"}], weight=Fraction(4))],
    )
    perturbed = perturb(profile, reverse, probability, rng_seed=0)

    assert perturbed._matrix.voter_weight_data.has_fraction_weights
    assert sum(perturbed.to_df()["Weight"]) == Fraction(4)


def test_profile_transform_receives_float_weights_for_fraction_profile():
    received = []

    class Record:
        def transform_matrix(self, matrix):
            received.append(matrix.voter_weight_data.weights.dtype)
            return matrix

    profile = RankProfile(ballots=[RankBallot(ranking=[{"A"}, {"B"}], weight=Fraction(4))])
    perturbed = perturb(profile, Record())

    assert received == [np.float64]
    assert perturbed._matrix.voter_weight_data.has_fraction_weights


def test_profile_transform_must_return_same_matrix_kind():
    class ReturnsScores:
        def transform_matrix(self, matrix):
            return make_score_profile()._matrix

    with pytest.raises(TransformContractError, match="must return a RankMatrix"):
        perturb(make_profile(), ReturnsScores())


def test_ranking_function_tie_exceeding_max_ranking_length_raises():
    profile = RankProfile(
        ballots=[RankBallot(ranking=[{"A"}, {"B"}])],
        candidates=["A", "B", "C"],
        max_ranking_length=2,
    )

    with pytest.raises(TransformContractError, match="3 candidates, more than"):
        perturb(profile, lambda ranking: (frozenset({"A", "B", "C"}), frozenset({"~"})))


def adjacent_swap_profile(num_voters: int = 3000) -> RankProfile:
    return RankProfile(ballots=[RankBallot(ranking=[{"A"}, {"B"}, {"C"}], weight=num_voters)])


def test_adjacent_swap_only_swaps_neighbors():
    transform = SwapAdjacentRankPositions(rng_seed=0)
    perturbed = perturb(adjacent_swap_profile(), transform, group_ballots=True)

    assert set(ranking_weights(perturbed)) <= {("A", "B", "C"), ("B", "A", "C"), ("A", "C", "B")}
    assert perturbed.total_ballot_wt == 3000


def test_adjacent_swap_follows_jitter_distribution():
    # position 0 only swaps right, position 2 only swaps left, position 1 swaps either way,
    # so each adjacent pair and no swap each occur with probability 1/3
    transform = SwapAdjacentRankPositions(rng_seed=1)
    weights = ranking_weights(perturb(adjacent_swap_profile(), transform, group_ballots=True))

    for ranking in [("A", "B", "C"), ("B", "A", "C"), ("A", "C", "B")]:
        assert weights[ranking] / 3000 == pytest.approx(1 / 3, abs=0.03)


def test_adjacent_swap_never_swaps_into_padding():
    profile = RankProfile(
        ballots=[RankBallot(ranking=[{"A"}, {"B"}], weight=500)],
        candidates=["A", "B", "C", "D"],
        max_ranking_length=4,
    )
    transform = SwapAdjacentRankPositions(rng_seed=2)
    weights = ranking_weights(perturb(profile, transform, group_ballots=True))

    assert set(weights) == {("A", "B"), ("B", "A")}


def test_adjacent_swap_is_reproducible_with_seed():
    def run():
        transform = SwapAdjacentRankPositions(rng_seed=5)
        return perturb(adjacent_swap_profile(100), transform, 0.5, rng_seed=5)

    assert ranking_weights(run()) == ranking_weights(run())


def test_adjacent_swap_reused_transform_continues_its_stream():
    transform = SwapAdjacentRankPositions(rng_seed=5)
    first = perturb(adjacent_swap_profile(100), transform, rng_seed=5)
    second = perturb(adjacent_swap_profile(100), transform, rng_seed=5)

    assert ranking_weights(first) != ranking_weights(second)


def test_adjacent_swap_default_generator():
    perturbed = perturb(adjacent_swap_profile(100), SwapAdjacentRankPositions())

    assert perturbed.total_ballot_wt == 100
