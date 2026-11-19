import re
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
from votekit.pref_profile.profile_transform import (
    JitterScores,
    SwapAdjacentRankPositions,
    SwapCandidates,
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
    perturbed = make_profile().perturb(SwapCandidates("A", "B"))

    assert ranking_weights(perturbed) == {("B", "A", "C"): 10.0, ("B", "C", "A"): 10.0}


def test_swap_candidates_with_probability():
    perturbed = make_profile().perturb(SwapCandidates("A", "B"), 0.5, rng_seed=42)
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
        ([{"A", "C"}, {"B"}], (frozenset({"B", "C"}), frozenset({"A"}))),
        ([{"A", "C"}, {"D"}], (frozenset({"B", "C"}), frozenset({"D"}))),
        ([{"A"}, {"C"}], (frozenset({"B"}), frozenset({"C"}))),
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
    first = make_profile().perturb(reverse, 0.5, rng_seed=7)
    second = make_profile().perturb(reverse, 0.5, rng_seed=7)

    assert ranking_weights(first) == ranking_weights(second)


def test_random_profile_transform_owns_its_randomness():
    def run():
        transform = SwapRandomAdjacentPositions(np.random.default_rng(11))
        return make_profile().perturb(transform, 0.5, rng_seed=7)

    assert ranking_weights(run()) == ranking_weights(run())


def test_zero_probability_leaves_profile_unchanged():
    perturbed = make_profile().perturb(reverse, 0.0, rng_seed=1)

    assert ranking_weights(perturbed) == ranking_weights(make_profile())


def test_partial_selection_keeps_profile_transform_rows_grouped():
    perturbed = make_profile().perturb(SwapCandidates("A", "B"), 0.5, rng_seed=42)
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

    make_profile().perturb(Record())

    assert received == [[10.0, 10.0]]


def test_per_voter_function_splits_selected_voters_into_unit_rows():
    perturbed = make_profile().perturb(reverse, 0.5, rng_seed=42)
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
    perturbed = profile.perturb(reverse)

    assert isinstance(perturbed, type(profile))
    assert perturbed.num_ballots == 0


def test_zero_weight_rows_are_kept():
    profile = RankProfile(
        ballots=[
            RankBallot(ranking=[{"A"}, {"B"}], weight=2),
            RankBallot(ranking=[{"B"}, {"A"}], weight=0),
        ]
    )
    perturbed = profile.perturb(reverse)

    # the exhausted weight 2 row is dropped, the zero weight row is kept
    assert sorted(perturbed.to_df()["Weight"]) == [0.0, 1.0, 1.0]
    assert ranking_weights(perturbed) == {("B", "A"): 2.0}


def test_group_ballots_regroups_output():
    perturbed = make_profile().perturb(SwapCandidates("A", "B"), group_ballots=True)

    assert len(perturbed.to_df()) == 2
    assert ranking_weights(perturbed) == {("B", "A", "C"): 10.0, ("B", "C", "A"): 10.0}


def test_ranking_function_is_called_once_per_selected_voter():
    calls = []

    def record(ranking):
        calls.append(ranking)
        return ranking

    make_profile().perturb(record)

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
    perturbed = profile.perturb(record)

    assert isinstance(received[0], np.ndarray)
    assert list(received[0]) == [frozenset({"A"}), frozenset({"B"}), frozenset({"~"})]
    assert ranking_weights(perturbed) == {("B", "A"): 1.0}


@pytest.mark.parametrize("container", [list, tuple, np.array])
def test_ranking_function_may_return_any_1d_container(container):
    def swap_first_two(ranking):
        return container([ranking[1], ranking[0], ranking[2]])

    perturbed = make_profile().perturb(swap_first_two, group_ballots=True)

    assert ranking_weights(perturbed) == {("B", "A", "C"): 10.0, ("C", "A", "B"): 10.0}


def test_ranking_function_may_return_sets():
    perturbed = make_profile().perturb(
        lambda ranking: [{"C"}, {"B"}, {"A"}, *ranking[3:]], group_ballots=True
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
    perturbed = profile.perturb(add_spoiler, group_ballots=True)

    assert ranking_weights(perturbed) == {("A", "B", "Spoiler"): 2.0}


def test_curried_ranking_function():
    def truncate(ranking, length):
        ranking[length:] = frozenset({"~"})
        return ranking

    perturbed = make_profile().perturb(partial(truncate, length=1), group_ballots=True)

    assert ranking_weights(perturbed) == {("A",): 20.0}


def test_probability_function_receives_read_only_array():
    def write(ranking):
        ranking[0] = frozenset({"B"})
        return 1.0

    with pytest.raises(ValueError, match="read-only"):
        make_profile().perturb(reverse, write)


def test_probability_function_filters_rows():
    perturbed = make_profile().perturb(
        reverse,
        lambda ranking: 1.0 if ranking[1] == {"B"} else 0.0,
        group_ballots=True,
    )

    assert ranking_weights(perturbed) == {("C", "B", "A"): 10.0, ("A", "C", "B"): 10.0}


@pytest.mark.parametrize("probability", [-0.1, 1.5, float("nan"), lambda ranking: 2.0])
def test_probability_out_of_range_raises(probability):
    with pytest.raises(ValueError, match="must lie in"):
        make_profile().perturb(reverse, probability)


def test_probability_not_numeric_raises():
    with pytest.raises(TypeError, match="real number"):
        make_profile().perturb(reverse, lambda ranking: "high")  # type: ignore[arg-type, ty:invalid-argument-type]


def test_non_integral_weights_raise():
    profile = RankProfile(ballots=[RankBallot(ranking=[{"A"}, {"B"}], weight=1.5)])

    with pytest.raises(ValueError, match="non-integral weights"):
        profile.perturb(reverse)


def test_class_instead_of_instance_raises():
    with pytest.raises(TransformContractError, match="not an instance"):
        make_profile().perturb(SwapCandidates)  # type: ignore[arg-type, ty:invalid-argument-type]


def test_non_callable_transform_raises():
    with pytest.raises(TransformContractError, match="per-voter transform function"):
        make_profile().perturb("reverse")  # type: ignore[arg-type, ty:invalid-argument-type]


@pytest.mark.parametrize(
    "ranking_fn, match",
    [
        (lambda ranking: "AB", "1D array of length 3"),
        (lambda ranking: None, "1D array of length 3"),
        (lambda ranking: np.stack([ranking] * 3), "1D array of length 3"),
        (lambda ranking: np.concatenate([ranking, ranking]), "1D array of length 3"),
        (lambda ranking: ranking[:2], "1D array of length 3"),
        (lambda ranking: ["A", "B", "C"], "sets or frozensets of candidates"),
        (
            lambda ranking: [frozenset({"A"}), ["B", "C"], frozenset({"~"})],
            "1D array, list, or tuple",
        ),
        (lambda ranking: [frozenset({"Z"}), *ranking[1:]], "Declare them"),
    ],
)
def test_invalid_ranking_raises(ranking_fn, match):
    with pytest.raises(TransformContractError, match=match):
        make_profile().perturb(ranking_fn)


def test_invalid_ranking_error_names_ballot_index():
    def break_second_ballot(ranking):
        if ranking[1] == {"C"}:
            ranking[0] = "A"
        return ranking

    def select_second_ballot(ranking):
        return 1.0 if ranking[1] == {"C"} else 0.0

    # only Ballot Index 1 is selected, so it is row 0 of the matrix the transform receives
    with pytest.raises(TransformContractError, match="for Ballot Index 1"):
        make_profile().perturb(break_second_ballot, select_second_ballot)


def test_undeclared_candidate_error_names_ballot_index():
    def add_z_to_second_ballot(ranking):
        if ranking[1] == {"C"}:
            ranking[2] = frozenset({"Z"})
        return ranking

    with pytest.raises(TransformContractError, match=r"\{'Z'\}.*for Ballot Index 1"):
        make_profile().perturb(add_z_to_second_ballot)


def make_labeled_profile(voter_sets=None) -> RankProfile:
    """Profile whose Ballot Index is 10 and 20, so labels differ from row positions."""
    df = pd.DataFrame(
        {
            "Ranking_1": [frozenset({"A"}), frozenset({"B"})],
            "Ranking_2": [frozenset({"B"}), frozenset({"A"})],
            "Voter Set": voter_sets or [frozenset(), frozenset()],
            "Weight": [2.0, 3.0],
        },
        index=pd.Index([10, 20], name="Ballot Index"),
    )
    return RankProfile(df=df, candidates=["A", "B"], max_ranking_length=2)


def test_errors_report_ballot_index_not_row_position():
    def break_ballot_starting_with_b(ranking):
        if ranking[0] == {"B"}:
            ranking[0] = "B"
        return ranking

    with pytest.raises(TransformContractError, match="for Ballot Index 20"):
        make_labeled_profile().perturb(break_ballot_starting_with_b)


def test_probability_error_shows_the_row():
    row = "[frozenset({'A'}) frozenset({'B'})]"

    with pytest.raises(ValueError, match=re.escape(f"per_voter_probability({row}) must lie in")):
        make_labeled_profile().perturb(reverse, lambda ranking: 2.0)


def test_voter_set_mismatch_reports_ballot_index():
    voter_sets = [frozenset({"v1", "v2"}), frozenset({"v3"})]

    with pytest.raises(ProfileError, match="Ballot Index 20 has a voter set of size 1"):
        make_labeled_profile(voter_sets).perturb(reverse)


def test_profile_transform_must_conserve_weight():
    class DropVoters:
        def transform_matrix(self, matrix):
            keep_first = np.zeros(len(matrix), dtype=bool)
            keep_first[0] = True
            return matrix.apply_mask(keep_first)

    with pytest.raises(TransformContractError, match="conserve weight"):
        make_profile().perturb(DropVoters())


def test_voter_sets_are_split_between_perturbed_and_unperturbed():
    voters = {f"v{i}" for i in range(10)}
    profile = RankProfile(ballots=[RankBallot(ranking=[{"A"}, {"B"}], weight=10, voter_set=voters)])
    perturbed = profile.perturb(reverse, 0.5, rng_seed=3)
    df = perturbed.to_df()

    unit_rows = df[df["Weight"] == 1]
    assert all(len(voter_set) == 1 for voter_set in unit_rows["Voter Set"])
    assert all(len(vs) == w for vs, w in zip(df["Voter Set"], df["Weight"]))
    assert set().union(*df["Voter Set"]) == voters


@pytest.mark.parametrize("probability", [0.0, 1.0])
def test_voter_sets_when_no_or_all_weight_is_selected(probability):
    voters = {"v1", "v2", "v3"}
    profile = RankProfile(ballots=[RankBallot(ranking=[{"A"}, {"B"}], weight=3, voter_set=voters)])
    df = profile.perturb(reverse, probability, rng_seed=0).to_df()

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
        profile.perturb(reverse, probability, rng_seed=0)


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
    perturbed = profile.perturb(double, group_ballots=True)

    perturbed_df, profile_df = perturbed.to_df(), profile.to_df()
    assert perturbed_df.loc[perturbed_df["Weight"] == 4.0, "A"].item() == 4.0
    assert profile_df.loc[profile_df["Weight"] == 4.0, "A"].item() == 2.0


def test_score_function_preserves_zero_and_nan():
    perturbed = make_score_profile().perturb(
        lambda scores: tuple(s * 2 for s in scores), group_ballots=True
    )
    df = perturbed.to_df()
    weight_4, weight_2 = df["Weight"] == 4.0, df["Weight"] == 2.0

    assert df.loc[weight_4, "A"].item() == 4.0
    assert df.loc[weight_4, "B"].item() == 0.0
    assert np.isnan(df.loc[weight_2, "A"].item())
    assert df.loc[weight_2, "B"].item() == 2.0


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
        make_score_profile().perturb(scores_fn)


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
        make_profile().perturb(ChangeWidth())


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
        make_score_profile().perturb(ReorderColumns())


def test_score_profile_from_ballots():
    profile = ScoreProfile(
        ballots=[ScoreBallot(scores={"A": 1, "B": 3}, weight=5)], candidates=["A", "B"]
    )
    perturbed = profile.perturb(lambda scores: (scores[1], scores[0]), 0.4, rng_seed=0)

    assert perturbed.total_ballot_wt == 5.0
    assert set(map(tuple, perturbed.to_df()[["A", "B"]].to_numpy())) <= {(1.0, 3.0), (3.0, 1.0)}


@pytest.mark.parametrize("probability", [0.0, 0.5, 1.0])
def test_fraction_weights_stay_exact(probability):
    profile = RankProfile(
        ballots=[RankBallot(ranking=[{"A"}, {"B"}], weight=Fraction(4))],
    )
    perturbed = profile.perturb(reverse, probability, rng_seed=0)

    assert perturbed._matrix.voter_weight_data.has_fraction_weights
    assert sum(perturbed.to_df()["Weight"]) == Fraction(4)


def test_profile_transform_receives_float_weights_for_fraction_profile():
    received = []

    class Record:
        def transform_matrix(self, matrix):
            received.append(matrix.voter_weight_data.weights.dtype)
            return matrix

    profile = RankProfile(ballots=[RankBallot(ranking=[{"A"}, {"B"}], weight=Fraction(4))])
    perturbed = profile.perturb(Record())

    assert received == [np.float64]
    assert perturbed._matrix.voter_weight_data.has_fraction_weights


def test_profile_transform_must_return_same_matrix_kind():
    class ReturnsScores:
        def transform_matrix(self, matrix):
            return make_score_profile()._matrix

    with pytest.raises(TransformContractError, match="must return a RankMatrix"):
        make_profile().perturb(ReturnsScores())  # type: ignore[arg-type]


def test_ranking_function_tie_exceeding_max_ranking_length_raises():
    profile = RankProfile(
        ballots=[RankBallot(ranking=[{"A"}, {"B"}])],
        candidates=["A", "B", "C"],
        max_ranking_length=2,
    )

    with pytest.raises(TransformContractError, match="3 candidates, more than"):
        profile.perturb(lambda ranking: (frozenset({"A", "B", "C"}), frozenset({"~"})))


def adjacent_swap_profile(num_voters: int = 3000) -> RankProfile:
    return RankProfile(ballots=[RankBallot(ranking=[{"A"}, {"B"}, {"C"}], weight=num_voters)])


def test_adjacent_swap_only_swaps_neighbors():
    transform = SwapAdjacentRankPositions(rng_seed=0)
    perturbed = adjacent_swap_profile().perturb(transform, group_ballots=True)

    assert set(ranking_weights(perturbed)) <= {("A", "B", "C"), ("B", "A", "C"), ("A", "C", "B")}
    assert perturbed.total_ballot_wt == 3000


def test_adjacent_swap_follows_jitter_distribution():
    # position 0 only swaps right, position 2 only swaps left, position 1 swaps either way,
    # so each adjacent pair and no swap each occur with probability 1/3
    transform = SwapAdjacentRankPositions(rng_seed=1)
    weights = ranking_weights(adjacent_swap_profile().perturb(transform, group_ballots=True))

    for ranking in [("A", "B", "C"), ("B", "A", "C"), ("A", "C", "B")]:
        assert weights[ranking] / 3000 == pytest.approx(1 / 3, abs=0.03)


def test_adjacent_swap_never_swaps_into_padding():
    profile = RankProfile(
        ballots=[RankBallot(ranking=[{"A"}, {"B"}], weight=500)],
        candidates=["A", "B", "C", "D"],
        max_ranking_length=4,
    )
    transform = SwapAdjacentRankPositions(rng_seed=2)
    weights = ranking_weights(profile.perturb(transform, group_ballots=True))

    assert set(weights) == {("A", "B"), ("B", "A")}


def test_adjacent_swap_is_reproducible_with_seed():
    def run():
        transform = SwapAdjacentRankPositions(rng_seed=5)
        return adjacent_swap_profile(100).perturb(transform, 0.5, rng_seed=5)

    assert ranking_weights(run()) == ranking_weights(run())


def test_adjacent_swap_reused_transform_continues_its_stream():
    transform = SwapAdjacentRankPositions(rng_seed=5)
    first = adjacent_swap_profile(100).perturb(transform, rng_seed=5)
    second = adjacent_swap_profile(100).perturb(transform, rng_seed=5)

    assert ranking_weights(first) != ranking_weights(second)


def test_adjacent_swap_default_generator():
    perturbed = adjacent_swap_profile(100).perturb(SwapAdjacentRankPositions())

    assert perturbed.total_ballot_wt == 100


def jitter_profile(a: float, b: float, num_voters: int) -> ScoreProfile:
    """One ballot type scoring A and B, with an undeclared Z column left unscored."""
    df = pd.DataFrame(
        {"A": [a], "B": [b], "Z": [np.nan], "Voter Set": [set()], "Weight": [float(num_voters)]}
    )
    df.index.name = "Ballot Index"
    return ScoreProfile(df=df, candidates=["A", "B"])


def score_weights(profile: ScoreProfile) -> dict[tuple[float | None, float | None], float]:
    """Map each (A, B) score pair to its total weight, with NaN shown as None."""
    weights: dict[tuple[float | None, float | None], float] = {}
    for _, row in profile.to_df().iterrows():
        a, b = (None if np.isnan(row[c]) else float(row[c]) for c in ("A", "B"))
        weights[(a, b)] = weights.get((a, b), 0.0) + row["Weight"]
    return weights


def test_jitter_scores_moves_one_declared_score_by_step():
    perturbed = jitter_profile(2.0, 1.0, 500).perturb(JitterScores(rng_seed=0))
    df = perturbed.to_df()

    changes = np.abs(df[["A", "B"]].to_numpy() - np.array([2.0, 1.0]))
    assert (changes.sum(axis=1) == 1.0).all()
    assert df["Z"].isna().all()
    assert perturbed.total_ballot_wt == 500


def test_jitter_scores_follows_jitter_distribution():
    num_voters = 4000
    weights = score_weights(
        jitter_profile(2.0, 1.0, num_voters).perturb(JitterScores(rng_seed=1), group_ballots=True)
    )

    for scores in [(3.0, 1.0), (1.0, 1.0), (2.0, 2.0), (2.0, 0.0)]:
        assert weights[scores] / num_voters == pytest.approx(1 / 4, abs=0.03)


def test_jitter_scores_keeps_unscored_and_stops_at_zero():
    num_voters = 4000
    weights = score_weights(
        jitter_profile(np.nan, 0.0, num_voters).perturb(
            JitterScores(rng_seed=2), group_ballots=True
        )
    )

    # A stays unscored; B only rises when it is drawn and moves up, otherwise it stays at 0
    assert set(weights) == {(None, 0.0), (None, 1.0)}
    assert weights[(None, 1.0)] / num_voters == pytest.approx(1 / 4, abs=0.03)


def test_jitter_scores_stops_at_max_score():
    weights = score_weights(
        jitter_profile(3.0, 3.0, 1000).perturb(
            JitterScores(rng_seed=3, max_score=3.0), group_ballots=True
        )
    )

    assert set(weights) == {(3.0, 3.0), (2.0, 3.0), (3.0, 2.0)}


def test_jitter_scores_rejects_scores_above_max_score():
    profile = jitter_profile(4.0, 1.0, 10)

    with pytest.raises(
        ValueError, match="max_score 3.0 is below max score given in the matrix, 4.0"
    ):
        profile.perturb(JitterScores(rng_seed=0, max_score=3.0))


def test_jitter_scores_is_reproducible_and_continues_its_stream():
    def run(transform):
        return score_weights(jitter_profile(2.0, 1.0, 200).perturb(transform, group_ballots=True))

    assert run(JitterScores(rng_seed=4)) == run(JitterScores(rng_seed=4))

    reused = JitterScores(rng_seed=4)
    assert run(reused) != run(reused)


@pytest.mark.parametrize("kwargs, match", [({"step": 0}, "step"), ({"max_score": -1}, "max_score")])
def test_jitter_scores_rejects_invalid_settings(kwargs, match):
    with pytest.raises(ValueError, match=match):
        JitterScores(**kwargs)
