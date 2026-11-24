import re
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from votekit.ballot import RankBallot, ScoreBallot
from votekit.constants import END_OF_RANKING_SET
from votekit.exceptions import TransformContractError
from votekit.pref_profile import ProfileError, RankProfile, ScoreProfile
from votekit.pref_profile.profile_matrix import (
    ProfileVoterWeightData,
    RankMatrix,
    ScoreMatrix,
    WeightKind,
)
from votekit.pref_profile.profile_transform import (
    SwapAdjacentRankPositions,
    SwapCandidates,
)
from votekit.types import Candidate


@pytest.fixture
def rank_profile() -> RankProfile:
    return RankProfile(
        ballots=[
            RankBallot(ranking=[{"A"}, {"B"}, {"C"}], weight=2),
            RankBallot(ranking=[{"B"}, {"A"}], weight=1),
        ],
        candidates=["A", "B", "C"],
        max_ranking_length=3,
    )


@pytest.fixture
def labeled_profile() -> RankProfile:
    """Ballots A > B and B > A whose Ballot Index is 10 and 20, not their row positions 0 and 1."""
    df = pd.DataFrame(
        {
            "Ranking_1": [frozenset({"A"}), frozenset({"B"})],
            "Ranking_2": [frozenset({"B"}), frozenset({"A"})],
            "Voter Set": [frozenset(), frozenset()],
            "Weight": [2.0, 1.0],
        },
        index=pd.Index([10, 20], name="Ballot Index"),
    )
    return RankProfile(df=df, candidates=["A", "B"], max_ranking_length=2)


@pytest.fixture
def score_profile() -> ScoreProfile:
    return ScoreProfile(ballots=[ScoreBallot(scores={"A": 1, "B": 2})], candidates=["A", "B"])


@pytest.fixture
def undeclared_column_profile() -> ScoreProfile:
    """Scores A 1 and B 2, with an unscored column Z that is not a declared candidate."""
    df = pd.DataFrame(
        {"A": [1.0], "B": [2.0], "Z": [np.nan], "Voter Set": [set()], "Weight": [1.0]}
    )
    df.index.name = "Ballot Index"
    return ScoreProfile(df=df, candidates=["A", "B"])


@pytest.fixture
def voter_profile() -> RankProfile:
    """A > B for voters v1 and v2, and B > A for voter v3."""
    return RankProfile(
        ballots=[
            RankBallot(ranking=[{"A"}, {"B"}], weight=2, voter_set={"v1", "v2"}),
            RankBallot(ranking=[{"B"}, {"A"}], weight=1, voter_set={"v3"}),
        ]
    )


@pytest.fixture
def non_integral_profile() -> RankProfile:
    return RankProfile(ballots=[RankBallot(ranking=[{"A"}, {"B"}], weight=1.5)])


class ReturnsVoterData:
    """A ProfileTransform that returns the selected rows with the given weights and voter sets."""

    def __init__(self, weights: list[float], voter_sets: list[frozenset[str]] | None = None):
        self.weights = weights
        self.voter_sets = voter_sets

    def transform_matrix(self, matrix, rng):
        voter_sets = None
        if self.voter_sets is not None:
            voter_sets = np.empty(len(self.voter_sets), dtype=object)
            voter_sets[:] = self.voter_sets
        data = ProfileVoterWeightData(
            weights=np.array(self.weights, dtype=np.float64),
            voter_sets=voter_sets,
            weight_kind=WeightKind.FLOAT,
        )
        return replace(matrix, voter_weight_data=data)


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


def score_weights(profile: ScoreProfile) -> dict[tuple[float | None, ...], float]:
    """Total weight of each (A, B) score pair, with unscored (NaN) shown as None."""
    weights: dict[tuple[float | None, ...], float] = {}
    for _, row in profile.to_df().iterrows():
        scores = tuple(None if np.isnan(row[c]) else float(row[c]) for c in ("A", "B"))
        weights[scores] = weights.get(scores, 0.0) + float(row["Weight"])
    return weights


def reverse(row, rng):
    """A row transform function that reverses the ranked positions, leaving padding at the end."""
    num_ranked = sum(cand_set != END_OF_RANKING_SET for cand_set in row)
    row[:num_ranked] = row[:num_ranked][::-1]
    return row


def test_transform_rejects_return_of_diff_matrix_type(rank_profile):
    score_matrix = ScoreProfile(ballots=[ScoreBallot(scores={"A": 1})], candidates=["A"])._matrix

    class ReturnsScoreMatrix:
        def transform_matrix(self, matrix, rng):
            return score_matrix

    with pytest.raises(TransformContractError, match="profile_transform must return a RankMatrix"):
        rank_profile.transform(ReturnsScoreMatrix())


def test_transform_accepts_a_change_in_total_weight(rank_profile):
    class DropsSecondRow:
        def transform_matrix(self, matrix, rng):
            return matrix.apply_mask(np.array([True, False]))

    transformed = rank_profile.transform(DropsSecondRow())

    assert ranking_weights(transformed) == {ranking("A", "B", "C"): 2.0}


def test_transform_accepts_non_integral_weights(rank_profile):
    transformed = rank_profile.transform(ReturnsVoterData([1.5, 0.25]))

    assert transformed.to_df()["Weight"].tolist() == [1.5, 0.25]


def test_transform_accepts_changed_voter_sets(voter_profile):
    voter_sets = [frozenset({"v1", "v9"}), frozenset({"v1"})]

    transformed = voter_profile.transform(ReturnsVoterData([2, 1], voter_sets))

    assert transformed.to_df()["Voter Set"].tolist() == voter_sets


class DropsLastColumn:
    """A ProfileTransform that drops the last ranking position."""

    def transform_matrix(self, matrix, rng):
        return replace(
            matrix,
            rankings=matrix.rankings[:, :-1],
            ranking_columns=matrix.ranking_columns[:-1],
        )


def only_ballots_starting_with_b(row):
    """A probability function that selects only the voters whose ballot starts with B."""
    return 1.0 if row[0] == {"B"} else 0.0


def test_transform_at_probability_one_follows_the_returned_ranking_length(rank_profile):
    transformed = rank_profile.transform(DropsLastColumn())

    assert transformed.max_ranking_length == 2
    assert ranking_weights(transformed) == {ranking("A", "B"): 2.0, ranking("B", "A"): 1.0}


def test_transform_keeps_the_ranking_length_of_unselected_voters(rank_profile):
    transformed = rank_profile.transform(DropsLastColumn(), only_ballots_starting_with_b)

    assert transformed.max_ranking_length == 3
    assert ranking_weights(transformed) == {ranking("A", "B", "C"): 2.0, ranking("B", "A"): 1.0}


def test_transform_widens_for_more_ranking_columns_and_adds_their_candidates(rank_profile):
    class AppendsD:
        def transform_matrix(self, matrix, rng):
            rows = np.column_stack(
                [matrix.get_ranking_array(), np.full(len(matrix), frozenset({"D"}))]
            )
            return RankMatrix.from_array(
                rows,
                voter_weight_data=matrix.voter_weight_data,
                ranking_columns=[*matrix.ranking_columns, "Ranking_4"],
                candidates=[*matrix.candidates, "D"],
            )

    transformed = rank_profile.transform(AppendsD(), 0.5, rng=0)

    assert transformed.max_ranking_length == 4
    assert transformed.candidates == ("A", "B", "C", "D")
    assert set(transformed.to_df()["Ranking_4"]) == {END_OF_RANKING_SET, frozenset({"D"})}


def test_transform_keeps_a_candidate_it_drops_from_the_matrix(rank_profile):
    class DropsCandidateC:
        def transform_matrix(self, matrix, rng):
            return replace(
                matrix.apply_mask(np.array([False, True])),
                candidates=("A", "B"),
            )

    transformed = rank_profile.transform(DropsCandidateC())

    assert transformed.candidates == ("A", "B", "C")


def test_transform_aligns_reordered_score_columns_by_candidate(score_profile):
    class ReversesColumns:
        def transform_matrix(self, matrix, rng):
            return ScoreMatrix(
                voter_weight_data=matrix.voter_weight_data,
                scores=matrix.scores[:, ::-1],
                score_columns=matrix.score_columns[::-1],
                candidates=matrix.candidates,
            )

    transformed = score_profile.transform(ReversesColumns())

    assert score_weights(transformed) == {(1.0, 2.0): 1.0}


def test_transform_adds_a_new_score_column_and_its_candidate():
    profile = ScoreProfile(
        ballots=[ScoreBallot(scores={"A": 1, "B": 2}, weight=4)], candidates=["A", "B"]
    )

    class ScoresC:
        def transform_matrix(self, matrix, rng):
            return ScoreMatrix(
                voter_weight_data=matrix.voter_weight_data,
                scores=np.column_stack([matrix.scores, np.full(len(matrix), 3.0)]),
                score_columns=(*matrix.score_columns, "C"),
                candidates=(*matrix.candidates, "C"),
            )

    df = profile.transform(ScoresC(), 0.5, rng=0).to_df()

    assert list(df.columns[:3]) == ["A", "B", "C"]
    assert df["C"].isna().tolist() == [True, False]
    assert profile.transform(ScoresC()).candidates == ("A", "B", "C")


def test_transform_rejects_transform_class_instead_of_an_instance(rank_profile):
    with pytest.raises(TransformContractError, match="profile_transform was given the class"):
        rank_profile.transform(SwapCandidates)


def test_transform_rejects_a_function(rank_profile):
    with pytest.raises(TransformContractError, match="use map"):
        rank_profile.transform(reverse)


def test_map_rejects_a_profile_transform(rank_profile):
    with pytest.raises(TransformContractError, match="Pass it to transform"):
        rank_profile.map(SwapCandidates("A", "B"))


def test_map_rejects_non_callable_row_transform_fn(rank_profile):
    with pytest.raises(
        TransformContractError, match="row_transform_fn must be a function that receives"
    ):
        rank_profile.map("reverse")


def test_map_rejects_an_unknown_mode(rank_profile):
    with pytest.raises(ValueError, match='mode must be "row" or "voter"'):
        rank_profile.map(reverse, mode="ballot")


def test_ranking_function_rejects_return_of_non_set_entries(rank_profile):
    with pytest.raises(
        TransformContractError,
        match="row_transform_fn must return an array of sets or frozensets of candidates",
    ):
        rank_profile.map(lambda row, rng: ["A", "B", "C"])


def test_ranking_function_rejects_return_of_a_2d_array(rank_profile):
    with pytest.raises(TransformContractError, match="row_transform_fn must return a 1D array"):
        rank_profile.map(lambda row, rng: np.stack([row, row]))


def test_ranking_function_rejects_padding_before_a_ranked_position(rank_profile):
    with pytest.raises(TransformContractError, match="only at the end of a ranking"):
        rank_profile.map(lambda row, rng: [END_OF_RANKING_SET, frozenset({"A"}), frozenset({"B"})])


def test_ranking_function_rejects_an_empty_position_after_padding(rank_profile):
    with pytest.raises(TransformContractError, match="only at the end of a ranking"):
        rank_profile.map(lambda row, rng: [frozenset({"A"}), END_OF_RANKING_SET, frozenset()])


def test_ranking_function_rejects_tilde_inside_a_candidate_set(rank_profile):
    with pytest.raises(TransformContractError, match="only use '~' as the padding"):
        rank_profile.map(
            lambda row, rng: [frozenset({"A", "~"}), END_OF_RANKING_SET, END_OF_RANKING_SET]
        )


def test_ranking_function_errors_show_the_ballot(labeled_profile):
    def unwrap_first_place_cand_B(row, rng):
        if row[0] == {"B"}:
            row[0] = "B"
        return row

    ballot = "(frozenset({'B'}), frozenset({'A'}))"
    with pytest.raises(TransformContractError, match=re.escape(f"for the ballot {ballot}")):
        labeled_profile.map(unwrap_first_place_cand_B)


def test_ranking_function_at_probability_one_may_shorten_the_rankings(rank_profile):
    transformed = rank_profile.map(lambda row, rng: row[:1])

    assert transformed.max_ranking_length == 1
    assert ranking_weights(transformed) == {ranking("A"): 2.0, ranking("B"): 1.0}


def test_ranking_function_pads_a_shorter_ranking_to_fit_unselected_voters(rank_profile):
    transformed = rank_profile.map(lambda row, rng: row[:1], only_ballots_starting_with_b)

    assert transformed.max_ranking_length == 3
    assert ranking_weights(transformed) == {ranking("A", "B", "C"): 2.0, ranking("B"): 1.0}


def test_ranking_function_widens_for_a_longer_ranking(rank_profile):
    transformed = rank_profile.map(lambda row, rng: [*row, frozenset({"D"})], 0.5, rng=0)

    assert transformed.max_ranking_length == 4
    assert ranking("A", "B", "C", "D") in ranking_weights(transformed)


def test_ranking_function_widens_for_a_tie_longer_than_max_ranking_length():
    profile = RankProfile(
        ballots=[RankBallot(ranking=[{"A"}, {"B"}])],
        candidates=["A", "B", "C"],
        max_ranking_length=2,
    )

    transformed = profile.map(lambda row, rng: [frozenset({"A", "B", "C"}), END_OF_RANKING_SET])

    assert transformed.max_ranking_length == 3
    assert ranking_weights(transformed) == {ranking({"A", "B", "C"}): 1.0}


def test_ranking_function_adds_a_new_candidate_after_the_profile_candidates(rank_profile):
    def add_z(row, rng):
        row[2] = frozenset({"Z"})
        return row

    transformed = rank_profile.map(add_z)

    assert transformed.candidates == ("A", "B", "C", "Z")


def test_ranking_function_can_rank_a_non_cast_candidate():
    def add_cand_to_cast(row, rng):
        row[2] = frozenset({"new_cand"})
        return row

    profile = RankProfile(
        ballots=[RankBallot(ranking=[{"A"}, {"B"}], weight=2)],
        candidates=["A", "B", "new_cand"],
        max_ranking_length=3,
    )
    transformed = profile.map(add_cand_to_cast)

    assert ranking_weights(transformed) == {ranking("A", "B", "new_cand"): 2.0}
    assert transformed.candidates == ("A", "B", "new_cand")


# map: row and voter modes


def test_row_mode_calls_once_per_row(rank_profile):
    calls = []

    def record(row, rng):
        calls.append(row)
        return row

    rank_profile.map(record)

    assert len(calls) == 2


def test_voter_mode_calls_once_per_voter(rank_profile):
    calls = []

    def record(row, rng):
        calls.append(row)
        return row

    rank_profile.map(record, mode="voter")

    assert len(calls) == 3


def test_row_mode_keeps_each_row_with_its_weight_and_voters(voter_profile):
    df = voter_profile.map(reverse).to_df()

    assert df["Weight"].tolist() == [2.0, 1.0]
    assert df["Voter Set"].tolist() == [frozenset({"v1", "v2"}), frozenset({"v3"})]


def test_voter_mode_gives_each_voter_its_own_row_with_voters_in_str_order():
    profile = RankProfile(
        ballots=[RankBallot(ranking=[{"A"}, {"B"}], weight=3, voter_set={"v2", "v10", "v1"})]
    )
    df = profile.map(lambda row, rng: row, mode="voter", group_ballots=False).to_df()

    assert df["Weight"].tolist() == [1.0, 1.0, 1.0]
    assert df["Voter Set"].tolist() == [frozenset({"v1"}), frozenset({"v10"}), frozenset({"v2"})]


def test_row_function_receives_a_writeable_copy(score_profile):
    def double(scores, rng):
        assert isinstance(scores, np.ndarray)
        scores *= 2
        return scores

    transformed = score_profile.map(double)

    assert score_weights(transformed) == {(2.0, 4.0): 1.0}
    assert score_weights(score_profile) == {(1.0, 2.0): 1.0}


# map: returned scores


def test_score_function_rejects_non_numeric_scores(score_profile):
    with pytest.raises(
        TransformContractError, match="row_transform_fn returned a non-numeric score"
    ):
        score_profile.map(lambda scores, rng: ("high", 1.0))


def test_score_function_rejects_a_negative_score(score_profile):
    with pytest.raises(TransformContractError, match="negative score"):
        score_profile.map(lambda scores, rng: (-1.0, 1.0))


def test_score_function_rejects_the_wrong_number_of_scores(score_profile):
    with pytest.raises(TransformContractError, match="1D array of length 2, one entry per score"):
        score_profile.map(lambda scores, rng: (1.0, 2.0, 3.0))


def test_score_function_adds_a_scored_undeclared_candidate(undeclared_column_profile):
    def score_z(scores, rng):
        scores[2] = 5.0
        return scores

    transformed = undeclared_column_profile.map(score_z)

    assert transformed.candidates == ("A", "B", "Z")
    assert transformed.to_df()["Z"].tolist() == [5.0]


def test_score_function_may_leave_an_undeclared_column_at_zero_or_unscored(
    undeclared_column_profile,
):
    def set_z_to(value):
        def transform(scores, rng):
            scores[2] = value
            return scores

        return transform

    zeroed = undeclared_column_profile.map(set_z_to(0.0))
    unscored = undeclared_column_profile.map(set_z_to(np.nan))

    assert zeroed.to_df()["Z"].tolist() == [0.0]
    assert zeroed.candidates == ("A", "B")
    assert unscored.to_df()["Z"].isna().all()


# selecting voters


def test_default_probability_maps_every_voter(rank_profile):
    transformed = rank_profile.map(reverse)

    assert ranking_weights(transformed) == {ranking("C", "B", "A"): 2.0, ranking("A", "B"): 1.0}


def test_zero_probability_leaves_every_ballot_unchanged(rank_profile):
    transformed = rank_profile.transform(SwapCandidates("A", "B"), 0.0, rng=0)

    pd.testing.assert_frame_equal(transformed.to_df(), rank_profile.to_df())


def test_zero_probability_leaves_a_score_profile_unchanged(score_profile):
    class DoublesScores:
        def transform_matrix(self, matrix, rng):
            return replace(matrix, scores=matrix.scores * 2)

    transformed = score_profile.transform(DoublesScores(), 0.0, rng=0)

    pd.testing.assert_frame_equal(transformed.to_df(), score_profile.to_df())


def test_probability_function_chooses_which_rows_to_map(rank_profile):
    transformed = rank_profile.map(reverse, only_ballots_starting_with_b)

    assert ranking_weights(transformed) == {ranking("A", "B", "C"): 2.0, ranking("A", "B"): 1.0}


def test_probability_rejects_a_non_real_number(rank_profile):
    with pytest.raises(TypeError, match="real number"):
        rank_profile.map(reverse, "high")


def test_probability_function_rejects_return_of_a_non_real_number(rank_profile):
    with pytest.raises(TypeError, match="real number"):
        rank_profile.map(reverse, lambda row: "high")


def test_probability_rejects_a_boolean(rank_profile):
    with pytest.raises(TypeError, match="real number"):
        rank_profile.map(reverse, True)
    with pytest.raises(TypeError, match="real number"):
        rank_profile.map(reverse, np.True_)


def test_probability_rejects_a_value_above_one(rank_profile):
    with pytest.raises(ValueError, match="must lie in"):
        rank_profile.map(reverse, 1.5)


def test_probability_rejects_a_value_below_zero(rank_profile):
    with pytest.raises(ValueError, match="must lie in"):
        rank_profile.transform(SwapCandidates("A", "B"), -0.1)


def test_probability_function_error_shows_the_row(labeled_profile):
    row = "[frozenset({'A'}) frozenset({'B'})]"

    with pytest.raises(ValueError, match=re.escape(f"per_voter_probability({row}) must lie in")):
        labeled_profile.map(reverse, lambda row: 2.0)


def test_probability_function_rejects_writing_to_its_row(rank_profile):
    def write_then_choose(row):
        row[0] = frozenset({"C"})
        return 1.0

    with pytest.raises(ValueError, match="read-only"):
        rank_profile.map(reverse, write_then_choose)


def test_row_mode_accepts_non_integral_weights_at_probability_zero_or_one(non_integral_profile):
    mapped = non_integral_profile.map(reverse)
    unchanged = non_integral_profile.map(reverse, 0.0)

    assert ranking_weights(mapped) == {ranking("B", "A"): 1.5}
    assert ranking_weights(unchanged) == {ranking("A", "B"): 1.5}


def test_transform_accepts_non_integral_weights_at_probability_one(non_integral_profile):
    transformed = non_integral_profile.transform(SwapCandidates("A", "B"))

    assert ranking_weights(transformed) == {ranking("B", "A"): 1.5}


def test_voter_mode_rejects_non_integral_weights_at_probability_one(non_integral_profile):
    with pytest.raises(ValueError, match="non-integral weights"):
        non_integral_profile.map(reverse, mode="voter")


def test_voter_mode_at_probability_zero_maps_no_voter_so_accepts_non_integral_weights(
    non_integral_profile,
):
    unchanged = non_integral_profile.map(reverse, 0.0, mode="voter")

    assert ranking_weights(unchanged) == {ranking("A", "B"): 1.5}


def test_sampling_voters_rejects_non_integral_weights(non_integral_profile):
    with pytest.raises(ValueError, match="non-integral weights"):
        non_integral_profile.map(reverse, 0.5)
    with pytest.raises(ValueError, match="non-integral weights"):
        non_integral_profile.transform(SwapCandidates("A", "B"), lambda row: 1.0)


def test_adjacent_swap_rejects_non_integral_weights_at_probability_one(non_integral_profile):
    with pytest.raises(ValueError, match="non-integral weights"):
        non_integral_profile.transform(SwapAdjacentRankPositions())


# voter sets


def test_sampling_rejects_voter_sets_not_matching_weights_even_when_none_are_selected():
    profile = RankProfile(
        ballots=[RankBallot(ranking=[{"A"}, {"B"}], weight=3, voter_set={"v1", "v2"})]
    )

    with pytest.raises(ProfileError, match="voter set of size 2 but weight 3"):
        profile.map(reverse, lambda row: 0.0)


def test_voter_mode_rejects_voter_sets_not_matching_weights_at_probability_one():
    profile = RankProfile(
        ballots=[RankBallot(ranking=[{"A"}, {"B"}], weight=3, voter_set={"v1", "v2"})]
    )

    with pytest.raises(ProfileError, match="voter set of size 2 but weight 3"):
        profile.map(reverse, mode="voter")


def test_row_mode_at_probability_one_keeps_voter_sets_not_matching_weights():
    profile = RankProfile(
        ballots=[RankBallot(ranking=[{"A"}, {"B"}], weight=3, voter_set={"v1", "v2"})]
    )

    df = profile.map(reverse).to_df()

    assert df["Voter Set"].tolist() == [frozenset({"v1", "v2"})]
    assert df["Weight"].tolist() == [3.0]


def test_voter_set_mismatch_shows_the_ballot():
    df = pd.DataFrame(
        {
            "Ranking_1": [frozenset({"A"}), frozenset({"B"})],
            "Voter Set": [frozenset({"v1"}), frozenset({"v2", "v3"})],
            "Weight": [1.0, 1.0],
        },
        index=pd.Index([10, 20], name="Ballot Index"),
    )
    profile = RankProfile(df=df, candidates=["A", "B"], max_ranking_length=1)

    with pytest.raises(
        ProfileError, match=re.escape("ballot (frozenset({'B'}),) has a voter set of size 2")
    ):
        profile.map(reverse, 0.5)


def test_sampling_splits_a_voter_set_into_disjoint_shares():
    voters = {"a", "b", "c", "d", "e", "f"}
    profile = RankProfile(ballots=[RankBallot(ranking=[{"A"}, {"B"}], weight=6, voter_set=voters)])

    df = profile.map(reverse, 0.5, rng=0).to_df()

    assert len(df) == 2
    assert df["Ranking_1"].tolist() == [frozenset({"A"}), frozenset({"B"})]
    unperturbed_voters, perturbed_voters = df["Voter Set"]
    assert unperturbed_voters | perturbed_voters == voters
    assert not unperturbed_voters & perturbed_voters
    assert [len(voter_set) for voter_set in df["Voter Set"]] == df["Weight"].tolist()


def test_selecting_no_voters_keeps_the_row_and_its_voters():
    profile = RankProfile(
        ballots=[RankBallot(ranking=[{"A"}, {"B"}], weight=2, voter_set={"v1", "v2"})]
    )
    df = profile.map(reverse, 0.0).to_df()

    assert df["Weight"].tolist() == [2.0]
    assert df["Voter Set"].tolist() == [frozenset({"v1", "v2"})]


def test_selecting_every_voter_leaves_no_unperturbed_row():
    profile = RankProfile(
        ballots=[RankBallot(ranking=[{"A"}, {"B"}], weight=2, voter_set={"v1", "v2"})]
    )
    df = profile.map(reverse, 1.0, mode="voter", group_ballots=False).to_df()

    assert df["Weight"].tolist() == [1.0, 1.0]
    assert df["Voter Set"].tolist() == [frozenset({"v1"}), frozenset({"v2"})]


def test_partial_selection_splits_a_row_into_kept_and_perturbed_rows():
    profile = RankProfile(ballots=[RankBallot(ranking=[{"A"}, {"B"}], weight=4)])
    df = profile.transform(SwapCandidates("A", "B"), 0.5, rng=0).to_df()

    assert df["Weight"].tolist() == [2.0, 2.0]
    assert df["Ranking_1"].tolist() == [frozenset({"A"}), frozenset({"B"})]


def test_sampling_keeps_zero_weight_rows_unselected():
    profile = RankProfile(
        ballots=[
            RankBallot(ranking=[{"A"}, {"B"}], weight=2),
            RankBallot(ranking=[{"B"}, {"A"}], weight=0),
        ]
    )
    df = profile.map(reverse, lambda row: 1.0, group_ballots=False).to_df()

    assert df["Weight"].tolist() == [0.0, 2.0]
    assert df["Ranking_1"].tolist() == [frozenset({"B"}), frozenset({"B"})]


def test_probability_one_maps_every_row_including_zero_weight_rows():
    profile = RankProfile(
        ballots=[
            RankBallot(ranking=[{"A"}, {"B"}], weight=1),
            RankBallot(ranking=[{"B"}, {"A"}], weight=0),
        ]
    )
    df = profile.map(reverse, group_ballots=False).to_df()

    assert df["Weight"].tolist() == [1.0, 0.0]
    assert df["Ranking_1"].tolist() == [frozenset({"B"}), frozenset({"A"})]


def test_transformed_profile_can_be_transformed_again(voter_profile):
    once = voter_profile.transform(SwapAdjacentRankPositions(), 0.5, rng=0)
    twice = once.map(reverse, 0.5, rng=1)

    assert twice.total_ballot_wt == 3
    assert set().union(*twice.to_df()["Voter Set"]) == {"v1", "v2", "v3"}


def test_output_is_grouped_by_ranking_by_default():
    profile = RankProfile(
        ballots=[
            RankBallot(ranking=[{"A"}, {"B"}], weight=1),
            RankBallot(ranking=[{"B"}, {"A"}], weight=1),
        ]
    )

    def a_over_b(row, rng):
        return [frozenset({"A"}), frozenset({"B"})]

    grouped = profile.map(a_over_b).to_df()
    ungrouped = profile.map(a_over_b, group_ballots=False).to_df()

    assert grouped["Weight"].tolist() == [2.0]
    assert ungrouped["Weight"].tolist() == [1.0, 1.0]


def test_fraction_weights_stay_fractions():
    profile = RankProfile(ballots=[RankBallot(ranking=[{"A"}, {"B"}], weight=Fraction(4))])

    for transformed in (profile.map(reverse), profile.map(reverse, 0.5, rng=0)):
        assert transformed._matrix.voter_weight_data.has_fraction_weights
        assert sum(transformed.to_df()["Weight"]) == Fraction(4)


def test_same_seed_gives_the_same_result(rank_profile):
    first = rank_profile.map(reverse, 0.5, rng=7)
    second = rank_profile.map(reverse, 0.5, rng=7)

    pd.testing.assert_frame_equal(first.to_df(), second.to_df())


def test_empty_profiles_are_returned_as_is():
    empty_rank, empty_score = RankProfile(), ScoreProfile()

    assert empty_rank.map(reverse) is empty_rank
    assert empty_rank.transform(SwapCandidates("A", "B")) is empty_rank
    assert empty_score.map(lambda row, rng: row) is empty_score


def test_zero_probability_returns_the_profile_without_checking_the_other_arguments(rank_profile):
    assert rank_profile.map("not a function", 0.0, mode="neither") is rank_profile
    assert rank_profile.transform("not a transform", 0) is rank_profile


def test_zero_probability_check_does_not_treat_false_as_zero(rank_profile):
    with pytest.raises(TypeError, match="real number"):
        rank_profile.map(reverse, False)


# the generator


def test_row_transform_fn_receives_the_generator_passed_as_rng(rank_profile):
    generator = np.random.default_rng(0)
    received = []

    def record_rng(row, rng):
        received.append(rng)
        return row

    rank_profile.map(record_rng, rng=generator)

    assert received == [generator, generator]


def test_profile_transform_receives_the_generator_passed_as_rng(rank_profile):
    generator = np.random.default_rng(0)
    received = []

    class RecordsRng:
        def transform_matrix(self, matrix, rng):
            received.append(rng)
            return matrix

    rank_profile.transform(RecordsRng(), rng=generator)

    assert received == [generator]


def test_one_seed_reproduces_a_random_row_transform_fn():
    profile = ScoreProfile(
        ballots=[ScoreBallot(scores={"A": 1, "B": 1}, weight=20)], candidates=["A", "B"]
    )

    def add_noise(scores, rng):
        return scores + rng.integers(0, 3, size=len(scores))

    first = profile.map(add_noise, 0.5, mode="voter", rng=3)
    again = profile.map(add_noise, 0.5, mode="voter", rng=3)

    pd.testing.assert_frame_equal(first.to_df(), again.to_df())


def test_voter_mode_gives_each_voter_its_own_draw():
    profile = ScoreProfile(
        ballots=[ScoreBallot(scores={"A": 1, "B": 1}, weight=20)], candidates=["A", "B"]
    )

    def add_noise(scores, rng):
        return scores + rng.integers(0, 3, size=len(scores))

    transformed = profile.map(add_noise, mode="voter", rng=0)

    assert len(score_weights(transformed)) > 1


def test_row_transform_fn_rejects_return_of_a_non_sequence(rank_profile):
    with pytest.raises(TransformContractError, match="must return a 1D array or a sequence"):
        rank_profile.map(lambda row, rng: set(row))
