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
from votekit.pref_profile.profile_matrix import ProfileVoterWeightData, ScoreMatrix, WeightKind
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
def voter_profile() -> RankProfile:
    """A > B for voters v1 and v2, and B > A for voter v3."""
    return RankProfile(
        ballots=[
            RankBallot(ranking=[{"A"}, {"B"}], weight=2, voter_set={"v1", "v2"}),
            RankBallot(ranking=[{"B"}, {"A"}], weight=1, voter_set={"v3"}),
        ]
    )


class ReturnsVoterData:
    """A ProfileTransform that returns the selected rows with the given weights and voter sets."""

    def __init__(self, weights: list[float], voter_sets: list[frozenset[str]] | None = None):
        self.weights = weights
        self.voter_sets = voter_sets

    def transform_matrix(self, matrix):
        voter_sets = None
        if self.voter_sets is not None:
            voter_sets = np.empty(len(self.voter_sets), dtype=object)
            voter_sets[:] = self.voter_sets
        data = ProfileVoterWeightData(
            weights=np.array(self.weights, dtype=np.float64),
            voter_sets=voter_sets,
            weight_kind=WeightKind.FLOAT,
            ballot_index=matrix.voter_weight_data.ballot_index,
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


def reverse(row):
    """A per-voter function that reverses the ranked positions, leaving padding at the end."""
    num_ranked = sum(cand_set != END_OF_RANKING_SET for cand_set in row)
    row[:num_ranked] = row[:num_ranked][::-1]
    return row


def test_profile_transform_rejects_return_of_diff_matrix_type(rank_profile):
    score_matrix = ScoreProfile(ballots=[ScoreBallot(scores={"A": 1})], candidates=["A"])._matrix

    class ReturnsScoreMatrix:
        def transform_matrix(self, matrix):
            return score_matrix

    with pytest.raises(TransformContractError, match="transform_fn must return a RankMatrix"):
        rank_profile.perturb(ReturnsScoreMatrix())


def test_profile_transform_rejects_change_in_total_weight(rank_profile):
    class DropsSecondRow:
        def transform_matrix(self, matrix):
            return matrix.apply_mask(np.array([True, False]))

    with pytest.raises(TransformContractError, match="transform_fn must conserve weight."):
        rank_profile.perturb(DropsSecondRow())


def test_profile_transform_rejects_non_integral_weights(rank_profile):
    with pytest.raises(TransformContractError, match="integral weights"):
        rank_profile.perturb(ReturnsVoterData([1.5, 1.5]))


@pytest.mark.skip(reason="voter-set checks on ProfileTransform output are pending discussion")
def test_profile_transform_rejects_a_voter_set_larger_than_its_weight(voter_profile):
    voter_sets = [frozenset({"v1", "v2", "v3"}), frozenset()]

    with pytest.raises(TransformContractError, match="one voter per unit of weight"):
        voter_profile.perturb(ReturnsVoterData([2, 1], voter_sets))


@pytest.mark.skip(reason="voter-set checks on ProfileTransform output are pending discussion")
def test_profile_transform_rejects_a_voter_on_two_rows(voter_profile):
    voter_sets = [frozenset({"v1", "v2"}), frozenset({"v2"})]

    with pytest.raises(TransformContractError, match="Voter 'v2' appears on more than one row"):
        voter_profile.perturb(ReturnsVoterData([2, 1], voter_sets))


@pytest.mark.skip(reason="voter-set checks on ProfileTransform output are pending discussion")
def test_profile_transform_rejects_a_dropped_voter(voter_profile):
    voter_sets = [frozenset({"v1", "v2"}), frozenset()]

    with pytest.raises(TransformContractError, match=r"Dropped voters \{'v3'\}"):
        voter_profile.perturb(ReturnsVoterData([2, 1], voter_sets))


@pytest.mark.skip(reason="voter-set checks on ProfileTransform output are pending discussion")
def test_profile_transform_rejects_an_invented_voter(voter_profile):
    voter_sets = [frozenset({"v1", "v2"}), frozenset({"v9"})]

    with pytest.raises(TransformContractError, match=r"Invented voters \{'v9'\}"):
        voter_profile.perturb(ReturnsVoterData([2, 1], voter_sets))


def test_perturbed_profile_can_be_perturbed_again(voter_profile):
    once = voter_profile.perturb(SwapAdjacentRankPositions(rng_seed=0), 0.5, rng_seed=0)
    twice = once.perturb(reverse, 0.5, rng_seed=1)

    assert twice.total_ballot_wt == 3
    assert set().union(*twice.to_df()["Voter Set"]) == {"v1", "v2", "v3"}


def test_profile_transform_rejects_changed_ranking_columns(rank_profile):
    class DropsLastColumn:
        def transform_matrix(self, matrix):
            return replace(
                matrix,
                rankings=matrix.rankings[:, :-1],
                ranking_columns=matrix.ranking_columns[:-1],
            )

    with pytest.raises(TransformContractError, match="transform_fn must keep the ranking columns"):
        rank_profile.perturb(DropsLastColumn())


def test_profile_transform_rejects_changed_score_columns(score_profile):
    class ReversesColumns:
        def transform_matrix(self, matrix):
            return ScoreMatrix(
                voter_weight_data=matrix.voter_weight_data,
                scores=matrix.scores[:, ::-1],
                score_columns=matrix.score_columns[::-1],
                candidates=matrix.candidates,
            )

    with pytest.raises(TransformContractError, match="transform_fn must keep the score columns"):
        score_profile.perturb(ReversesColumns())


def test_perturb_rejects_transform_class_instead_of_an_instance(rank_profile):
    with pytest.raises(TransformContractError, match="transform_fn was given the class"):
        rank_profile.perturb(SwapCandidates)


def test_perturb_rejects_non_callable_transform(rank_profile):
    with pytest.raises(
        TransformContractError,
        match="transform_fn must be a per-voter transform function or ProfileTransform instance,",
    ):
        rank_profile.perturb("transform_matrix")


def test_ranking_function_rejects_return_of_non_set_entries(rank_profile):
    with pytest.raises(
        TransformContractError,
        match="transform_fn must return an array of sets or frozensets of candidates",
    ):
        rank_profile.perturb(lambda row: ["A", "B", "C"])


def test_ranking_function_rejects_return_of_the_wrong_length(rank_profile):
    with pytest.raises(
        TransformContractError, match="transform_fn must return a 1D array of length 3"
    ):
        rank_profile.perturb(lambda row: row[:2])


def test_ranking_function_rejects_return_of_a_2d_array(rank_profile):
    with pytest.raises(
        TransformContractError, match="transform_fn must return a 1D array of length 3"
    ):
        rank_profile.perturb(lambda row: np.stack([row, row]))


def test_ranking_function_rejects_padding_before_a_ranked_position(rank_profile):
    with pytest.raises(TransformContractError, match="only at the end of a ranking"):
        rank_profile.perturb(lambda row: [END_OF_RANKING_SET, frozenset({"A"}), frozenset({"B"})])


def test_ranking_function_rejects_a_gap_in_the_ranking(rank_profile):
    with pytest.raises(
        TransformContractError, match="only at the end of a ranking.*Ballot Index 0"
    ):
        rank_profile.perturb(lambda row: [frozenset({"A"}), END_OF_RANKING_SET, frozenset({"B"})])


def test_ranking_function_rejects_an_empty_position_after_padding(rank_profile):
    with pytest.raises(
        TransformContractError, match="only at the end of a ranking.*Ballot Index 0"
    ):
        rank_profile.perturb(lambda row: [frozenset({"A"}), END_OF_RANKING_SET, frozenset()])


def test_ranking_function_rejects_tilde_inside_a_candidate_set(rank_profile):
    with pytest.raises(TransformContractError, match="only use '~' as the padding.*Ballot Index 0"):
        rank_profile.perturb(
            lambda row: [frozenset({"A", "~"}), END_OF_RANKING_SET, END_OF_RANKING_SET]
        )


def test_ranking_function_rejects_an_undeclared_candidate(rank_profile):
    def add_z(row):
        row[2] = frozenset({"Z"})
        return row

    with pytest.raises(
        TransformContractError,
        match=r"returned candidates \{'Z'\} not in the"
        " profile",
    ):
        rank_profile.perturb(add_z)


def test_ranking_function_can_rank_a_non_cast_candidate():
    def add_cand_to_cast(row):
        row[2] = frozenset({"new_cand"})
        return row

    profile = RankProfile(
        ballots=[RankBallot(ranking=[{"A"}, {"B"}], weight=2)],
        candidates=["A", "B", "new_cand"],
        max_ranking_length=3,
    )
    perturbed = profile.perturb(add_cand_to_cast)

    assert ranking_weights(perturbed) == {ranking("A", "B", "new_cand"): 2.0}


def test_ranking_function_rejects_a_tie_longer_than_max_ranking_length():
    profile = RankProfile(
        ballots=[RankBallot(ranking=[{"A"}, {"B"}])],
        candidates=["A", "B", "C"],
        max_ranking_length=2,
    )

    with pytest.raises(
        TransformContractError, match="3 candidates, more than the profile's max_ranking_length"
    ):
        profile.perturb(lambda row: [frozenset({"A", "B", "C"}), END_OF_RANKING_SET])


def test_per_voter_function_is_called_once_per_selected_voter(rank_profile):
    calls = []

    def record(row):
        calls.append(row)
        return row

    rank_profile.perturb(record)

    assert len(calls) == 3


def test_each_selected_voter_gets_its_own_row_with_voters_in_str_order():
    profile = RankProfile(
        ballots=[RankBallot(ranking=[{"A"}, {"B"}], weight=3, voter_set={"v2", "v10", "v1"})]
    )
    df = profile.perturb(lambda row: row).to_df()

    assert df["Weight"].tolist() == [1.0, 1.0, 1.0]

    assert df["Voter Set"].tolist() == [frozenset({"v1"}), frozenset({"v10"}), frozenset({"v2"})]


def test_per_voter_errors_name_the_ballot_index(labeled_profile):
    def unwrap_first_place_cand_B(row):
        if row[0] == {"B"}:
            row[0] = "B"
        return row

    with pytest.raises(TransformContractError, match="for Ballot Index 20"):
        labeled_profile.perturb(unwrap_first_place_cand_B)


def test_transform_fn_receives_a_writeable_copy(score_profile):
    def double(scores):
        assert isinstance(scores, np.ndarray)
        scores *= 2
        return scores

    perturbed = score_profile.perturb(double)

    assert score_weights(perturbed) == {(2.0, 4.0): 1.0}
    assert score_weights(score_profile) == {(1.0, 2.0): 1.0}


def test_score_transform_fn_rejects_non_numeric_scores(score_profile):
    with pytest.raises(TransformContractError, match="transform_fn returned a non-numeric score"):
        score_profile.perturb(lambda scores: ("high", 1.0))


def test_score_transform_fn_rejects_a_negative_score(score_profile):
    with pytest.raises(TransformContractError, match="negative score"):
        score_profile.perturb(lambda scores: (-1.0, 1.0))


def test_score_transform_fn_rejects_scoring_an_undeclared_candidate():
    df = pd.DataFrame(
        {"A": [1.0], "B": [2.0], "Z": [np.nan], "Voter Set": [set()], "Weight": [1.0]}
    )
    df.index.name = "Ballot Index"
    profile = ScoreProfile(df=df, candidates=["A", "B"])

    def score_z(scores):
        scores[2] = 5.0
        return scores

    with pytest.raises(
        TransformContractError,
        match=r"\{'Z'\} not in the profile's candidates. Declare them.*Ballot Index 0",
    ):
        profile.perturb(score_z)


def test_score_transform_fn_may_leave_an_undeclared_column_at_zero_or_unscored():
    df = pd.DataFrame(
        {"A": [1.0], "B": [2.0], "Z": [np.nan], "Voter Set": [set()], "Weight": [1.0]}
    )
    df.index.name = "Ballot Index"
    profile = ScoreProfile(df=df, candidates=["A", "B"])

    def set_z_to(value):
        def transform(scores):
            scores[2] = value
            return scores

        return transform

    assert profile.perturb(set_z_to(0.0)).to_df()["Z"].tolist() == [0.0]
    assert profile.perturb(set_z_to(np.nan)).to_df()["Z"].isna().all()


def test_default_probability_perturbs_every_voter(rank_profile):
    perturbed = rank_profile.perturb(reverse)

    assert ranking_weights(perturbed) == {ranking("C", "B", "A"): 2.0, ranking("A", "B"): 1.0}


def test_zero_probability_leaves_every_ballot_unchanged(rank_profile):
    perturbed = rank_profile.perturb(SwapCandidates("A", "B"), 0.0, rng_seed=0)

    pd.testing.assert_frame_equal(perturbed.to_df(), rank_profile.to_df())


def test_zero_probability_leaves_a_score_profile_unchanged(score_profile):
    class DoublesScores:
        def transform_matrix(self, matrix):
            return replace(matrix, scores=matrix.scores * 2)

    perturbed = score_profile.perturb(DoublesScores(), 0.0, rng_seed=0)

    pd.testing.assert_frame_equal(perturbed.to_df(), score_profile.to_df())


def test_probability_function_chooses_which_rows_to_perturb(rank_profile):
    def only_ballots_starting_with_b(row):
        return 1.0 if row[0] == {"B"} else 0.0

    perturbed = rank_profile.perturb(reverse, only_ballots_starting_with_b)

    assert ranking_weights(perturbed) == {ranking("A", "B", "C"): 2.0, ranking("A", "B"): 1.0}


def test_probability_rejects_a_non_real_number(rank_profile):
    with pytest.raises(TypeError, match="real number"):
        rank_profile.perturb(reverse, "high")


def test_probability_function_rejects_return_of_a_non_real_number(rank_profile):
    with pytest.raises(TypeError, match="real number"):
        rank_profile.perturb(reverse, lambda row: "high")


def test_probability_rejects_a_boolean(rank_profile):
    with pytest.raises(TypeError, match="real number"):
        rank_profile.perturb(reverse, True)
    with pytest.raises(TypeError, match="real number"):
        rank_profile.perturb(reverse, np.True_)


def test_probability_rejects_a_value_above_one(rank_profile):
    with pytest.raises(ValueError, match="must lie in"):
        rank_profile.perturb(reverse, 1.5)


def test_probability_rejects_a_value_below_zero(rank_profile):
    with pytest.raises(ValueError, match="must lie in"):
        rank_profile.perturb(reverse, -0.1)


def test_probability_function_error_shows_the_row(labeled_profile):
    row = "[frozenset({'A'}) frozenset({'B'})]"

    with pytest.raises(ValueError, match=re.escape(f"per_voter_probability({row}) must lie in")):
        labeled_profile.perturb(reverse, lambda row: 2.0)


def test_probability_function_rejects_writing_to_its_row(rank_profile):
    def write_then_choose(row):
        row[0] = frozenset({"C"})
        return 1.0

    with pytest.raises(ValueError, match="read-only"):
        rank_profile.perturb(reverse, write_then_choose)


def test_perturb_rejects_non_integral_weights():
    profile = RankProfile(ballots=[RankBallot(ranking=[{"A"}, {"B"}], weight=1.5)])

    with pytest.raises(ValueError, match="non-integral weights"):
        profile.perturb(reverse)


def test_perturb_rejects_voter_sets_not_matching_weights_even_when_none_are_selected():
    profile = RankProfile(
        ballots=[RankBallot(ranking=[{"A"}, {"B"}], weight=3, voter_set={"v1", "v2"})]
    )

    with pytest.raises(ProfileError, match="voter set of size 2 but weight 3"):
        profile.perturb(reverse, 0.0)


def test_voter_set_mismatch_names_the_ballot_index():
    df = pd.DataFrame(
        {
            "Ranking_1": [frozenset({"A"}), frozenset({"B"})],
            "Voter Set": [frozenset({"v1"}), frozenset({"v2", "v3"})],
            "Weight": [1.0, 1.0],
        },
        index=pd.Index([10, 20], name="Ballot Index"),
    )
    profile = RankProfile(df=df, candidates=["A", "B"], max_ranking_length=1)

    with pytest.raises(ProfileError, match="Ballot Index 20 has a voter set of size 2"):
        profile.perturb(reverse)


def test_seeded_selection_splits_the_voter_set_exactly():
    profile = RankProfile(
        ballots=[RankBallot(ranking=[{"A"}, {"B"}], weight=4, voter_set={"a", "b", "c", "d"})]
    )
    df = profile.perturb(reverse, 0.5, rng_seed=0).to_df()

    assert df["Weight"].tolist() == [2.0, 1.0, 1.0]
    assert df["Voter Set"].tolist() == [frozenset({"a", "c"}), frozenset({"b"}), frozenset({"d"})]
    assert df["Ranking_1"].tolist() == [frozenset({"A"}), frozenset({"B"}), frozenset({"B"})]


def test_selecting_no_voters_keeps_the_row_and_its_voters():
    profile = RankProfile(
        ballots=[RankBallot(ranking=[{"A"}, {"B"}], weight=2, voter_set={"v1", "v2"})]
    )
    df = profile.perturb(reverse, 0.0).to_df()

    assert df["Weight"].tolist() == [2.0]
    assert df["Voter Set"].tolist() == [frozenset({"v1", "v2"})]


def test_selecting_every_voter_leaves_no_unperturbed_row():
    profile = RankProfile(
        ballots=[RankBallot(ranking=[{"A"}, {"B"}], weight=2, voter_set={"v1", "v2"})]
    )
    df = profile.perturb(reverse, 1.0).to_df()

    assert df["Weight"].tolist() == [1.0, 1.0]
    assert df["Voter Set"].tolist() == [frozenset({"v1"}), frozenset({"v2"})]


def test_partial_selection_splits_a_row_into_kept_and_perturbed_rows():
    profile = RankProfile(ballots=[RankBallot(ranking=[{"A"}, {"B"}], weight=4)])
    df = profile.perturb(SwapCandidates("A", "B"), 0.5, rng_seed=0).to_df()

    assert df["Weight"].tolist() == [2.0, 2.0]
    assert df["Ranking_1"].tolist() == [frozenset({"A"}), frozenset({"B"})]


def test_zero_weight_rows_are_kept_and_never_perturbed():
    profile = RankProfile(
        ballots=[
            RankBallot(ranking=[{"A"}, {"B"}], weight=1),
            RankBallot(ranking=[{"B"}, {"A"}], weight=0),
        ]
    )
    df = profile.perturb(reverse).to_df()

    assert df["Weight"].tolist() == [0.0, 1.0]
    assert df["Ranking_1"].tolist() == [frozenset({"B"}), frozenset({"B"})]


def test_group_ballots_combines_equal_rankings():
    profile = RankProfile(
        ballots=[
            RankBallot(ranking=[{"A"}, {"B"}], weight=1),
            RankBallot(ranking=[{"B"}, {"A"}], weight=1),
        ]
    )
    df = profile.perturb(
        lambda row: [frozenset({"A"}), frozenset({"B"})], group_ballots=True
    ).to_df()

    assert df["Weight"].tolist() == [2.0]


def test_fraction_weights_stay_fractions_when_every_voter_is_perturbed():
    profile = RankProfile(ballots=[RankBallot(ranking=[{"A"}, {"B"}], weight=Fraction(2))])
    perturbed = profile.perturb(reverse)

    assert perturbed._matrix.voter_weight_data.has_fraction_weights
    assert sum(perturbed.to_df()["Weight"]) == Fraction(2)


def test_same_seed_gives_the_same_result(rank_profile):
    first = rank_profile.perturb(reverse, 0.5, rng_seed=7)
    second = rank_profile.perturb(reverse, 0.5, rng_seed=7)

    pd.testing.assert_frame_equal(first.to_df(), second.to_df())


def test_empty_profiles_are_returned_unchanged():
    empty_rank, empty_score = RankProfile(), ScoreProfile()

    assert empty_rank.perturb(reverse) is empty_rank
    assert empty_score.perturb(lambda row: row) is empty_score
