from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction

import pandas as pd
import pytest

from votekit.ballot import RankBallot, ScoreBallot
from votekit.pref_profile import (
    RankProfile,
    ScoreProfile,
    convert_rank_profile_to_score_profile_via_score_vector,
)
from votekit.pref_profile.profile_matrix import RankMatrix


def test_convert_rank_profile_to_score_profile_via_score_vector():
    rank_profile = RankProfile(
        ballots=(
            RankBallot(ranking=({"A"}, {"B"})),
            RankBallot(ranking=({"A"}, {"B"}, {"C"}), voter_set={"Chris"}),
            RankBallot(ranking=({"C"}, {"B"}, {"A"}), weight=2),
        )
    )

    score_vector = [1, 1, 0]

    score_profile_derived = convert_rank_profile_to_score_profile_via_score_vector(
        rank_profile, score_vector
    )

    score_profile_expected = ScoreProfile(
        ballots=(
            ScoreBallot(scores={"A": 1, "B": 1}, weight=1),
            ScoreBallot(scores={"A": 1, "B": 1}, weight=1, voter_set={"Chris"}),
            ScoreBallot(
                scores={
                    "C": 1,
                    "B": 1,
                },
                weight=2,
            ),
        ),
        candidates=("A", "B", "C"),
    )

    assert score_profile_derived == score_profile_expected


def test_convert_rank_profile_to_score_profile_short_score_vector():
    rank_profile = RankProfile(
        ballots=(
            RankBallot(ranking=({"A"}, {"B"})),
            RankBallot(ranking=({"A"}, {"B"}, {"C"}), voter_set={"Chris"}),
            RankBallot(ranking=({"C"}, {"B"}, {"A"}), weight=2),
        )
    )

    score_vector = [1, 1]

    score_profile_derived = convert_rank_profile_to_score_profile_via_score_vector(
        rank_profile, score_vector
    )

    score_profile_expected = ScoreProfile(
        ballots=(
            ScoreBallot(scores={"A": 1, "B": 1}, weight=1),
            ScoreBallot(scores={"A": 1, "B": 1}, weight=1, voter_set={"Chris"}),
            ScoreBallot(
                scores={
                    "C": 1,
                    "B": 1,
                },
                weight=2,
            ),
        ),
        candidates=("A", "B", "C"),
    )

    assert isinstance(score_profile_derived, ScoreProfile)
    assert score_profile_derived == score_profile_expected


def test_convert_catches_ties():
    rank_profile = RankProfile(
        ballots=(
            RankBallot(ranking=({"A"}, {"B"})),
            RankBallot(ranking=({"A", "B"}, {"C"}), voter_set={"Chris"}),
            RankBallot(ranking=({"C"}, {"B"}, {"A"}), weight=2),
        )
    )

    with pytest.raises(ValueError, match="Ballots must not contain ties."):
        convert_rank_profile_to_score_profile_via_score_vector(rank_profile, [1, 1, 0])


def test_convert_catches_bad_score_vector():
    rank_profile = RankProfile(
        ballots=(
            RankBallot(ranking=({"A"}, {"B"})),
            RankBallot(ranking=({"A"}, {"B"}, {"C"}), voter_set={"Chris"}),
            RankBallot(ranking=({"C"}, {"B"}, {"A"}), weight=2),
        )
    )

    with pytest.raises(ValueError, match="Score vector must be non-negative."):
        convert_rank_profile_to_score_profile_via_score_vector(rank_profile, [-1, 1, 0])

    with pytest.raises(ValueError, match="Score vector must be non-increasing."):
        convert_rank_profile_to_score_profile_via_score_vector(rank_profile, [0, 1, 0])


@pytest.mark.parametrize("labels", [[0, 1], [42, 10], [7, 7]])
def test_conversion_avoids_dataframe_conversion_and_preserves_row_alignment(
    labels: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    source_df = RankProfile(
        ballots=[
            RankBallot(ranking=[{"A"}, {"B"}], weight=2, voter_set={"Alice"}),
            RankBallot(ranking=[{"B"}, {"A"}], weight=3, voter_set={"Bob"}),
        ]
    ).to_df()
    source_df.index = pd.Index(labels, name="Ballot Index")
    profile = RankProfile(df=source_df, candidates=["A", "B"], max_ranking_length=2)
    dataframe_conversion_count = 0
    original_to_df = RankMatrix.to_df

    def count_dataframe_conversions(matrix: RankMatrix) -> pd.DataFrame:
        nonlocal dataframe_conversion_count
        dataframe_conversion_count += 1

        return original_to_df(matrix)

    monkeypatch.setattr(RankMatrix, "to_df", count_dataframe_conversions)

    converted = convert_rank_profile_to_score_profile_via_score_vector(profile, [2, 1])

    assert dataframe_conversion_count == 0

    expected_df = pd.DataFrame(
        {
            "A": [2.0, 1.0],
            "B": [1.0, 2.0],
            "Voter Set": [frozenset({"Alice"}), frozenset({"Bob"})],
            "Weight": [2.0, 3.0],
        },
        index=pd.Index(labels, name="Ballot Index"),
    )

    pd.testing.assert_frame_equal(converted.to_df(), expected_df)
    pd.testing.assert_frame_equal(profile.to_df(), source_df)


def test_rank_to_score_preserves_repeated_candidate_last_position_and_fraction_labels(
    forbid_dataframe_conversion: Callable[[], None],
) -> None:
    source = RankProfile(
        ballots=[RankBallot(ranking=[{"A"}, {"A"}], weight=Fraction(1, 3), voter_set={"v"})],
        candidates=("A", "unused"),
    ).to_df()
    source.index = pd.Index([37], name="Ballot Index")
    profile = RankProfile(df=source, candidates=("A", "unused"), max_ranking_length=2)

    forbid_dataframe_conversion()

    converted = convert_rank_profile_to_score_profile_via_score_vector(profile, [2, 1])

    assert converted.ballots == (
        ScoreBallot(scores={"A": 1}, weight=float(Fraction(1, 3)), voter_set={"v"}),
    )

    labels = converted._matrix.voter_weight_data.ballot_index

    assert labels is not None
    assert labels.tolist() == [37]
    assert converted.candidates == ("A", "unused")
