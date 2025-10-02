import pytest
from votekit.pref_profile import RankProfile, ScoreProfile
from votekit.ballot import RankBallot, ScoreBallot
from votekit.pref_profile import (
    convert_rank_profile_to_score_profile_via_score_vector,
)


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
