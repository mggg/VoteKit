import pytest

from votekit.ballot import ScoreBallot
from votekit.elections import GeneralRating
from votekit.pref_profile import ScoreProfile


def test_init():
    e = GeneralRating(
        ScoreProfile(ballots=[ScoreBallot(scores={"A": 2, "B": 2, "C": 0})]),
        n_seats=2,
        per_candidate_limit=2,
        budget=4,
    )

    assert e.get_elected(1) == (frozenset({"A", "B"}),)


def test_errors():
    with pytest.raises(ValueError, match="n_seats must be positive."):
        GeneralRating(ScoreProfile(), n_seats=0)

    with pytest.raises(ValueError, match="per_candidate_limit must be positive."):
        GeneralRating(ScoreProfile(), per_candidate_limit=0)

    with pytest.raises(
        ValueError, match="per_candidate_limit must be less than or equal to budget."
    ):
        GeneralRating(ScoreProfile(), per_candidate_limit=4, budget=2)

    with pytest.raises(ValueError, match="tiebreak must be None or 'random'."):
        GeneralRating(ScoreProfile(), tiebreak="borda")


def test_validate_profile():
    with pytest.raises(TypeError, match="violates score limit"):
        profile = ScoreProfile(ballots=[ScoreBallot(scores={"A": 3, "B": 2})])
        GeneralRating(profile, n_seats=2, per_candidate_limit=2)

    with pytest.raises(TypeError, match="must have non-negative scores."):
        profile = ScoreProfile(ballots=[ScoreBallot(scores={"A": -3, "B": 2, "C": 2})])
        GeneralRating(profile, n_seats=2, per_candidate_limit=2)

    with pytest.raises(TypeError, match="All ballots must have score dictionary."):
        profile = ScoreProfile(ballots=[ScoreBallot(), ScoreBallot(scores={"A": 2, "B": 2})])
        GeneralRating(profile, n_seats=2, per_candidate_limit=2)

    with pytest.raises(TypeError, match="violates total score budget"):
        profile = ScoreProfile(ballots=[ScoreBallot(scores={"A": 2, "C": 2, "B": 2})])
        GeneralRating(profile, per_candidate_limit=2, budget=2)
