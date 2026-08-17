import pytest

from votekit.ballot import RankBallot, ScoreBallot
from votekit.pref_profile import RankProfile, ScoreProfile
from votekit.pref_profile.utils import subtract_profiles


def test_subtract_rank_profiles_weights():
    minuend = RankProfile(
        ballots=[
            RankBallot(ranking=({"A"}, {"B"}, {"C"}), weight=5),
            RankBallot(ranking=({"A", "B"}, frozenset(), {"D"}), weight=3, voter_set={"Chris"}),
        ],
        candidates=["A", "B", "C", "D"],
        max_ranking_length=3,
    )
    subtrahend = RankProfile(
        ballots=[
            RankBallot(ranking=({"A"}, {"B"}, {"C"}), weight=2),
        ],
        candidates=["A", "B", "C", "D"],
        max_ranking_length=3,
    )
    result = minuend - subtrahend
    weights = {b.ranking: b.weight for b in result.ballots}
    assert weights[(frozenset({"A"}), frozenset({"B"}), frozenset({"C"}))] == 3
    assert weights[(frozenset({"A", "B"}), frozenset(), frozenset({"D"}))] == 3


def test_subtract_rank_profiles_retains_voter_set():
    minuend = RankProfile(
        ballots=[
            RankBallot(ranking=({"A"}, {"B"}), weight=4, voter_set={"Chris", "Sam"}),
        ],
        candidates=["A", "B"],
        max_ranking_length=2,
    )
    subtrahend = RankProfile(
        ballots=[
            RankBallot(ranking=({"A"}, {"B"}), weight=1),
        ],
        candidates=["A", "B"],
        max_ranking_length=2,
    )
    result = minuend - subtrahend
    assert result.ballots[0].voter_set == {"Chris", "Sam"}


def test_subtract_rank_profiles_negative_weight_raises():
    minuend = RankProfile(
        ballots=[
            RankBallot(ranking=({"A"}, {"B"}), weight=1),
        ],
        candidates=["A", "B"],
        max_ranking_length=2,
    )
    subtrahend = RankProfile(
        ballots=[
            RankBallot(ranking=({"A"}, {"B"}), weight=3),
        ],
        candidates=["A", "B"],
        max_ranking_length=2,
    )
    with pytest.raises(ValueError, match="negative"):
        minuend - subtrahend


def test_subtract_rank_profiles_different_lengths():
    minuend = RankProfile(
        ballots=[
            RankBallot(ranking=({"A"}, {"B"}), weight=5),
        ],
        candidates=["A", "B", "C"],
        max_ranking_length=3,
    )
    subtrahend = RankProfile(
        ballots=[
            RankBallot(ranking=({"A"}, {"B"}), weight=2),
        ],
        candidates=["A", "B", "C"],
        max_ranking_length=2,
    )
    result = minuend - subtrahend
    assert result.ballots[0].weight == 3
    assert result.max_ranking_length == 3


def test_subtract_score_profiles_weights():
    minuend = ScoreProfile(
        ballots=[
            ScoreBallot(scores={"A": 2, "B": 2}, weight=4),
            ScoreBallot(scores={"A": 1, "C": 3}, weight=2),
        ],
        candidates=["A", "B", "C"],
    )
    subtrahend = ScoreProfile(
        ballots=[
            ScoreBallot(scores={"A": 2, "B": 2}, weight=1),
        ],
        candidates=["A", "B", "C"],
    )
    result = minuend - subtrahend
    weights = {tuple(sorted(b.scores.items())): b.weight for b in result.ballots}
    assert weights[(("A", 2), ("B", 2))] == 3
    assert weights[(("A", 1), ("C", 3))] == 2


def test_subtract_score_profiles_negative_weight_raises():
    minuend = ScoreProfile(
        ballots=[
            ScoreBallot(scores={"A": 2}, weight=1),
        ],
        candidates=["A"],
    )
    subtrahend = ScoreProfile(
        ballots=[
            ScoreBallot(scores={"A": 2}, weight=3),
        ],
        candidates=["A"],
    )
    with pytest.raises(ValueError, match="negative"):
        minuend - subtrahend


def test_subtract_profiles_mixed_types_raises():
    score_profile = ScoreProfile(
        ballots=[ScoreBallot(scores={"A": 2}, weight=2)],
        candidates=["A"],
    )
    rank_profile = RankProfile(
        ballots=[RankBallot(ranking=({"A"},), weight=2)],
        candidates=["A"],
        max_ranking_length=1,
    )
    with pytest.raises(TypeError, match="same type"):
        subtract_profiles(rank_profile, score_profile)


def test_subtract_profiles_function():
    minuend = RankProfile(
        ballots=[
            RankBallot(ranking=({"A"}, {"B"}), weight=5),
        ],
        candidates=["A", "B"],
        max_ranking_length=2,
    )
    subtrahend = RankProfile(
        ballots=[
            RankBallot(ranking=({"A"}, {"B"}), weight=2),
        ],
        candidates=["A", "B"],
        max_ranking_length=2,
    )
    result = subtract_profiles(minuend, subtrahend)
    assert result.ballots[0].weight == 3


def test_subtract_profiles_unknown_type_raises():
    class FakeProfile:
        pass

    with pytest.raises(TypeError, match="Cannot subtract"):
        subtract_profiles(FakeProfile(), FakeProfile())  # type: ignore[arg-type]
