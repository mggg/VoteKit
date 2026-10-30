from fractions import Fraction

import pandas as pd
import pytest

from votekit.ballot import RankBallot, ScoreBallot
from votekit.pref_profile import RankProfile, ScoreProfile
from votekit.pref_profile.utils import subtract_profiles


def test_subtract_rank_profiles_weights():
    base_profile = RankProfile(
        ballots=[
            RankBallot(ranking=({"A"}, {"B"}, {"C"}), weight=5),
            RankBallot(ranking=({"A", "B"}, frozenset(), {"D"}), weight=3, voter_set={"Chris"}),
        ],
        candidates=["A", "B", "C", "D"],
        max_ranking_length=3,
    )
    subtracted_profile = RankProfile(
        ballots=[
            RankBallot(ranking=({"A"}, {"B"}, {"C"}), weight=2),
        ],
        candidates=["A", "B", "C", "D"],
        max_ranking_length=3,
    )
    result = base_profile - subtracted_profile
    weights = {b.ranking: b.weight for b in result.ballots}
    assert weights[(frozenset({"A"}), frozenset({"B"}), frozenset({"C"}))] == 3
    assert weights[(frozenset({"A", "B"}), frozenset(), frozenset({"D"}))] == 3


def test_subtract_rank_profiles_retains_voter_set():
    base_profile = RankProfile(
        ballots=[
            RankBallot(ranking=({"A"}, {"B"}), weight=4, voter_set={"Chris", "Sam"}),
        ],
        candidates=["A", "B"],
        max_ranking_length=2,
    )
    subtracted_profile = RankProfile(
        ballots=[
            RankBallot(ranking=({"A"}, {"B"}), weight=1),
        ],
        candidates=["A", "B"],
        max_ranking_length=2,
    )
    result = base_profile - subtracted_profile
    assert result.ballots[0].voter_set == {"Chris", "Sam"}


def test_subtract_rank_profiles_removes_exhausted_ballot():
    base_profile = RankProfile(
        ballots=[
            RankBallot(ranking=({"A"}, {"B"}), weight=1),
            RankBallot(ranking=({"B"},), weight=2),
        ],
        candidates=["A", "B"],
        max_ranking_length=2,
    )
    subtracted_profile = RankProfile(
        ballots=[
            RankBallot(ranking=({"A"}, {"B"}), weight=3),
        ],
        candidates=["A", "B"],
        max_ranking_length=2,
    )
    result = base_profile - subtracted_profile
    assert result.ballots == (RankBallot(ranking=({"B"},), weight=2),)


def test_subtract_rank_profiles_groups_base_profile():
    base_profile = RankProfile(
        ballots=[
            RankBallot(ranking=({"A"},), weight=2, voter_set={"Chris"}),
            RankBallot(ranking=({"A"},), weight=1, voter_set={"Sam"}),
        ],
        candidates=["A"],
        max_ranking_length=1,
    )
    subtracted_profile = RankProfile(
        ballots=[RankBallot(ranking=({"A"},), weight=2)], candidates=["A"]
    )
    result = base_profile - subtracted_profile
    assert result.ballots == (RankBallot(ranking=({"A"},), weight=1, voter_set={"Chris", "Sam"}),)


def test_subtract_rank_profiles_ignores_unmatched_ballots():
    base_profile = RankProfile(
        ballots=[RankBallot(ranking=({"A"},), weight=2)], candidates=["A", "B"]
    )
    subtracted_profile = RankProfile(
        ballots=[RankBallot(ranking=({"B"},), weight=1), RankBallot(ranking=({"C"},), weight=1)],
        candidates=["B", "C"],
    )
    result = base_profile - subtracted_profile
    assert result.ballots == (RankBallot(ranking=({"A"},), weight=2),)
    assert result.candidates == base_profile.candidates


def test_subtract_rank_profiles_different_lengths():
    base_profile = RankProfile(
        ballots=[
            RankBallot(ranking=({"A"}, {"B"}), weight=5),
        ],
        candidates=["A", "B", "C"],
        max_ranking_length=3,
    )
    subtracted_profile = RankProfile(
        ballots=[
            RankBallot(ranking=({"A"}, {"B"}), weight=2),
        ],
        candidates=["A", "B", "C"],
        max_ranking_length=2,
    )
    result = base_profile - subtracted_profile
    assert result.ballots[0].weight == 3
    assert result.max_ranking_length == 3


def test_subtract_rank_profiles_with_longer_ranking_length_for_subtracted_profile():
    base_profile = RankProfile(
        ballots=[RankBallot(ranking=({"A"},), weight=3)], max_ranking_length=1
    )
    subtracted_profile = RankProfile(
        ballots=[RankBallot(ranking=({"A"},), weight=1)], max_ranking_length=4
    )
    result = base_profile - subtracted_profile
    assert result.ballots[0].weight == 2
    assert result.max_ranking_length == 1


def test_subtract_score_profiles_weights():
    base_profile = ScoreProfile(
        ballots=[
            ScoreBallot(scores={"A": 2, "B": 2}, weight=4),
            ScoreBallot(scores={"A": 1, "C": 3}, weight=2),
        ],
        candidates=["A", "B", "C"],
    )
    subtracted_profile = ScoreProfile(
        ballots=[
            ScoreBallot(scores={"A": 2, "B": 2}, weight=1),
        ],
        candidates=["A", "B", "C"],
    )
    result = base_profile - subtracted_profile
    weights = {tuple(sorted(b.scores.items())): b.weight for b in result.ballots}
    assert weights[(("A", 2), ("B", 2))] == 3
    assert weights[(("A", 1), ("C", 3))] == 2


def test_subtract_score_profiles_removes_exhausted_ballot():
    base_profile = ScoreProfile(
        ballots=[
            ScoreBallot(scores={"A": 2}, weight=1),
        ],
        candidates=["A"],
    )
    subtracted_profile = ScoreProfile(
        ballots=[
            ScoreBallot(scores={"A": 2}, weight=3),
        ],
        candidates=["A"],
    )
    result = base_profile - subtracted_profile
    assert result.num_ballots == 0


def test_subtract_score_profiles_matches_unscored_candidates():
    base_profile = ScoreProfile(
        ballots=[
            ScoreBallot(scores={"A": 2}, weight=3),
            ScoreBallot(scores={"A": 2, "B": 1}, weight=1),
        ],
        candidates=["A", "B"],
    )
    subtracted_profile = ScoreProfile(
        ballots=[ScoreBallot(scores={"A": 2}, weight=1)], candidates=["A", "C"]
    )
    result = base_profile - subtracted_profile
    weights = {tuple(sorted(ballot.scores.items())): ballot.weight for ballot in result.ballots}
    assert weights == {(("A", 2),): 2, (("A", 2), ("B", 1)): 1}


def test_subtract_score_profiles_zero_score_does_not_match_unscored_candidate():
    base_profile = ScoreProfile(
        ballots=[
            ScoreBallot(scores={"A": 2}, weight=3),
            ScoreBallot(scores={"A": 2, "B": 1}, weight=1),
        ],
        candidates=["A", "B"],
    )
    # Built ScoreProfile from df to keep 0 score, ScoreBallot will drop it
    subtracted_df = pd.DataFrame(
        {"A": [2.0], "C": [0.0], "Weight": [1.0], "Voter Set": [frozenset()]}
    )
    subtracted_df.index.name = "Ballot Index"
    subtracted_profile = ScoreProfile(df=subtracted_df, candidates=["A", "C"])
    result = base_profile - subtracted_profile
    weights = {tuple(sorted(ballot.scores.items())): ballot.weight for ballot in result.ballots}
    assert weights == {(("A", 2),): 3, (("A", 2), ("B", 1)): 1}
    assert result.candidates == base_profile.candidates


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
    base_profile = RankProfile(
        ballots=[
            RankBallot(ranking=({"A"}, {"B"}), weight=5),
        ],
        candidates=["A", "B"],
        max_ranking_length=2,
    )
    subtracted_profile = RankProfile(
        ballots=[
            RankBallot(ranking=({"A"}, {"B"}), weight=2),
        ],
        candidates=["A", "B"],
        max_ranking_length=2,
    )
    result = subtract_profiles(base_profile, subtracted_profile)
    assert result.ballots[0].weight == 3


def test_subtract_profiles_unknown_type_raises():
    class FakeProfile:
        pass

    with pytest.raises(TypeError, match="Cannot subtract"):
        subtract_profiles(FakeProfile(), FakeProfile())  # type: ignore[arg-type]


def test_subtract_sequence_of_profiles():
    base_profile = RankProfile(ballots=[RankBallot(ranking=({"A"},), weight=5)], candidates=["A"])
    subtracted_profiles = [
        RankProfile(ballots=[RankBallot(ranking=({"A"},), weight=1)], candidates=["A"]),
        RankProfile(ballots=[RankBallot(ranking=({"A"},), weight=2)], candidates=["A"]),
    ]
    result = subtract_profiles(base_profile, subtracted_profiles)
    assert result.ballots[0].weight == 2


@pytest.mark.parametrize(
    "base_profile_weight, subtracted_profile_weight, expected",
    [
        (Fraction(2, 3), Fraction(1, 6), Fraction(1, 2)),
        (Fraction(5, 3), 1.0, Fraction(2, 3)),
        (2.0, Fraction(1, 2), 1.5),
    ],
)
def test_subtract_profiles_keeps_base_profile_weight_type(
    base_profile_weight, subtracted_profile_weight, expected
):
    base_profile = RankProfile(ballots=[RankBallot(ranking=({"A"},), weight=base_profile_weight)])
    subtracted_profile = RankProfile(
        ballots=[RankBallot(ranking=({"A"},), weight=subtracted_profile_weight)]
    )
    result = base_profile - subtracted_profile
    assert result.ballots[0].weight == expected
    assert type(result.ballots[0].weight) is type(expected)


@pytest.mark.parametrize(
    "base_profile_weight, subtracted_profile_weight, message",
    [
        (Fraction(2, 3), 0.5, "Cannot mix Fraction weights with non-integral float weights"),
        (2.0, Fraction(1, 3), "Cannot subtract weight 1/3 from float weights without rounding"),
    ],
)
def test_subtract_profiles_rejects_inexact_weight_conversion(
    base_profile_weight, subtracted_profile_weight, message
):
    base_profile = RankProfile(ballots=[RankBallot(ranking=({"A"},), weight=base_profile_weight)])
    subtracted_profile = RankProfile(
        ballots=[RankBallot(ranking=({"A"},), weight=subtracted_profile_weight)]
    )
    with pytest.raises(ValueError, match=message):
        base_profile - subtracted_profile


def test_subtract_profiles_removes_weight_within_zero_tolerance():
    base_profile = RankProfile(
        ballots=[RankBallot(ranking=({"A"},), weight=0.1), RankBallot(ranking=({"A"},), weight=0.2)]
    )
    subtracted_profile = RankProfile(ballots=[RankBallot(ranking=({"A"},), weight=0.3)])
    result = base_profile - subtracted_profile
    assert result.num_ballots == 0
