import pytest

from votekit.ballot import RankBallot, ScoreBallot
from votekit.pref_profile import RankProfile, ScoreProfile
from votekit.pref_profile.utils import sum_profiles


def test_sum_profiles_with_mixed_types_raises_type_error():
    score_profile = ScoreProfile(
        ballots=[
            ScoreBallot(scores={"A": 2, "B": 2}, weight=2),
            ScoreBallot(scores={"A": 2, "C": 2}, voter_set={"Chris"}),
            ScoreBallot(),
            ScoreBallot(weight=0),
        ],
        candidates=["A", "B", "C", "D"],
    )
    rank_profile = RankProfile(
        ballots=[
            RankBallot(ranking=({"A"}, {"B"}, {"C"}), weight=2),
            RankBallot(ranking=({"A", "B"}, frozenset(), {"D"}), voter_set={"Chris"}),
            RankBallot(),
            RankBallot(weight=0),
        ],
        candidates=["A", "B", "C", "D"],
        max_ranking_length=3,
    )
    with pytest.raises(
        TypeError,
        match="All profiles must be of the same type.",
    ):
        sum_profiles([score_profile, score_profile, score_profile, rank_profile])

    with pytest.raises(
        TypeError,
        match="All profiles must be of the same type.",
    ):
        sum_profiles([rank_profile, score_profile, score_profile, rank_profile])


def test_sum_empty_profile_raises_value_error():
    with pytest.raises(
        ValueError,
        match="Cannot sum an empty list of profiles",
    ):
        sum_profiles([])


def test_sum_one_profile_returns_same_profile():
    profile = ScoreProfile(
        ballots=[
            ScoreBallot(scores={"A": 2, "B": 2}, weight=2),
            ScoreBallot(scores={"A": 2, "C": 2}, voter_set={"Chris"}),
            ScoreBallot(),
            ScoreBallot(weight=0),
        ],
        candidates=["A", "B", "C", "D"],
    )
    summed_profile = sum_profiles([profile])
    assert summed_profile == profile

    profile = RankProfile(
        ballots=[
            RankBallot(ranking=({"A"}, {"B"}, {"C"}), weight=2),
            RankBallot(ranking=({"A", "B"}, frozenset(), {"D"}), voter_set={"Chris"}),
            RankBallot(),
            RankBallot(weight=0),
        ],
        candidates=["A", "B", "C", "D"],
        max_ranking_length=3,
    )
    summed_profile = sum_profiles([profile])
    assert summed_profile == profile


def test_sum_one_profile_no_list_raises_type_error():
    profile = ScoreProfile(
        ballots=[
            ScoreBallot(scores={"A": 2, "B": 2}, weight=2),
            ScoreBallot(scores={"A": 2, "C": 2}, voter_set={"Chris"}),
            ScoreBallot(),
            ScoreBallot(weight=0),
        ],
        candidates=["A", "B", "C", "D"],
    )
    with pytest.raises(TypeError, match="has no len()"):
        sum_profiles(profile)  # type: ignore[arg-type]

    profile = RankProfile(
        ballots=[
            RankBallot(ranking=({"A"}, {"B"}, {"C"}), weight=2),
            RankBallot(ranking=({"A", "B"}, frozenset(), {"D"}), voter_set={"Chris"}),
            RankBallot(),
            RankBallot(weight=0),
        ],
        candidates=["A", "B", "C", "D"],
        max_ranking_length=3,
    )
    with pytest.raises(TypeError, match="has no len()"):
        sum_profiles(profile)  # type: ignore[arg-type]


def test_sum_score_profiles():
    profile_1 = ScoreProfile(
        ballots=[
            ScoreBallot(scores={"A": 2, "B": 2}, weight=2),
            ScoreBallot(scores={"A": 2, "C": 2}, voter_set={"Chris"}),
            ScoreBallot(),
            ScoreBallot(weight=0),
        ],
        candidates=["A", "B", "C", "D"],
    )

    profile_2 = ScoreProfile(
        ballots=[
            ScoreBallot(scores={"D": 2, "E": 2}, weight=2),
            ScoreBallot(scores={"D": 2, "E": 2, "F": 3.1}, weight=2),
            ScoreBallot(),
            ScoreBallot(weight=0),
        ],
        candidates=["D", "E", "F"],
    )

    profile_3 = ScoreProfile(
        ballots=[
            ScoreBallot(scores={"G": 2, "H": 2}, weight=2),
            ScoreBallot(scores={"G": 2, "H": 2, "I": 3.1}, weight=2),
            ScoreBallot(),
            ScoreBallot(weight=0),
        ],
        candidates=["G", "H", "I"],
    )
    summed_profile = sum_profiles([profile_1, profile_2, profile_3])
    true_summed_profile = ScoreProfile(
        ballots=[
            ScoreBallot(scores={"A": 2, "B": 2}, weight=2),
            ScoreBallot(scores={"A": 2, "C": 2}, voter_set={"Chris"}),
            ScoreBallot(),
            ScoreBallot(weight=0),
            ScoreBallot(scores={"D": 2, "E": 2}, weight=2),
            ScoreBallot(scores={"D": 2, "E": 2, "F": 3.1}, weight=2),
            ScoreBallot(),
            ScoreBallot(weight=0),
            ScoreBallot(scores={"G": 2, "H": 2}, weight=2),
            ScoreBallot(scores={"G": 2, "H": 2, "I": 3.1}, weight=2),
            ScoreBallot(),
            ScoreBallot(weight=0),
        ],
        candidates=["A", "B", "C", "D", "E", "F", "G", "H", "I"],
    )

    assert set(summed_profile.candidates) == set(["A", "B", "C", "D", "E", "F", "G", "H", "I"])
    assert isinstance(summed_profile, ScoreProfile)
    assert true_summed_profile == summed_profile


def test_sum_rank_profiles():
    profile_1 = RankProfile(
        ballots=[
            RankBallot(ranking=({"A"}, {"B"}, {"C"}), weight=2),
            RankBallot(ranking=({"A", "B"}, frozenset(), {"D"}), voter_set={"Chris"}),
            RankBallot(),
            RankBallot(weight=0),
        ],
        candidates=["A", "B", "C", "D"],
        max_ranking_length=3,
    )

    profile_2 = RankProfile(
        ballots=[
            RankBallot(ranking=({"E"}, {"D"}, {"F"}, {"E"}), weight=2),
            RankBallot(ranking=({"D"}, {"E"}, {"F"}), weight=2),
            RankBallot(),
            RankBallot(weight=0),
        ],
        candidates=["D", "E", "F"],
        max_ranking_length=0,
    )

    profile_3 = RankProfile(
        ballots=[
            RankBallot(ranking=({"G"}, {"H"}, {"I"}, {"G"}), weight=2),
            RankBallot(ranking=({"G"}, {"H"}, {"I"}), weight=2),
            RankBallot(),
            RankBallot(weight=0),
        ],
        candidates=["G", "H", "I"],
        max_ranking_length=0,
    )
    summed_profile = sum_profiles([profile_1, profile_2, profile_3])
    true_summed_profile = RankProfile(
        ballots=[
            RankBallot(ranking=({"A"}, {"B"}, {"C"}), weight=2),
            RankBallot(ranking=({"A", "B"}, frozenset(), {"D"}), voter_set={"Chris"}),
            RankBallot(),
            RankBallot(weight=0),
            RankBallot(ranking=({"E"}, {"D"}, {"F"}, {"E"}), weight=2),
            RankBallot(ranking=({"D"}, {"E"}, {"F"}), weight=2),
            RankBallot(),
            RankBallot(weight=0),
            RankBallot(ranking=({"G"}, {"H"}, {"I"}, {"G"}), weight=2),
            RankBallot(ranking=({"G"}, {"H"}, {"I"}), weight=2),
            RankBallot(),
            RankBallot(weight=0),
        ],
        candidates=["A", "B", "C", "D", "E", "F", "G", "H", "I"],
        max_ranking_length=4,
    )

    assert set(summed_profile.candidates) == set(["A", "B", "C", "D", "E", "F", "G", "H", "I"])
    assert summed_profile.max_ranking_length == 4
    assert isinstance(summed_profile, RankProfile)
    assert true_summed_profile == summed_profile
