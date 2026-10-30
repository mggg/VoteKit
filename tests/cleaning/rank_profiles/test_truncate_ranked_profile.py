import pytest

from votekit.ballot import RankBallot, ScoreBallot
from votekit.cleaning import truncate_at_cand_rank_profile
from votekit.pref_profile import CleanedRankProfile, RankProfile, ScoreProfile


@pytest.fixture
def profile():
    return RankProfile(
        ballots=[
            RankBallot(ranking=[{"A"}, {"overvote"}, {"B"}, {"C"}], weight=1),
            RankBallot(ranking=[{"A"}, {"B"}, {"undervote"}], weight=2),
            RankBallot(ranking=[{"undervote"}, {"C"}], weight=3),
            RankBallot(ranking=[{"A"}, {"B"}], weight=0),
        ]
    )


@pytest.fixture
def profile_with_ties():
    return RankProfile(
        ballots=[
            RankBallot(ranking=[{"A"}, {"B", "C"}], weight=1),
            RankBallot(ranking=[{"A"}, {"B"}, {"C"}], weight=2),
            RankBallot(ranking=[{"A", "C"}], weight=3),
            RankBallot(ranking=[{"A"}, {"B"}], weight=0),
        ]
    )


def test_truncate_at_cand_rank_profile_default_behavior(profile):
    cleaned_profile = truncate_at_cand_rank_profile(["overvote", "undervote"], profile)

    assert isinstance(cleaned_profile, CleanedRankProfile)
    assert cleaned_profile.parent_profile == profile
    assert cleaned_profile.ballots == (
        RankBallot(ranking=[{"A"}], weight=1),
        RankBallot(ranking=[{"A"}, {"B"}], weight=2),
    )
    assert cleaned_profile.no_rank_altr_idxs == {2}
    assert cleaned_profile.nonempty_altr_idxs == {0, 1}
    assert cleaned_profile.unaltr_idxs == {3}
    assert cleaned_profile.no_wt_altr_idxs == set()
    # default behavior is to reduce the profile's candidates to exclude truncated candidates
    assert set(profile.candidates) == set(("A", "B", "C", "overvote", "undervote"))
    assert set(cleaned_profile.candidates_cast) == set(("A", "B"))
    assert set(cleaned_profile.candidates) == set(("A", "B", "C"))


def test_truncate_at_cand_rank_profile_can_retain_null_and_zero_weight_ballots(profile):
    cleaned_profile = truncate_at_cand_rank_profile(
        "overvote",
        profile,
        remove_null_ballots=False,
        remove_zero_weight_ballots=False,
    )

    assert cleaned_profile.ballots == (
        RankBallot(ranking=[{"A"}], weight=1),
        RankBallot(ranking=[{"A"}, {"B"}, {"undervote"}], weight=2),
        RankBallot(ranking=[{"undervote"}, {"C"}], weight=3),
        RankBallot(ranking=[{"A"}, {"B"}], weight=0),
    )
    assert cleaned_profile.no_rank_altr_idxs == set()
    assert cleaned_profile.nonempty_altr_idxs == {0}
    assert cleaned_profile.unaltr_idxs == {1, 2, 3}
    assert cleaned_profile.no_wt_altr_idxs == set()


def test_truncate_at_cand_rank_profile_requires_rank_profile():
    score_profile = ScoreProfile(ballots=[ScoreBallot(scores={"A": 1})])

    with pytest.raises(TypeError, match="Profile must be a RankProfile."):
        truncate_at_cand_rank_profile("overvote", score_profile)  # type: ignore[arg-type]


def test_truncate_at_cand_rank_profile_with_ties(profile_with_ties):
    cleaned_profile = truncate_at_cand_rank_profile("C", profile_with_ties)

    assert cleaned_profile.ballots == (
        RankBallot(ranking=[{"A"}], weight=1),
        RankBallot(ranking=[{"A"}, {"B"}], weight=2),
    )

    assert cleaned_profile.no_rank_altr_idxs == {2}
    assert cleaned_profile.nonempty_altr_idxs == {0, 1}
    assert cleaned_profile.unaltr_idxs == {3}
    assert cleaned_profile.no_wt_altr_idxs == set()


def test_truncate_at_cand_rank_profile_retain_original_candidate_list(profile):
    cleaned_profile = truncate_at_cand_rank_profile(
        ["overvote", "undervote"],
        profile,
        retain_original_candidate_list=True,
    )
    assert profile.candidates == cleaned_profile.candidates


def test_truncate_at_cand_rank_profile_reduce_max_ranking_length(profile_with_ties):
    cleaned_profile = truncate_at_cand_rank_profile(
        "C",
        profile_with_ties,
        reduce_max_ranking_length=True,
    )
    assert profile_with_ties.max_ranking_length == 3
    assert cleaned_profile.max_ranking_length == 2


def test_truncate_at_cand_rank_profile_rejects_invalid_truncated_candidate(profile):
    with pytest.raises(TypeError, match=r"Boolean candidate\(s\) found in truncated"):
        truncate_at_cand_rank_profile(truncated=True, profile=profile)
