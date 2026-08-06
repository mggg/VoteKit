import pytest

from votekit.ballot import RankBallot, ScoreBallot
from votekit.cleaning import truncate_rank_profile
from votekit.pref_profile import CleanedRankProfile, ProfileError, RankProfile, ScoreProfile

profile = RankProfile(
    ballots=[
        RankBallot(ranking=[{"A"}, {"overvote"}, {"B"}, {"C"}], weight=1),
        RankBallot(ranking=[{"A"}, {"B"}, {"C"}], weight=2),
        RankBallot(ranking=[{"undervote"}, {"C"}], weight=3),
        RankBallot(ranking=[{"A"}, {"B"}], weight=0),
    ]
)


def test_truncate_rank_profile_at_candidate_or_marker():
    cleaned_profile = truncate_rank_profile(["overvote", "undervote"], profile)

    assert isinstance(cleaned_profile, CleanedRankProfile)
    assert cleaned_profile.parent_profile == profile
    assert cleaned_profile.ballots == (
        RankBallot(ranking=[{"A"}], weight=1),
        RankBallot(ranking=[{"A"}, {"B"}, {"C"}], weight=2),
    )
    assert cleaned_profile.no_rank_altr_idxs == {2}
    assert cleaned_profile.nonempty_altr_idxs == {0}
    assert cleaned_profile.unaltr_idxs == {1, 3}
    assert cleaned_profile.no_wt_altr_idxs == set()


def test_truncate_rank_profile_can_retain_empty_and_zero_weight_ballots():
    cleaned_profile = truncate_rank_profile(
        "overvote",
        profile,
        remove_empty_ballots=False,
        remove_zero_weight_ballots=False,
    )

    assert cleaned_profile.ballots == (
        RankBallot(ranking=[{"A"}], weight=1),
        RankBallot(ranking=[{"A"}, {"B"}, {"C"}], weight=2),
        RankBallot(ranking=[{"undervote"}, {"C"}], weight=3),
        RankBallot(ranking=[{"A"}, {"B"}], weight=0),
    )


def test_truncate_rank_profile_requires_rank_profile():
    score_profile = ScoreProfile(ballots=[ScoreBallot(scores={"A": 1})])

    with pytest.raises(ProfileError, match="Profile must be a RankProfile."):
        truncate_rank_profile("overvote", score_profile)  # type: ignore[arg-type]
