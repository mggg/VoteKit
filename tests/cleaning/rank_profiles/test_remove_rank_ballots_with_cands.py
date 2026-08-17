import pytest

from votekit.ballot import RankBallot, ScoreBallot
from votekit.cleaning import remove_rank_ballots_with_cands
from votekit.pref_profile import CleanedRankProfile, ProfileError, RankProfile, ScoreProfile

profile = RankProfile(
    ballots=[
        RankBallot(ranking=[{"A"}, {"B"}], weight=1),
        RankBallot(ranking=[{"A"}, {"overvote"}, {"C"}], weight=2),
        RankBallot(ranking=[{"C"}, {"A"}], weight=3),
        RankBallot(ranking=[{"undervote", "B"}, {"C"}], weight=4),
        RankBallot(ranking=[{"A"}], weight=0),
    ]
)


def test_remove_rank_ballots_with_candidate():
    cleaned_profile = remove_rank_ballots_with_cands("overvote", profile)

    assert isinstance(cleaned_profile, CleanedRankProfile)
    assert cleaned_profile.parent_profile == profile
    assert cleaned_profile.ballots == (
        RankBallot(ranking=[{"A"}, {"B"}], weight=1),
        RankBallot(ranking=[{"C"}, {"A"}], weight=3),
        RankBallot(ranking=[{"undervote", "B"}, {"C"}], weight=4),
    )
    assert cleaned_profile.candidates == profile.candidates
    assert cleaned_profile.no_wt_altr_idxs == set()
    assert cleaned_profile.no_rank_altr_idxs == {1}
    assert cleaned_profile.nonempty_altr_idxs == set()
    assert cleaned_profile.unaltr_idxs == {0, 2, 3, 4}


def test_remove_rank_ballots_with_multiple_candidates_and_ties():
    cleaned_profile = remove_rank_ballots_with_cands(["overvote", "undervote"], profile)

    assert cleaned_profile.ballots == (
        RankBallot(ranking=[{"A"}, {"B"}], weight=1),
        RankBallot(ranking=[{"C"}, {"A"}], weight=3),
    )
    assert cleaned_profile.no_rank_altr_idxs == {1, 3}
    assert cleaned_profile.unaltr_idxs == {0, 2, 4}


def test_remove_rank_ballots_with_candidate_retains_zero_weight_when_requested():
    cleaned_profile = remove_rank_ballots_with_cands(
        "overvote",
        profile,
        remove_zero_weight_ballots=False,
    )

    assert cleaned_profile.ballots == (
        RankBallot(ranking=[{"A"}, {"B"}], weight=1),
        RankBallot(ranking=[{"C"}, {"A"}], weight=3),
        RankBallot(ranking=[{"undervote", "B"}, {"C"}], weight=4),
        RankBallot(ranking=[{"A"}], weight=0),
    )


def test_remove_rank_ballots_with_cands_requires_rank_profile():
    score_profile = ScoreProfile(ballots=[ScoreBallot(scores={"A": 1})])

    with pytest.raises(ProfileError, match="Profile must be a RankProfile."):
        remove_rank_ballots_with_cands("A", score_profile)  # type: ignore[arg-type]
