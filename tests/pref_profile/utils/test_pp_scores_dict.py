from votekit.ballot import ScoreBallot, RankBallot
from votekit.pref_profile import ScoreProfile, RankProfile, ProfileError
from votekit.pref_profile.utils import score_profile_to_scores_dict
import pytest


def test_to_scores_dict():
    profile = ScoreProfile(
        ballots=(
            ScoreBallot(scores={"A": 4}, weight=2),
            ScoreBallot(scores={"A": 3}),
        )
    )
    rv = score_profile_to_scores_dict(profile, standardize=False)
    assert rv[(("A", 4),)] == 2
    assert rv[(("A", 3),)] == 1

    rv = score_profile_to_scores_dict(profile, standardize=True)
    assert rv[(("A", 4),)] == 2 / 3
    assert rv[(("A", 3),)] == 1 / 3


def test_scores_dict_error():
    profile = RankProfile(ballots=(RankBallot(ranking=({"A"}, {"B"})),))

    with pytest.raises(
        ProfileError,
        match=("Profile must be a ScoreProfile"),
    ):
        score_profile_to_scores_dict(profile)
