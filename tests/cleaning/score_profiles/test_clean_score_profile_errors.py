import pytest

from votekit.pref_profile import RankProfile, ProfileError
from votekit.ballot import RankBallot
from votekit.cleaning import clean_score_profile


def test_clean_ranked_error():
    profile = RankProfile(
        ballots=[
            RankBallot(weight=0),
        ]
    )

    with pytest.raises(ProfileError, match="Profile must be a ScoreProfile."):
        clean_score_profile(profile, lambda x: x)
