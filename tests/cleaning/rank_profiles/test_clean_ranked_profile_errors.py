import pytest

from votekit.pref_profile import ScoreProfile, ProfileError
from votekit.ballot import ScoreBallot
from votekit.cleaning import clean_rank_profile


def test_clean_ranked_error():
    profile = ScoreProfile(
        ballots=[
            ScoreBallot(weight=0),
        ]
    )

    with pytest.raises(ProfileError, match="Profile must be a RankProfile."):
        clean_rank_profile(profile, lambda x: x)
