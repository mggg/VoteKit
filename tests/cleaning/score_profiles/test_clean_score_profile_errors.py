from typing import cast

import pytest

from votekit.ballot import RankBallot
from votekit.cleaning import clean_score_profile
from votekit.pref_profile import ProfileError, RankProfile, ScoreProfile


def test_clean_ranked_error():
    profile = RankProfile(
        ballots=[
            RankBallot(weight=0),
        ]
    )

    with pytest.raises(ProfileError, match="Profile must be a ScoreProfile."):
        clean_score_profile(cast(ScoreProfile, profile), lambda x: x)
