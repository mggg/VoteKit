from typing import cast

import pytest

from votekit.ballot import ScoreBallot
from votekit.cleaning import clean_rank_profile
from votekit.pref_profile import ProfileError, RankProfile, ScoreProfile


def test_clean_ranked_error():
    profile = ScoreProfile(
        ballots=[
            ScoreBallot(weight=0),
        ]
    )

    with pytest.raises(ProfileError, match="Profile must be a RankProfile."):
        clean_rank_profile(cast(RankProfile, profile), lambda x: x)
