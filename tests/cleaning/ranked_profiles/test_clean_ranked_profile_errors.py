import pytest

from votekit.pref_profile import PreferenceProfile, ProfileError
from votekit.ballot import Ballot
from votekit.cleaning import clean_ranked_profile


def test_clean_ranked_error():
    profile = PreferenceProfile(
        ballots=[
            Ballot(
                ranking=[
                    {"A"},
                ],
                weight=1,
                scores={"A": 1},
            ),
            Ballot(weight=0),
            Ballot(ranking=[{"C"}, {"B"}, {"A"}], weight=3),
        ]
    )

    with pytest.raises(ProfileError, match="Profile must only contain ranked ballots."):
        clean_ranked_profile(profile, lambda x: x)
