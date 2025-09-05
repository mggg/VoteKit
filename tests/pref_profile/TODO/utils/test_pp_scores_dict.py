from votekit.ballot import Ballot
from votekit.pref_profile import PreferenceProfile, ProfileError
from votekit.pref_profile.utils import profile_to_scores_dict
import pytest


def test_to_scores_dict():
    profile = PreferenceProfile(
        ballots=(
            Ballot(ranking=({"A"}, {"B"})),
            Ballot(ranking=({"A"}, {"B"}), weight=3 / 2),
            Ballot(ranking=({"C"}, {"B"}), scores={"A": 4}, weight=2),
            Ballot(scores={"A": 4}),
        )
    )
    rv = profile_to_scores_dict(profile, standardize=False)
    assert rv[(("A", 4),)] == 3
    assert rv[None] == 5 / 2

    rv = profile_to_scores_dict(profile, standardize=True)
    assert rv[(("A", 4),)] == 6 / 11
    assert rv[None] == 5 / 11


def test_scores_dict_error():
    profile = PreferenceProfile(ballots=(Ballot(ranking=({"A"}, {"B"})),))

    with pytest.raises(
        ProfileError,
        match=(
            "You are trying to convert a profile that contains "
            "no scores to a scores_dict."
        ),
    ):
        profile_to_scores_dict(profile)
