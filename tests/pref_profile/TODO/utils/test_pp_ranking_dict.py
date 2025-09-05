from votekit.ballot import Ballot
from votekit.pref_profile import PreferenceProfile, ProfileError
from votekit.pref_profile.utils import profile_to_ranking_dict
import pytest


def test_to_ranking_dict():
    profile = PreferenceProfile(
        ballots=(
            Ballot(ranking=({"A"}, {"B"})),
            Ballot(ranking=({"A"}, {"B"}), weight=3 / 2),
            Ballot(ranking=({"C"}, {"B"}), scores={"A": 4}, weight=2),
            Ballot(scores={"A": 4}),
        )
    )

    rv = profile_to_ranking_dict(profile, standardize=False)
    assert rv[(frozenset({"A"}), frozenset({"B"}))] == 5 / 2
    assert rv[(frozenset({"C"}), frozenset({"B"}))] == 2 / 1
    assert rv[None] == 1

    rv = profile_to_ranking_dict(profile, standardize=True)
    assert rv[(frozenset({"A"}), frozenset({"B"}))] == 5 / 11
    assert rv[(frozenset({"C"}), frozenset({"B"}))] == 4 / 11
    assert rv[None] == 2 / 11


def test_ranking_dict_warn():
    profile = PreferenceProfile(ballots=(Ballot(scores={"A": 4}),))

    with pytest.raises(
        ProfileError,
        match=(
            "You are trying to convert a profile that contains "
            "no rankings to a ranking_dict."
        ),
    ):
        profile_to_ranking_dict(profile)
