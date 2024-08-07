from votekit.elections import GeneralRating
from votekit import PreferenceProfile, Ballot
import pytest


def test_init():
    e = GeneralRating(
        PreferenceProfile(ballots=[Ballot(scores={"A": 2, "B": 2, "C": 0})]),
        m=2,
        L=2,
        k=4,
    )

    assert e.get_elected(1) == (frozenset({"A", "B"}),)


def test_errors():
    with pytest.raises(ValueError):  # k must be positive
        GeneralRating(PreferenceProfile(), k=0)

    with pytest.raises(ValueError):  # L must be positive
        GeneralRating(PreferenceProfile(), L=0)


def test_validate_profile():
    with pytest.raises(TypeError):  # must be less than limit L
        profile = PreferenceProfile(ballots=[Ballot(scores={"A": 3})])
        GeneralRating(profile, m=2, L=2)

    with pytest.raises(TypeError):  # must be non-negative
        profile = PreferenceProfile(ballots=[Ballot(scores={"A": -3})])
        GeneralRating(profile, m=2, L=2)

    with pytest.raises(TypeError):  # must have scores
        profile = PreferenceProfile(ballots=[Ballot()])
        GeneralRating(profile, m=2, L=2)

    with pytest.raises(TypeError):  # must be less than budget k
        profile = PreferenceProfile(ballots=[Ballot()])
        GeneralRating(profile, m=2, L=2)
