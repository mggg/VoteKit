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
    with pytest.raises(ValueError, match="m must be positive."):
        GeneralRating(PreferenceProfile(), m=0)

    with pytest.raises(ValueError, match="L must be positive."):
        GeneralRating(PreferenceProfile(), L=0)

    with pytest.raises(ValueError, match="L must be less than or equal to k."):
        GeneralRating(PreferenceProfile(), L=4, k=2)


def test_validate_profile():
    with pytest.raises(TypeError, match="violates score limit"):
        profile = PreferenceProfile(ballots=[Ballot(scores={"A": 3, "B": 2})])
        GeneralRating(profile, m=2, L=2)

    with pytest.raises(TypeError, match="must have non-negative scores."):
        profile = PreferenceProfile(ballots=[Ballot(scores={"A": -3, "B": 2, "C": 2})])
        GeneralRating(profile, m=2, L=2)

    with pytest.raises(TypeError, match="All ballots must have score dictionary."):
        profile = PreferenceProfile(ballots=[Ballot(), Ballot(scores={"A": 2, "B": 2})])
        GeneralRating(profile, m=2, L=2)

    with pytest.raises(TypeError, match="violates total score budget"):
        profile = PreferenceProfile(ballots=[Ballot(scores={"A": 2, "C": 2, "B": 2})])
        GeneralRating(profile, L=2, k=2)
