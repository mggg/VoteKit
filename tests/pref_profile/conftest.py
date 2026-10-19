from __future__ import annotations

from collections.abc import Callable

import pytest

from votekit.pref_profile import RankProfile, ScoreProfile
from votekit.pref_profile.profile_matrix import RankMatrix, ScoreMatrix


@pytest.fixture
def unexpected_dataframe_conversion() -> Callable[..., None]:
    """
    Return a replacement that fails if an operation calls to_df().
    """

    def reject_dataframe_conversion(*args: object, **kwargs: object) -> None:
        raise AssertionError(
            "Internal profile operation must not construct a dataframe via to_df()"
        )

    return reject_dataframe_conversion


@pytest.fixture
def forbid_dataframe_conversion(
    monkeypatch: pytest.MonkeyPatch, unexpected_dataframe_conversion: Callable[..., None]
) -> Callable[[], None]:
    """
    Return a to_df() guard to activate after preparing inputs and expected results.
    """

    def forbid() -> None:
        for profile_type in (RankProfile, ScoreProfile, RankMatrix, ScoreMatrix):
            monkeypatch.setattr(profile_type, "to_df", unexpected_dataframe_conversion)

    return forbid
