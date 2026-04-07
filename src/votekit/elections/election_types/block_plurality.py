from typing import Literal, Optional, Union

from votekit.elections.election_types.ranking.block_plurality import _RankedBlockPlurality
from votekit.elections.election_types.scores.block_plurality import (
    _ScoreBlockPlurality,
)
from votekit.models import Election
from votekit.pref_profile import RankProfile, ScoreProfile


class BlockPlurality(Election):
    """
    Block plurality election. Supports both ranked and score profiles.

    When given a ``RankProfile``, the top ``budget`` ranked candidates each receive 1 point
    and all others receive 0. When given a ``ScoreProfile``, voters may give at most 1 point
    to each of ``budget`` candidates.

    Args:
        profile (RankProfile | ScoreProfile): Profile to conduct election on.
        n_seats (int, optional): Number of seats to elect. Defaults to 1.
        budget (int, optional): Number of positions or approvals that count. Defaults to
            None, which results in ``n_seats``.
        tiebreak (str, optional): Tiebreak method to use. Options are None and 'random'.
            Defaults to None, in which case a tie raises a ValueError.
        scoring_tie_convention (Literal["high", "average", "low"], optional): How to award
            points for tied rankings. Only used when ``profile`` is a ``RankProfile``.
            Defaults to "low".

    Raises:
        TypeError: If ``profile`` is not a ``RankProfile`` or ``ScoreProfile``.
    """

    def __init__(
        self,
        profile: Union[RankProfile, ScoreProfile],
        n_seats: int = 1,
        budget: Optional[int] = None,
        tiebreak: Optional[str] = None,
        scoring_tie_convention: Literal["high", "average", "low"] = "low",
    ) -> None:
        # Never called; __new__ returns a different type.
        # Added as a stub for the type checker
        ...

    def __new__(
        cls,
        profile: Union[RankProfile, ScoreProfile],
        n_seats: int = 1,
        budget: Optional[int] = None,
        tiebreak: Optional[str] = None,
        scoring_tie_convention: Literal["high", "average", "low"] = "low",
    ) -> _RankedBlockPlurality | _ScoreBlockPlurality:
        if isinstance(profile, RankProfile):
            return _RankedBlockPlurality(
                profile,
                n_seats=n_seats,
                budget=budget,
                tiebreak=tiebreak,
                scoring_tie_convention=scoring_tie_convention,
            )
        elif isinstance(profile, ScoreProfile):
            return _ScoreBlockPlurality(
                profile,
                n_seats=n_seats,
                budget=budget,
                tiebreak=tiebreak,
            )
        else:
            raise TypeError(
                f"profile must be a RankProfile or ScoreProfile, got {type(profile).__name__}."
            )
