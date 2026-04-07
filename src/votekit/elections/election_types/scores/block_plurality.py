import warnings
from typing import Optional

from votekit.elections.election_types.scores import GeneralRating
from votekit.pref_profile import ScoreProfile


class _ScoreBlockPlurality(GeneralRating):
    """
    Score-based block plurality. Voters may give at most 1 point to each of ``budget`` candidates.

    Most commonly run with ``budget=n_seats``.

    Args:
        profile (ScoreProfile): Profile to conduct election on.
        n_seats (int, optional): Number of seats to elect. Defaults to 1.
        budget (int, optional): Total candidates a voter may score. Defaults to None, which
            results in ``n_seats``.
        tiebreak (str, optional): Tiebreak method to use. Options are None and 'random'.
            Defaults to None, in which case a tie raises a ValueError.
    """

    def __init__(
        self,
        profile: ScoreProfile,
        n_seats: int = 1,
        budget: Optional[int] = None,
        tiebreak: Optional[str] = None,
    ):
        if budget is None or budget == 0:
            budget = n_seats
        super().__init__(
            profile, n_seats=n_seats, per_candidate_limit=1, budget=budget, tiebreak=tiebreak
        )


class BlocPlurality(_ScoreBlockPlurality):
    """
    Deprecated. Use :class:`BlockPlurality` instead.

    Args:
        profile (ScoreProfile): Profile to conduct election on.
        n_seats (int, optional): Number of seats to elect. Defaults to 1.
        budget (int, optional): Total candidates a voter may score. Defaults to None, which
            results in ``n_seats``.
        tiebreak (str, optional): Tiebreak method to use. Options are None and 'random'.
            Defaults to None, in which case a tie raises a ValueError.
    """

    def __init__(
        self,
        profile: ScoreProfile,
        n_seats: int = 1,
        budget: Optional[int] = None,
        tiebreak: Optional[str] = None,
    ):
        warnings.warn(
            "BlocPlurality has been renamed to BlockPlurality. "
            "BlocPlurality will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(profile, n_seats=n_seats, budget=budget, tiebreak=tiebreak)
