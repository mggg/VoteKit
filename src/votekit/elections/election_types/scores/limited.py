from typing import Optional

from votekit.elections._deprecation import _handle_deprecated_kwargs
from votekit.elections.election_types.scores.rating import GeneralRating
from votekit.pref_profile import ScoreProfile


class Limited(GeneralRating):
    r"""
    A scoring election with a per-voter budget constraint.

    Voters can score each candidate, but have a total budget of
    :math:`\text{budget} \le n_\text{seats}` points. Winners are those with the highest
    total score.

    Args:
        profile (ScoreProfile): Profile to conduct election on.
        n_seats (int, optional): Number of seats to elect. Defaults to 1.
        budget (float, optional): Total points allowed per voter. Defaults to 1.
        tiebreak (str, optional): Tiebreak method to use. Options are None and 'random'.
            Defaults to None, in which case a tie raises a ValueError.

    Raises:
        ValueError: If ``budget`` exceeds ``n_seats``.
    """

    def __init__(
        self,
        profile: ScoreProfile,
        n_seats: int | None = None,
        budget: float | None = None,
        tiebreak: Optional[str] = None,
        **kwargs,
    ):
        kwargs = _handle_deprecated_kwargs(kwargs, {"m": "n_seats", "k": "budget"})
        if "n_seats" in kwargs:
            if n_seats is not None:
                raise TypeError("Cannot pass both 'm' and 'n_seats'.")
            n_seats = kwargs.pop("n_seats")
        if "budget" in kwargs:
            if budget is not None:
                raise TypeError("Cannot pass both 'k' and 'budget'.")
            budget = kwargs.pop("budget")
        if n_seats is None:
            n_seats = 1
        if budget is None:
            budget = 1
        if budget > n_seats:
            raise ValueError("budget must be less than or equal to n_seats.")

        super().__init__(
            profile, n_seats=n_seats, per_candidate_limit=budget, budget=budget, tiebreak=tiebreak
        )
