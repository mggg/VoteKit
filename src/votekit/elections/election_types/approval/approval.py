from typing import Optional

from votekit.elections._deprecation import _handle_deprecated_kwargs
from votekit.elections.election_types.scores import GeneralRating
from votekit.pref_profile import ScoreProfile


class Approval(GeneralRating):
    """
    Approval election. Standard approval voting lets voters choose any subset of candidates to
    approve.  Winners are the :math:`n_\text{seats}` candidates who received the most approval
    votes.

    Args:
        profile (ScoreProfile): Profile to conduct election on.
        n_seats (int, optional): Number of seats to elect. Defaults to 1.
        tiebreak (str, optional): Tiebreak method to use. Options are None and 'random'.
            Defaults to None, in which case a tie raises a ValueError.

    """

    def __init__(
        self,
        profile: ScoreProfile,
        n_seats: int | None = None,
        tiebreak: Optional[str] = None,
        **kwargs,
    ):
        kwargs = _handle_deprecated_kwargs(kwargs, {"m": "n_seats"})
        if "n_seats" in kwargs:
            if n_seats is not None:
                raise TypeError("Cannot pass both 'm' and 'n_seats'.")
            n_seats = kwargs.pop("n_seats")
        if n_seats is None:
            n_seats = 1
        # limit one per candidate,  but no total budget limit
        super().__init__(profile, n_seats=n_seats, per_candidate_limit=1, tiebreak=tiebreak)
