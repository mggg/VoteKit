from typing import Optional

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

    def __init__(self, profile: ScoreProfile, n_seats: int = 1, tiebreak: Optional[str] = None):
        # limit one per candidate,  but no total budget limit
        super().__init__(profile, n_seats=n_seats, per_candidate_limit=1, tiebreak=tiebreak)


class BlocPlurality(GeneralRating):
    """
    Like approval voting, but there is a user-specified limit of ``budget`` approvals per voter.

    Note: Since this is an approval-based election, the per-candidate approval is limited to 1.
    Most commonly, this would be run with ``budget=n_seats``.

    Args:
        profile (ScoreProfile): Profile to conduct election on.
        n_seats (int, optional): Number of seats to elect. Defaults to 1.
        budget (int, optional): Total budget per voter. Defaults to None, which results in
            ``n_seats`` approvals per voter.
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
        # Limit per-candidate approval to 1, and total approvals per voter to budget.
        super().__init__(
            profile, n_seats=n_seats, per_candidate_limit=1, budget=budget, tiebreak=tiebreak
        )
