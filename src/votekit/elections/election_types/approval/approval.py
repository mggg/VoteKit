from ..scores import GeneralRating
from ....pref_profile import PreferenceProfile
from typing import Optional


class Approval(GeneralRating):
    """
    Approval election. Standard approval voting lets voters choose any subset of candidates to
    approve.  Winners are the :math:`m` candidates who received the most approval votes.

    Args:
        profile (PreferenceProfile): Profile to conduct election on.
        m (int, optional): Number of seats to elect. Defaults to 1.
        tiebreak (str, optional): Tiebreak method to use. Options are None and 'random'.
            Defaults to None, in which case a tie raises a ValueError.

    """

    def __init__(
        self, profile: PreferenceProfile, m: int = 1, tiebreak: Optional[str] = None
    ):
        # limit one per candidate,  but no total budget limit
        super().__init__(profile, m=m, L=1, tiebreak=tiebreak)


class BlocPlurality(GeneralRating):
    """
    Like approval voting, but there is a user-specified limit of :math:`k` approvals per voter.
    Most commonly, this would be run with :math:`k=m`.

    Args:
        profile (PreferenceProfile): Profile to conduct election on.
        m (int, optional): Number of seats to elect. Defaults to 1.
        k (int, optional): Total budget per voter. Defaults to None, which results in ``m``
            approvals per voter.
        tiebreak (str, optional): Tiebreak method to use. Options are None and 'random'.
            Defaults to None, in which case a tie raises a ValueError.

    """

    def __init__(
        self,
        profile: PreferenceProfile,
        m: int = 1,
        k: Optional[int] = None,
        tiebreak: Optional[str] = None,
    ):
        if not k:
            k = m
        # limit one per candidate, total budget limit k
        super().__init__(profile, m=m, L=1, k=k, tiebreak=tiebreak)
