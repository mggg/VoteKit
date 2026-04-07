from typing import Optional

from votekit.elections.election_types.scores.limited import Limited
from votekit.pref_profile import ScoreProfile


class Cumulative(Limited):
    """
    Scoring election where voters distribute up to ``n_seats`` points total.

    Voters can score each candidate, but have a total budget of :math:`n_\text{seats}` points.
    Winners are those with the highest total score.

    Args:
        profile (ScoreProfile): Profile to conduct election on.
        n_seats (int, optional): Number of seats to elect. Defaults to 1.
        tiebreak (str, optional): Tiebreak method to use. Options are None and 'random'.
            Defaults to None, in which case a tie raises a ValueError.
    """

    def __init__(self, profile: ScoreProfile, n_seats: int = 1, tiebreak: Optional[str] = None):
        super().__init__(profile, n_seats=n_seats, budget=n_seats, tiebreak=tiebreak)
