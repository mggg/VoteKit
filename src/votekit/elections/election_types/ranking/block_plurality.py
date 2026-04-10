from typing import Literal, Optional

from votekit.elections.election_types.ranking.borda import Borda
from votekit.pref_profile import RankProfile


class _RankedBlockPlurality(Borda):
    r"""
    Ranking-based block plurality. The top ``budget`` ranked candidates each receive 1 point.

    Applies the scoring vector :math:`(\underbrace{1,\dots,1}_{\text{budget}}, 0, \dots, 0)`
    to each ranked ballot. Winners are the :math:`n_\text{seats}` candidates with the highest
    totals.

    Args:
        profile (RankProfile): Profile to conduct election on.
        n_seats (int, optional): Number of seats to elect. Defaults to 1.
        budget (int, optional): Number of top-ranked positions that count. Defaults to None,
            which results in ``n_seats`` positions counting.
        tiebreak (str, optional): Tiebreak method to use. Options are None, 'random', and
            'first_place'. Defaults to None, in which case a tie raises a ValueError.
        scoring_tie_convention (Literal["high", "average", "low"], optional): How to award
            points for tied rankings. Defaults to "low", where candidates tied for a position
            receive the lowest possible points for that position. See :class:`Borda` for details.

    Raises:
        ValueError: If ``budget`` exceeds ``profile.max_ranking_length``.
    """

    def __init__(
        self,
        profile: RankProfile,
        n_seats: int = 1,
        budget: Optional[int] = None,
        tiebreak: Optional[str] = None,
        scoring_tie_convention: Literal["high", "average", "low"] = "low",
    ):
        if budget is None or budget == 0:
            budget = n_seats
        assert profile.max_ranking_length is not None
        if budget > profile.max_ranking_length:
            raise ValueError(
                f"budget ({budget}) cannot exceed max_ranking_length "
                f"({profile.max_ranking_length})."
            )
        score_vector = [1] * budget + [0] * (profile.max_ranking_length - budget)
        super().__init__(
            profile,
            n_seats=n_seats,
            score_vector=score_vector,
            tiebreak=tiebreak,
            scoring_tie_convention=scoring_tie_convention,
        )
