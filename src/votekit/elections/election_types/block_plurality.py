from typing import Literal

from votekit.elections._deprecation import _handle_deprecated_kwargs
from votekit.elections.election_state import ElectionState
from votekit.elections.election_types.ranking.block_plurality import _RankedBlockPlurality
from votekit.elections.election_types.scores.block_plurality import (
    _ScoreBlockPlurality,
)
from votekit.models import Election
from votekit.pref_profile import RankProfile, ScoreProfile


class BlockPlurality(Election[RankProfile | ScoreProfile]):
    """
    Block plurality election. Supports both ranked and score profiles.

    When given a ``RankProfile``, the top ``budget`` ranked candidates each receive 1 point
    and all others receive 0. When given a ``ScoreProfile``, voters may give at most 1 point
    to each of ``budget`` candidates.

    Construction returns a concrete ranked or score election. The factory's own election hooks
    raise NotImplementedError if called directly.

    Args:
        profile (RankProfile | ScoreProfile): Profile to conduct election on.
        n_seats (int | None, optional): Number of seats to elect. Defaults to None, which selects 1.
        budget (int | None, optional): Number of positions or approvals that count. Defaults to
            None, which results in ``n_seats``.
        tiebreak (str | None, optional): Tiebreak method to use. Options are None and 'random'.
            Defaults to None, in which case a tie raises a ValueError.
        scoring_tie_convention (Literal["high", "average", "low"], optional): How to award
            points for tied rankings. Only used when ``profile`` is a ``RankProfile``.
            Defaults to "low".
        rng_seed (int | None, optional): Seed for random number generator. An integer seed produces
            the same output given identical inputs. Defaults to None for non-deterministic results.

    Raises:
        TypeError: If ``profile`` is not a ``RankProfile`` or ``ScoreProfile``.
    """

    def __init__(
        self,
        profile: RankProfile | ScoreProfile,
        n_seats: int | None = None,
        budget: int | None = None,
        tiebreak: str | None = None,
        scoring_tie_convention: Literal["high", "average", "low"] = "low",
        *,
        rng_seed: int | None = None,
        **kwargs,
    ) -> None:
        # Never called; __new__ returns a different type.
        # Added as a stub for the type checker
        ...

    def __new__(
        cls,
        profile: RankProfile | ScoreProfile,
        n_seats: int | None = None,
        budget: int | None = None,
        tiebreak: str | None = None,
        scoring_tie_convention: Literal["high", "average", "low"] = "low",
        *,
        rng_seed: int | None = None,
        **kwargs,
    ) -> _RankedBlockPlurality | _ScoreBlockPlurality:
        kwargs = _handle_deprecated_kwargs(kwargs, {"m": "n_seats"})
        if "n_seats" in kwargs:
            if n_seats is not None:
                raise TypeError("Cannot pass both 'm' and 'n_seats'.")
            n_seats = kwargs.pop("n_seats")
        if n_seats is None:
            n_seats = 1
        if isinstance(profile, RankProfile):
            return _RankedBlockPlurality(
                profile,
                n_seats=n_seats,
                budget=budget,
                tiebreak=tiebreak,
                scoring_tie_convention=scoring_tie_convention,
                rng_seed=rng_seed,
            )
        elif isinstance(profile, ScoreProfile):
            return _ScoreBlockPlurality(
                profile,
                n_seats=n_seats,
                budget=budget,
                tiebreak=tiebreak,
                rng_seed=rng_seed,
            )
        else:
            raise TypeError(
                f"profile must be a RankProfile or ScoreProfile, got {type(profile).__name__}."
            )

    def _validate_params(self, profile: RankProfile | ScoreProfile) -> None:
        """
        Reject parameter validation on the factory itself.

        Raises:
            NotImplementedError: Validation belongs to the returned concrete election.
        """
        raise NotImplementedError("Use the concrete election returned by BlockPlurality().")

    def _validate_profile(self, profile: RankProfile | ScoreProfile) -> None:
        """
        Reject profile validation on the factory itself.

        Raises:
            NotImplementedError: Validation belongs to the returned concrete election.
        """
        raise NotImplementedError("Use the concrete election returned by BlockPlurality().")

    def _run_step(
        self,
        profile: RankProfile | ScoreProfile,
        prev_state: ElectionState,
        store_states: bool = False,
    ) -> RankProfile | ScoreProfile:
        """
        Reject election execution on the factory itself.

        Raises:
            NotImplementedError: Execution belongs to the returned concrete election.
        """
        raise NotImplementedError("Use the concrete election returned by BlockPlurality().")

    def _is_finished(self) -> bool:
        """
        Reject completion checks on the factory itself.

        Raises:
            NotImplementedError: Completion state belongs to the returned concrete election.
        """
        raise NotImplementedError("Use the concrete election returned by BlockPlurality().")
