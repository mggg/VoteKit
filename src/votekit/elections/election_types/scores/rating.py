from __future__ import annotations

import random
from typing import Optional

import numpy as np

from votekit.ballot import ScoreBallot
from votekit.cleaning import remove_cand_score_profile
from votekit.elections._deprecation import _handle_deprecated_kwargs
from votekit.elections.election_state import ElectionState
from votekit.exceptions import ProfileError
from votekit.models import Election
from votekit.pref_profile import ScoreProfile
from votekit.utils import (
    elect_cands_from_set_ranking,
    score_profile_from_ballot_scores,
)


class GeneralRating(Election[ScoreProfile]):
    """
    General rating election. To fill :math:`n_\text{seats}` seats, voters score each candidate
    from :math:`0` to ``per_candidate_limit``.  There is also a total budget
    of ``budget`` points per voter. The :math:`n_\text{seats}` winners are those with the highest
    total score.

    Args:
        profile (ScoreProfile): Profile to conduct election on.
        n_seats (int, optional): Number of seats to elect. Defaults to 1.
        per_candidate_limit (float, optional): Rating per candidate limit. Defaults to 1.
        budget (float, optional): Budget per ballot limit. Defaults to None, in which
            case voters can score each candidate independently.
        tiebreak (str, optional): Tiebreak method to use. Options are None and 'random'.
            Defaults to None, in which case a tie raises a ValueError.
        rng_seed (int, optional)): seed for RNG, allows for reproducible results given the same
            inputs. Seed set to None by default, different results will be generated each time.

    """

    def __init__(
        self,
        profile: ScoreProfile,
        n_seats: int | None = None,
        per_candidate_limit: float | None = None,
        budget: Optional[float] = None,
        tiebreak: Optional[str] = None,
        *,
        rng_seed: Optional[int] = None,
        **kwargs,
    ):
        kwargs = _handle_deprecated_kwargs(kwargs, {"m": "n_seats", "k": "per_candidate_limit"})
        if "n_seats" in kwargs:
            if n_seats is not None:
                raise TypeError("Cannot pass both 'm' and 'n_seats'.")
            n_seats = kwargs.pop("n_seats")
        if "per_candidate_limit" in kwargs:
            if per_candidate_limit is not None:
                raise TypeError("Cannot pass both 'k' and 'per_candidate_limit'.")
            per_candidate_limit = kwargs.pop("per_candidate_limit")
        if n_seats is None:
            n_seats = 1
        if per_candidate_limit is None:
            per_candidate_limit = 1
        if n_seats <= 0:
            raise ValueError("n_seats must be positive.")
        self.n_seats = n_seats
        if tiebreak not in (None, "random"):
            raise ValueError("tiebreak must be None or 'random'.")
        if per_candidate_limit <= 0:
            raise ValueError("per_candidate_limit must be positive.")
        self.per_candidate_limit = per_candidate_limit
        if budget and budget <= 0:
            raise ValueError("budget must be positive.")
        if budget and per_candidate_limit > budget:
            raise ValueError("per_candidate_limit must be less than or equal to budget.")
        self.budget = budget
        self.tiebreak = tiebreak
        self._rng = random.Random(rng_seed)
        super().__init__(
            profile,
            score_function=score_profile_from_ballot_scores,
            sort_high_low=True,
        )

    def _validate_params(self, profile: ScoreProfile):
        """
        Validates election parameters against the profile.

        Args:
            profile (ScoreProfile): Profile of ballots.

        Raises:
            ValueError: If there are not enough candidates who received votes to fill
                the requested seats.
        """
        if len(profile.candidates_cast) < self.n_seats:
            raise ValueError("Not enough candidates received votes to be elected.")

    def _validate_profile(self, profile: ScoreProfile):
        """
        Validates that every ballot has a score dictionary and each voter has not gone over
        their score limit per candidate and total budget.

        Args:
            profile (ScoreProfile): Profile to validate.

        Raises:
            ProfileError: If profile is not a ScoreProfile or is empty.
            TypeError: If a ballot has no scores, negative scores, or violates limits.
        """
        if not isinstance(profile, ScoreProfile):
            raise ProfileError("Profile must be of type ScoreProfile.")

        matrix = profile._matrix
        if not len(matrix):
            raise ProfileError("Profile must contain at least one ballot.")

        column_positions = {
            candidate: column for column, candidate in enumerate(matrix.score_columns)
        }
        candidates = [
            candidate for candidate in profile.candidates if candidate in column_positions
        ]
        scores = matrix.scores[:, [column_positions[candidate] for candidate in candidates]]
        missing_scores = np.all(np.isnan(scores), axis=1)
        above_limit = np.any(scores > self.per_candidate_limit, axis=1)
        negative_scores = np.any(scores < 0, axis=1)
        above_budget = np.zeros(len(matrix), dtype=bool)
        if self.budget:
            # Use Python floats and sum, matching ScoreBallot's budget arithmetic.
            totals = np.fromiter(
                (sum(float(score) for score in row if not np.isnan(score)) for row in scores),
                dtype=float,
                count=len(matrix),
            )
            above_budget = totals > self.budget
        invalid_rows = np.flatnonzero(missing_scores | above_limit | negative_scores | above_budget)
        if not invalid_rows.size:
            return

        row = int(invalid_rows[0])
        if missing_scores[row]:
            raise TypeError("All ballots must have score dictionary.")
        data = matrix.voter_weight_data
        voters = data.voter_sets[row] if data.voter_sets is not None else frozenset()
        ballot = ScoreBallot(
            scores={
                candidate: score
                for candidate, score in zip(candidates, scores[row])
                if not np.isnan(score)
            },
            weight=data.weights[row],
            voter_set=voters,
        )
        if above_limit[row]:
            raise TypeError(
                f"Ballot {ballot} violates score limit {self.per_candidate_limit} per candidate."
            )
        if negative_scores[row]:
            raise TypeError(f"Ballot {ballot} must have non-negative scores.")
        raise TypeError(f"Ballot {ballot} violates total score budget {self.budget}.")

    def _is_finished(self):
        # single round election
        if len(self.election_states) == 2:
            return True
        return False

    def _run_step(
        self, profile: ScoreProfile, prev_state: ElectionState, store_states=False
    ) -> ScoreProfile:
        """
        Run one step of an election from the given profile and previous state.

        Args:
            profile (ScoreProfile): Profile of ballots.
            prev_state (ElectionState): The previous ElectionState.
            store_states (bool, optional): True if `self.election_states` should be updated with the
                ElectionState generated by this round. This should only be True when used by
                `self._run_election()`. Defaults to False.

        Returns:
            ScoreProfile: The profile of ballots after the round is completed.
        """
        # since score_profile_from_ballot_scores is the score function, the remaining cands from
        # round 0 are ranked by score
        # raises a ValueError is tiebreak is None and a tie occurs.
        elected, remaining, tie_resolution = elect_cands_from_set_ranking(
            prev_state.remaining, self.n_seats, tiebreak=self.tiebreak, rng=self._rng
        )

        new_profile = remove_cand_score_profile([c for s in elected for c in s], profile)

        if store_states:
            if self.score_function:
                scores = self.score_function(new_profile)
            else:
                raise ValueError()

            if tie_resolution:
                tiebreaks = {tie_resolution[0]: tie_resolution[1]}
            else:
                tiebreaks = {}

            new_state = ElectionState(
                round_number=1,  # single shot election
                remaining=remaining,
                elected=elected,
                scores=scores,
                tiebreaks=tiebreaks,
            )

            self.election_states.append(new_state)

        return new_profile


class Rating(GeneralRating):
    """
    Rating election. To fill :math:`n_\text{seats}` seats, voters score each candidate
    independently from :math:`0` to ``per_candidate_limit``.  The
    :math:`n_\text{seats}` winners are those with the highest total score.

    Args:
        profile (ScoreProfile): Profile to conduct election on.
        n_seats (int, optional): Number of seats to elect. Defaults to 1.
        per_candidate_limit (float, optional): Rating per candidate limit. Defaults to 1.
        tiebreak (str, optional): Tiebreak method to use. Options are None and 'random'.
            Defaults to None, in which case a tie raises a ValueError.
        rng_seed (int, optional)): seed for RNG, allows for reproducible results given the same
            inputs. Seed set to None by default, different results will be generated each time.

    """

    def __init__(
        self,
        profile: ScoreProfile,
        n_seats: int | None = None,
        per_candidate_limit: float | None = None,
        tiebreak: Optional[str] = None,
        *,
        rng_seed: Optional[int] = None,
        **kwargs,
    ):
        kwargs = _handle_deprecated_kwargs(kwargs, {"m": "n_seats", "k": "per_candidate_limit"})
        if "n_seats" in kwargs:
            if n_seats is not None:
                raise TypeError("Cannot pass both 'm' and 'n_seats'.")
            n_seats = kwargs.pop("n_seats")
        if "per_candidate_limit" in kwargs:
            if per_candidate_limit is not None:
                raise TypeError("Cannot pass both 'k' and 'per_candidate_limit'.")
            per_candidate_limit = kwargs.pop("per_candidate_limit")
        if n_seats is None:
            n_seats = 1
        if per_candidate_limit is None:
            per_candidate_limit = 1
        super().__init__(
            profile,
            n_seats=n_seats,
            per_candidate_limit=per_candidate_limit,
            tiebreak=tiebreak,
            rng_seed=rng_seed,
        )
