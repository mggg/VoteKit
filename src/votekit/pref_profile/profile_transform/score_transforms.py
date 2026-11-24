import math
from dataclasses import dataclass, replace

import numpy as np

from votekit.pref_profile.profile_matrix import ScoreMatrix
from votekit.pref_profile.profile_transform.transform_utils import (
    _split_per_voter,
    _validate_one_voter_per_unit_weight,
)


@dataclass
class JitterScores:
    """
    ProfileTransform class to move one random candidate's score up or down on each ballot.

    For each ballot, a candidate is drawn uniformly from the profile's candidates and a direction,
    up or down, is drawn uniformly. The candidate's score moves by ``step`` in that direction. An
    unscored candidate (NaN) stays unscored, so that ballot is unchanged. Scores cannot be negative,
    so a move below 0 stops at 0, and if ``max_score`` is given, a move above it stops at it.
    Undeclared score columns are never drawn.

    Use with ``transform``, whose ``per_voter_probability`` selects the ballots to jitter. Each
    selected voter receives its own move, so rows are split into one row per unit of weight. The
    moves are drawn from the generator ``transform`` passes in, so its ``rng`` reproduces them.

    Attributes:
        step (float, optional): Size of each move in score. Defaults to 1.0.
        max_score (float, optional): Highest score a move can reach. Every score on the ballots the
            transform receives must already be at most ``max_score``. Defaults to None.

    Raises:
        ValueError: ``step`` is not a positive finite number, or ``max_score`` is not a non-negative
            finite number.
        ValueError: A ballot already gives a score above ``max_score``.
        ValueError: The matrix has non-integral weights, so its voters cannot be counted.
        ProfileError: A row's non-empty voter set does not have one voter per unit of weight.
    """

    step: float = 1.0
    max_score: float | None = None

    def __post_init__(self) -> None:
        if not (math.isfinite(self.step) and self.step > 0):
            raise ValueError(f"step must be a positive finite number, received {self.step}.")
        if self.max_score is not None and not (
            math.isfinite(self.max_score) and self.max_score >= 0
        ):
            raise ValueError(
                f"max_score must be a non-negative finite number, received {self.max_score}."
            )

    def transform_matrix(self, matrix: ScoreMatrix, rng: np.random.Generator) -> ScoreMatrix:
        if len(matrix) == 0:
            return matrix
        _validate_one_voter_per_unit_weight(matrix)

        candidates = frozenset(matrix.candidates)
        declared_columns = np.array(
            [column for column, cand in enumerate(matrix.score_columns) if cand in candidates],
            dtype=np.intp,
        )
        if len(declared_columns) == 0:
            return matrix

        if self.max_score is not None:
            above_max = matrix.scores[:, declared_columns] > self.max_score
            if np.any(above_max):
                max_matrix_score = np.nanmax(matrix.scores[:, declared_columns])
                raise ValueError(
                    f"max_score {self.max_score} is below max score given in the matrix,"
                    f" {max_matrix_score}. Set max_score to at least the profile's highest score."
                )

        voter_matrix = _split_per_voter(matrix)
        unit_scores = voter_matrix.scores.copy()

        row_idxs = np.arange(len(unit_scores))
        jitter_columns = declared_columns[
            rng.integers(0, len(declared_columns), size=len(unit_scores))
        ]
        score_moves = rng.choice([-self.step, self.step], size=len(unit_scores))

        jittered_scores = np.maximum(unit_scores[row_idxs, jitter_columns] + score_moves, 0.0)
        if self.max_score is not None:
            jittered_scores = np.minimum(jittered_scores, self.max_score)
        unit_scores[row_idxs, jitter_columns] = jittered_scores

        return replace(voter_matrix, scores=unit_scores)
