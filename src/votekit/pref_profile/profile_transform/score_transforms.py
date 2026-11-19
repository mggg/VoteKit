from dataclasses import dataclass, field

import numpy as np

from votekit.pref_profile.profile_matrix import ScoreMatrix
from votekit.pref_profile.profile_transform.transform_utils import (
    _expand_to_unit_weights,
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

    Use with ``perturb``, whose ``per_voter_probability`` selects the ballots to jitter. Each
    selected voter receives its own move, so rows are split into one row per unit of weight.

    The generator is created once from ``rng_seed``, so repeated calls, such as repeated ``perturb``
    runs with the same transform, continue its random stream instead of repeating the same moves.

    Attributes:
        rng_seed (int, optional): Seed for sampling the candidates and directions. Pass a seed for
            reproducible moves. Defaults to None, which creates an unseeded generator.
        step (float, optional): Size of each move in score. Defaults to 1.0.
        max_score (float, optional): Highest score a move can reach. Every score on the ballots
            the transform receives must already be at most ``max_score``. Defaults to None, which
            sets no upper bound.

    Raises:
        ValueError: ``step`` is not positive, or ``max_score`` is negative.
        ValueError: A ballot already gives a score above ``max_score``.
    """

    rng_seed: int | None = None
    step: float = 1.0
    max_score: float | None = None
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.step > 0:
            raise ValueError(f"step must be positive, received {self.step}.")
        if self.max_score is not None and self.max_score < 0:
            raise ValueError(f"max_score cannot be negative, received {self.max_score}.")
        self._rng = np.random.default_rng(self.rng_seed)

    def transform_matrix(self, matrix: ScoreMatrix) -> ScoreMatrix:
        if len(matrix) == 0:
            return matrix

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

        source_row_idxs, unit_matrix_data = _expand_to_unit_weights(matrix)
        unit_scores = matrix.scores[source_row_idxs]

        row_idxs = np.arange(len(unit_scores))
        jitter_columns = declared_columns[
            self._rng.integers(0, len(declared_columns), size=len(unit_scores))
        ]
        score_moves = self._rng.choice([-self.step, self.step], size=len(unit_scores))

        jittered_scores = np.maximum(unit_scores[row_idxs, jitter_columns] + score_moves, 0.0)
        if self.max_score is not None:
            jittered_scores = np.minimum(jittered_scores, self.max_score)
        unit_scores[row_idxs, jitter_columns] = jittered_scores

        return ScoreMatrix(
            voter_weight_data=unit_matrix_data,
            scores=unit_scores,
            score_columns=matrix.score_columns,
            candidates=matrix.candidates,
        )
