from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Self

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from votekit._df_schema import (
    VOTER_SET_COLUMN,
    WEIGHT_COLUMN,
)
from votekit.pref_profile.profile_matrix.shared_data import (
    ProfileVoterWeightData,
    _arrays_equal,
    _attach_index_voterset_and_weight_columns,
    _concatenate_voter_weight_data,
)
from votekit.types import Candidate


@dataclass(frozen=True, slots=True, eq=False)
class ScoreMatrix:
    """
    Store scored ballots as a floating-point array with candidate-labelled columns.

    Construction copies scores into a read-only array without changing their dtype and retains
    column and candidate metadata as tuples. The validated shared-data block is reused.
    Signed zeros are stored as positive zero. Candidate names are not normalized here.

    Args:
        voter_weight_data (_ProfileVoterWeightData): Shared data with one entry per ballot.
        scores (NDArray): Two-dimensional floating-point array with one row per ballot and
            one column per score column. NaN means unscored. Empty dimensions are supported;
            negative scores are rejected. Positive infinity is retained.
        score_columns (tuple[Candidate, ...]): Unique candidate name for each score column,
            in array order. Weight and Voter Set are reserved and cannot be score columns.
        candidates (tuple[Candidate, ...]): Profile candidate universe, which may include
            candidates without score columns. An undeclared column may contain nonzero,
            non-NaN scores only on zero-weight ballots.
    """

    voter_weight_data: ProfileVoterWeightData
    scores: NDArray
    score_columns: tuple[Candidate, ...]
    candidates: tuple[Candidate, ...]
    _declared_score_columns: tuple[tuple[Candidate, int], ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """
        Validate score storage and column mappings, then retain independently owned data.

        Raises:
            TypeError: Scores are not an array or column names have unsupported types.
            ValueError: Scores are negative, or the dtype, dimensions, or column mappings
                are invalid.
        """
        self._validate_score_array()
        self._validate_score_columns()

        owned_scores = np.array(self.scores, copy=True)
        owned_scores[owned_scores == 0] = 0.0
        owned_scores.setflags(write=False)

        object.__setattr__(self, "scores", owned_scores)
        object.__setattr__(self, "score_columns", tuple(self.score_columns))
        object.__setattr__(self, "candidates", tuple(self.candidates))

        column_positions = {
            candidate: column for column, candidate in enumerate(self.score_columns)
        }
        declared_columns = tuple(
            (candidate, column_positions[candidate])
            for candidate in self.candidates
            if candidate in column_positions
        )
        object.__setattr__(self, "_declared_score_columns", declared_columns)

    def _validate_score_array(self) -> None:
        """
        Check floating-point scores and alignment with weights and score columns.

        Raises:
            TypeError: Scores are not a NumPy array.
            ValueError: Scores are not a two-dimensional floating-point array or row/column counts
                disagree with the shared data or score labels, or a score is negative.
        """
        if not isinstance(self.scores, np.ndarray):
            raise TypeError(f"scores must be a numpy array, not {type(self.scores)}.")
        if self.scores.ndim != 2:
            raise ValueError(
                f"scores must be a 2D array, not {self.scores.ndim}D: {self.scores.shape}"
            )
        if not np.issubdtype(self.scores.dtype, np.floating):
            raise ValueError(f"scores must have a floating-point dtype, not {self.scores.dtype}.")

        if self.scores.shape[0] != len(self.voter_weight_data):
            raise ValueError(
                f"scores must have the same number of rows as weights: "
                f"{self.scores.shape[0]} != {len(self.voter_weight_data)}"
            )
        if self.scores.shape[1] != len(self.score_columns):
            raise ValueError(
                f"scores must have the same number of columns as score_columns: "
                f"{self.scores.shape[1]} != {len(self.score_columns)}"
            )

        if np.any(self.scores < 0):
            raise ValueError("Scores must be non-negative.")

    def _validate_score_columns(self) -> None:
        """
        Check score column names and candidate membership on positive-weight ballots.

        Raises:
            TypeError: A column name is not a string or non-boolean integer.
            ValueError: Column names are repeated or reserved, or an undeclared candidate has a
                nonzero, non-NaN score on a positive-weight ballot.
        """
        if any(
            not isinstance(candidate, (str, int)) or isinstance(candidate, bool)
            for candidate in self.score_columns
        ):
            raise TypeError("Score columns must contain string or integer candidate names.")
        if len(set(self.score_columns)) != len(self.score_columns):
            raise ValueError("Score columns must contain unique candidate names.")
        if set(self.score_columns).intersection((WEIGHT_COLUMN, VOTER_SET_COLUMN)):
            raise ValueError("Score columns cannot use reserved weight or voter-set column names.")

        candidates = set(self.candidates)
        undeclared_columns = [
            index
            for index, candidate in enumerate(self.score_columns)
            if candidate not in candidates
        ]
        if undeclared_columns:
            # Profiles may retain candidates outside their universe on zero-weight ballots.
            active_scores = self.scores[self.voter_weight_data.weights > 0][:, undeclared_columns]
            if np.any(~np.isnan(active_scores) & (active_scores != 0)):
                raise ValueError(
                    "Scores on positive-weight ballots contain candidates outside "
                    "the declared candidates."
                )

    @classmethod
    def from_df(cls, df: pd.DataFrame, candidates: Sequence[Candidate]) -> Self:
        """
        Extract candidate-labelled score columns into owned matrix storage.

        Args:
            df (pd.DataFrame): Profile dataframe with Weight and Voter Set columns. All remaining
                columns are candidate scores convertible to float64; NaN means unscored.
            candidates (Sequence[Candidate]): Profile candidate universe in declared order.
                Candidate names and dataframe labels must already be normalized.

        Returns:
            Self: Matrix preserving input score-column order with a float64 score array.

        Raises:
            KeyError: A required shared column is missing.
            TypeError: Score values cannot be converted to floats or column names are unsupported.
            ValueError: Scores are negative, conversion fails, or column mappings or shared-data
                structure are invalid.
            ProfileError: Shared weight validation rejects a value.
        """
        score_columns = tuple(
            col for col in df.columns if col not in (WEIGHT_COLUMN, VOTER_SET_COLUMN)
        )

        return cls(
            voter_weight_data=ProfileVoterWeightData.from_df(df),
            scores=df[list(score_columns)].to_numpy(dtype=float),
            score_columns=score_columns,
            candidates=tuple(candidates),
        )

    def __len__(self) -> int:
        """
        Return the number of ballots, including zero-weight ballots.
        """
        return len(self.voter_weight_data)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ScoreMatrix):
            return NotImplemented
        return (
            self.voter_weight_data == other.voter_weight_data
            and _arrays_equal(self.scores, other.scores)
            and self.score_columns == other.score_columns
            and self.candidates == other.candidates
        )

    def to_df(self, *, rows: NDArray[np.intp] | None = None) -> pd.DataFrame:
        """
        Export scores to a new dataframe in stored candidate-column order.

        Args:
            rows (NDArray[np.intp] | None, optional): Positional row indices in output order.
                Repeated and negative indices are supported. Defaults to None for all rows.
                Source ballot labels are preserved, including implicit labels.

        Returns:
            pd.DataFrame: Writable score columns followed by Voter Set and Weight. Score values
                have independent storage. The index is named Ballot Index and uses retained labels
                when present; its label storage is independent of the matrix.
        """
        scores = self.scores if rows is None else self.scores[rows]
        df = pd.DataFrame(scores, columns=list(self.score_columns), copy=True)
        return _attach_index_voterset_and_weight_columns(df, self.voter_weight_data, rows=rows)

    def copy(self) -> ScoreMatrix:
        """
        Return a deep copy of the matrix with independent arrays and tuples.

        Returns:
            ScoreMatrix: New matrix with the same data.
        """
        return ScoreMatrix(
            voter_weight_data=self.voter_weight_data.copy(),
            scores=self.scores,
            score_columns=self.score_columns,
            candidates=self.candidates,
        )

    def get_scores(self, row: int) -> dict[Candidate, float]:
        """
        Decode one ballot's scores in declared candidate order, retaining explicit zeros.

        Args:
            row (int): Positional row index, with negative indices counted from the end.

        Returns:
            dict[Candidate, float]: Fresh dictionary excluding NaNs and undeclared candidates.
                An empty dictionary means the row has no scores for declared candidates.

        Raises:
            IndexError: The row is outside the matrix.
        """
        scores = self.scores[row]
        return {
            candidate: float(scores[column])
            for candidate, column in self._declared_score_columns
            if not np.isnan(scores[column])
        }

    def score_bounds_by_ballot(self) -> tuple[NDArray, NDArray]:
        """
        Find each ballot's minimum and maximum scores among declared candidates.

        Returns:
            tuple[NDArray, NDArray]: Independent minimum and maximum arrays in source row order,
                preserving the score dtype. Both bounds are NaN for an entirely unscored row.
                Explicit zero scores and scores on zero-weight rows are included.
        """
        columns = [column for _, column in self._declared_score_columns]
        scores = self.scores[:, columns]

        return (
            np.fmin.reduce(scores, axis=1, initial=np.nan),
            np.fmax.reduce(scores, axis=1, initial=np.nan),
        )

    def sum_scores_by_ballot(self) -> NDArray[np.float64]:
        """
        Sum each ballot's declared scores without applying ballot weights.

        Returns:
            NDArray[np.float64]: Independent totals in source row order, computed with NumPy's
                float64 reduction over declared candidate columns. Missing scores contribute zero;
                entirely unscored rows total zero.
        """
        columns = [column for _, column in self._declared_score_columns]
        return np.array(
            np.nansum(self.scores[:, columns], axis=1, dtype=np.float64), dtype=np.float64
        )

    def get_candidates_cast(self) -> frozenset[Candidate]:
        """
        Return candidates with a positive score on at least one positive-weight ballot.
        """
        active_scores = self.scores[self.voter_weight_data.weights > 0]
        active_columns = np.flatnonzero(np.any(active_scores > 0, axis=0))
        return frozenset(self.score_columns[column] for column in active_columns)

    def _group_score_rows(self) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
        """
        Identify equal scores in first-appearance order, treating all NaNs as unscored.

        Zero and unscored entries remain distinct. Constructor normalization already makes
        signed zeros equal. Comparison keys do not alter the stored scores.

        Returns:
            tuple[NDArray[np.intp], NDArray[np.intp]]: Representative source rows and one
                group ID per source row. Both arrays are empty for an empty matrix.
        """
        if not len(self):
            return np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp)

        if not self.scores.shape[1]:
            return np.array([0], dtype=np.intp), np.zeros(len(self), dtype=np.intp)

        if self.scores.dtype.itemsize > np.dtype(np.float64).itemsize:
            # Extended floats may have padding bytes; compare values before making byte keys.
            rows = np.empty(self.scores.shape, dtype=np.intp)
            for column in range(self.scores.shape[1]):
                _, rows[:, column] = np.unique(self.scores[:, column], return_inverse=True)
        else:
            rows = np.array(self.scores, copy=True, order="C")
            rows[np.isnan(rows)] = np.nan

        row_dtype = np.dtype((np.void, rows.dtype.itemsize * len(self.score_columns)))
        row_keys = rows.view(row_dtype).reshape(-1)
        _, first_rows_idx, group_ids = np.unique(row_keys, return_index=True, return_inverse=True)

        group_order = np.argsort(first_rows_idx)
        reordered_ids = np.empty(len(group_order), dtype=np.intp)
        reordered_ids[group_order] = np.arange(len(group_order))
        return first_rows_idx[group_order], reordered_ids[group_ids]

    def group_ballots(self) -> ScoreMatrix:
        """
        Combine equal score rows in first-appearance order without dataframe conversion.

        Returns:
            ScoreMatrix: Grouped matrix with owned, read-only arrays. An empty result uses
                float64 weights. Immutable voter sets may be shared.

        Raises:
            ProfileError: A grouped floating-point weight overflows to a nonfinite value.
        """
        first_rows_idx, group_ids = self._group_score_rows()
        grouped_data = self.voter_weight_data.group_rows(first_rows_idx, group_ids)
        return ScoreMatrix(
            voter_weight_data=grouped_data,
            scores=self.scores[first_rows_idx],
            score_columns=self.score_columns,
            candidates=self.candidates,
        )


def _concatenate_score_matrices(
    matrices: Sequence[ScoreMatrix], candidates: tuple[Candidate, ...]
) -> ScoreMatrix:
    """
    Concatenate score rows, aligning columns to the combined candidate universe.

    Args:
        matrices (Sequence[ScoreMatrix]): Nonempty sequence of matrices in source row order.
        candidates (tuple[Candidate, ...]): Output column order and candidate universe.

    Returns:
        ScoreMatrix: Owned combined arrays with reset labels. Columns outside a source's
            declared candidate universe are excluded from that source's rows.
    """
    data = _concatenate_voter_weight_data([matrix.voter_weight_data for matrix in matrices])
    scores = np.full((len(data), len(candidates)), np.nan)
    candidate_columns = {candidate: column for column, candidate in enumerate(candidates)}

    offset = 0
    for matrix in matrices:
        end = offset + len(matrix)
        for column, candidate in enumerate(matrix.score_columns):
            if candidate in matrix.candidates:
                scores[offset:end, candidate_columns[candidate]] = matrix.scores[:, column]
        offset = end

    return ScoreMatrix(
        voter_weight_data=data,
        scores=scores,
        score_columns=candidates,
        candidates=candidates,
    )
