from dataclasses import dataclass
from typing import Self, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from votekit._df_schema import (
    VOTER_SET_COLUMN,
    WEIGHT_COLUMN,
)
from votekit.pref_profile.profile_matrix.shared_data import (
    _arrays_equal,
    _attach_ballot_columns,
    _ProfileVoterWeightData,
)
from votekit.types import Candidate


@dataclass(frozen=True, slots=True, eq=False)
class ScoreMatrix:
    """
    Store scored ballots as a floating-point array with candidate-labelled columns.

    Construction copies scores into a read-only array without changing their dtype and retains
    column and candidate metadata as tuples. The validated shared-data block is reused.
    Candidate names are not normalized here.

    Args:
        voter_weight_data (_ProfileVoterWeightData): Shared data with one entry per ballot.
        scores (NDArray): Two-dimensional floating-point array with one row per ballot and
            one column per score column. NaN means unscored. Empty dimensions are supported;
            score ranges and finiteness are not constrained by this constructor.
        score_columns (tuple[Candidate, ...]): Unique candidate name for each score column,
            in array order. Weight and Voter Set are reserved and cannot be score columns.
        candidates (tuple[Candidate, ...]): Profile candidate universe, which may include
            candidates without score columns. An undeclared column may contain nonzero,
            non-NaN scores only on zero-weight ballots.
    """

    voter_weight_data: _ProfileVoterWeightData
    scores: NDArray
    score_columns: tuple[Candidate, ...]
    candidates: tuple[Candidate, ...]

    def __post_init__(self) -> None:
        """
        Validate score storage and column mappings, then retain independently owned data.

        Raises:
            TypeError: Scores are not an array or column names have unsupported types.
            ValueError: The score dtype, dimensions, or column mappings are invalid.
        """
        self._validate_score_array()
        self._validate_score_columns()

        owned_scores = np.array(self.scores, copy=True)
        owned_scores.setflags(write=False)

        object.__setattr__(self, "scores", owned_scores)
        object.__setattr__(self, "score_columns", tuple(self.score_columns))
        object.__setattr__(self, "candidates", tuple(self.candidates))

    def _validate_score_array(self) -> None:
        """
        Check floating-point scores and alignment with weights and score columns.

        Raises:
            TypeError: Scores are not a NumPy array.
            ValueError: Scores are not a two-dimensional floating-point array or row/column counts
                disagree with the shared data or score labels.
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
            ValueError: Score conversion fails or column mappings or shared-data structure are
                invalid.
            ProfileError: Shared weight validation rejects a value.
        """
        score_columns = tuple(
            col for col in df.columns if col not in (WEIGHT_COLUMN, VOTER_SET_COLUMN)
        )

        return cls(
            voter_weight_data=_ProfileVoterWeightData.from_df(df),
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

    def to_df(self) -> pd.DataFrame:
        """
        Export scores to a new dataframe in stored candidate-column order.

        Returns:
            pd.DataFrame: Writable score columns followed by Voter Set and Weight. Score values
                have independent storage. The index is named Ballot Index and uses retained labels
                when present; its label storage is independent of the matrix.
        """
        df = pd.DataFrame(self.scores, columns=list(self.score_columns), copy=True)
        return _attach_ballot_columns(df, self.voter_weight_data)
