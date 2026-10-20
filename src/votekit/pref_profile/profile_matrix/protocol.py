from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, Self, runtime_checkable

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from votekit.pref_profile.profile_matrix.shared_data import ProfileVoterWeightData
from votekit.types import Candidate


@runtime_checkable
class ProfileMatrix(Protocol):
    """
    Define the shared construction, grouping, and export interface for profile matrices.

    RankMatrix and ScoreMatrix implement this protocol structurally. Common profile operations
    use this interface; rank- and score-specific operations use their concrete matrix types.
    Dataframe parsing belongs to the Python adapters exposed by ``from_df``. Construction
    owns input storage and preserves ballot row order. Exports are independent snapshots;
    round trips preserve ballot meaning, not necessarily input dtypes or column order.

    Attributes:
        voter_weight_data (_ProfileVoterWeightData): Aligned weights, voters, and index labels.
        candidates (tuple[Candidate, ...]): Candidate universe associated with the profile.
    """

    @property
    def voter_weight_data(self) -> ProfileVoterWeightData:
        """
        Aligned weights, voters, and index labels, exposed without a setter.
        """
        ...

    @property
    def candidates(self) -> tuple[Candidate, ...]:
        """
        Candidate universe in its declared order, exposed without a setter.
        """
        ...

    @classmethod
    def from_df(cls, df: pd.DataFrame, candidates: Sequence[Candidate]) -> Self:
        """
        Construct a matrix from a dataframe using public candidate names or candidate sets.

        Args:
            df (pd.DataFrame): Profile dataframe with ballot rows and shared Weight and Voter Set
                columns. Payload columns follow the concrete matrix's schema.
            candidates (Sequence[Candidate]): Candidate universe in its declared order.

        Returns:
            Self: Matrix containing owned, validated ballot data in input row order. Later
                changes to the dataframe do not affect the matrix.
        """
        ...

    def __len__(self) -> int:
        """
        Return the number of ballots, including zero-weight ballots.
        """
        ...

    def to_df(self, *, rows: NDArray[np.intp] | None = None) -> pd.DataFrame:
        """
        Export stored ballot data using the concrete matrix's public dataframe schema.

        Args:
            rows (NDArray[np.intp] | None, optional): Positional rows in output order, allowing
                repeats and negative indices. Defaults to None for all rows. Preserve source
                labels, including implicit row labels, when selecting rows.

        Returns:
            pd.DataFrame: Ballot rows in stored order, with decoded candidate data, shared columns,
                and source index labels when retained. Mutating an export affects neither the
                matrix nor other exports. Immutable sets may be shared. Weight dtypes, column
                ordering, and index metadata may differ from the original input.
        """
        ...

    def group_ballots(self) -> ProfileMatrix:
        """
        Combine equal ballot payloads in first-appearance order.

        Returns:
            ProfileMatrix: Grouped matrix of the same ballot kind, with owned, read-only arrays.
                Empty input produces float64 weights.

        Raises:
            ProfileError: A grouped floating-point weight overflows to a nonfinite value.
        """
        ...
