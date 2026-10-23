from __future__ import annotations

from copy import copy as shallow_copy
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from math import fsum
from typing import Any, Self, Sequence, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from votekit._df_schema import (
    BALLOT_INDEX_NAME,
    VOTER_SET_COLUMN,
    WEIGHT_COLUMN,
)
from votekit.ballot import _to_float_weight, validate_weight
from votekit.exceptions import ProfileError

MAX_TOTAL_BALLOT_WEIGHT = 2**50
EMPTY_VOTER_SET: frozenset[str] = frozenset()


def _arrays_equal(arr_1: NDArray | None, arr_2: NDArray | None) -> bool:
    """
    Compare optional arrays, treating NaNs at matching floating-point positions as equal.

    Two absent arrays compare equal; an absent array and a present array do not. Object arrays
    use NumPy's elementwise equality without special handling for NaN values.

    Args:
        arr_1 (NDArray | None): First array to compare.
        arr_2 (NDArray | None): Second array to compare.

    Returns:
        bool: Whether shapes and values match under these rules.
    """
    if arr_1 is None or arr_2 is None:
        return arr_1 is arr_2
    if arr_1.dtype.kind == "f" and arr_2.dtype.kind == "f":
        return np.array_equal(arr_1, arr_2, equal_nan=True)
    return np.array_equal(arr_1, arr_2)


class WeightKind(Enum):
    """
    Identify the numeric representation of a profile's ballot weights.

    FLOAT requires a float64 array. FRACTION requires an object array containing only
    ``Fraction`` values. The shared-data constructor validates the selected representation.
    """

    FLOAT = "float"
    FRACTION = "fraction"


@dataclass(frozen=True, slots=True, eq=False)
class ProfileVoterWeightData:
    """
    Store aligned ballot weights, voter sets, and optional source index labels.

    Total ballot weight must not exceed 2**50.
    Construction copies each array and marks it read-only. Voter entries become frozensets;
    index labels in object arrays are retained without recursively copying their contents.

    Args:
        weights (``NDArray[np.float64] | NDArray[np.object_]``): One-dimensional array of finite,
            nonnegative weights. FLOAT requires float64; FRACTION requires only Fraction values.
        voter_sets (``NDArray[np.object_] | None``): One voter collection per ballot, or None when
            voter data is absent. Collections are converted to frozensets during construction.
        weight_kind (WeightKind): Numeric representation required for ``weights``.
        ballot_index (NDArray[Any] | None, optional): One source index label per ballot. None
            recreates a default RangeIndex on export. Defaults to None.
    """

    weights: NDArray[np.float64] | NDArray[np.object_]
    voter_sets: NDArray[np.object_] | None
    weight_kind: WeightKind
    ballot_index: NDArray[Any] | None = None

    def __post_init__(self) -> None:
        """
        Validate aligned columns and retain owned, read-only array storage.

        Raises:
            ValueError: Array dimensions or lengths disagree, the weight kind or dtype is invalid,
                or a Fraction weight is negative or has the wrong type.
            ProfileError: A floating-point weight is negative, NaN, or infinite, or total ballot
                weight exceeds 2**50.
            TypeError: A voter entry cannot be converted to a frozenset.
        """
        self._validate_weight_voterset_index_alignment()

        self._validate_weight_values_and_types()
        _validate_total_ballot_weight(self.weights)
        owned_weights = np.copy(self.weights)
        owned_weights.setflags(write=False)

        owned_voter_sets = None
        if self.voter_sets is not None:
            if all(isinstance(voter_set, frozenset) for voter_set in self.voter_sets):
                owned_voter_sets = np.array(self.voter_sets, dtype=object, copy=True)
            else:
                owned_voter_sets = np.empty(len(self.voter_sets), dtype=object)
                for index, voter_set in enumerate(self.voter_sets):
                    owned_voter_sets[index] = frozenset(voter_set)
            owned_voter_sets.setflags(write=False)

        owned_ballot_index = None
        if self.ballot_index is not None:
            owned_ballot_index = np.copy(self.ballot_index)
            owned_ballot_index.setflags(write=False)

        object.__setattr__(self, "weights", owned_weights)
        object.__setattr__(self, "voter_sets", owned_voter_sets)
        object.__setattr__(self, "ballot_index", owned_ballot_index)

    def _validate_weight_voterset_index_alignment(self) -> None:
        """
        Check that all supplied per-ballot arrays are one-dimensional and equally long.

        Raises:
            ValueError: An array is not one-dimensional or an optional column differs in length
                from the weights.
        """
        if self.weights.ndim != 1:
            raise ValueError("weights must be a 1D array.")

        for name, column in (
            (VOTER_SET_COLUMN, self.voter_sets),
            (BALLOT_INDEX_NAME, self.ballot_index),
        ):
            if column is None:
                continue
            if column.ndim != 1:
                raise ValueError(f"{name} must be a 1D array of the same length as weights.")
            if len(column) != len(self.weights):
                raise ValueError(
                    f"{name} must be the same length as weights: "
                    f"{len(column)} ({column}) != {len(self.weights)} (weights)."
                )

    def _validate_weight_values_and_types(self) -> None:
        """
        Check weight representation and nonnegative, finite values without converting them.

        Raises:
            ValueError: The weight kind is unknown, a dtype disagrees with the weight kind, or a
                FRACTION array contains a non-Fraction or negative value.
            ProfileError: A FLOAT array contains a negative, NaN, or infinite value.
        """

        if self.weight_kind is WeightKind.FLOAT:
            if self.weights.dtype != np.float64:
                raise ValueError(
                    f"weights must be float64 when weight_kind is FLOAT. Found:{self.weights.dtype}"
                )
            _validate_numeric_weights(self.weights)

        elif self.weight_kind is WeightKind.FRACTION:
            if self.weights.dtype != np.object_:
                raise ValueError(
                    f"weights must be object when weight_kind is FRACTION. "
                    f"Found: {self.weights.dtype}"
                )

            if not all(isinstance(weight, Fraction) for weight in self.weights):
                bad_weights = [
                    weight for weight in self.weights if not isinstance(weight, Fraction)
                ]
                bad_weight_err_msg = ", ".join(
                    f"{weight} ({type(weight).__name__})" for weight in bad_weights[:3]
                )

                if len(bad_weights) > 3:
                    bad_weight_err_msg += f", and {len(bad_weights) - 3} more..."
                raise ValueError(
                    "weights must be an object array of Fractions when weight_kind is FRACTION."
                    f" Found: {bad_weight_err_msg}"
                )

            if any(weight < 0 for weight in self.weights):
                neg_weights = [weight for weight in self.weights if weight < 0]
                neg_weight_err_msg = ", ".join(
                    f"{weight} ({type(weight).__name__})" for weight in neg_weights[:3]
                )

                if len(neg_weights) > 3:
                    neg_weight_err_msg += f", and {len(neg_weights) - 3} more..."

                raise ValueError(f"Ballot weights cannot be negative. Found: {neg_weight_err_msg}")

        else:
            raise ValueError(f"weight_kind must be FLOAT or FRACTION. Found: {self.weight_kind}")

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> Self:
        """
        Build owned ballot data from a profile dataframe's shared columns.

        Weights follow ``_standardize_weights``. A default zero-based, unit-step RangeIndex is
        represented by None; other index labels are retained for export.

        Args:
            df (pd.DataFrame): Dataframe with Weight and Voter Set columns. Voter entries must be
                collections that can be converted to frozensets.

        Returns:
            Self: Validated ballot data with independent array storage.

        Raises:
            KeyError: A required shared column is missing.
            ProfileError: Weight validation rejects a value before or after conversion.
            ValueError: Converted data violates the shared-data constructor's shape or type rules.
            TypeError: Voter entries cannot be converted to frozensets.
        """
        weights, weight_kind = _standardize_weights(df[WEIGHT_COLUMN].to_numpy())

        voter_set_column = df[VOTER_SET_COLUMN]
        voter_sets = (
            voter_set_column.to_numpy() if (voter_set_column != EMPTY_VOTER_SET).any() else None
        )

        return cls(
            weights=weights,
            voter_sets=voter_sets,
            weight_kind=weight_kind,
            ballot_index=_non_default_index(df.index),
        )

    @property
    def has_fraction_weights(self) -> bool:
        """
        Return whether the stored weights use the Fraction representation.
        """
        return self.weight_kind is WeightKind.FRACTION

    def __len__(self) -> int:
        """
        Return the number of ballots in the shared columns.
        """
        return len(self.weights)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ProfileVoterWeightData):
            return NotImplemented
        return (
            _arrays_equal(self.weights, other.weights)
            and _arrays_equal(self.voter_sets, other.voter_sets)
            and _arrays_equal(self.ballot_index, other.ballot_index)
        )

    def voter_set_column(self) -> Any:
        """
        Return voter entries for dataframe export, filling absent data with empty frozensets.

        Returns:
            ``NDArray[np.object_] | list[frozenset[str]]``: The stored read-only array when present,
                otherwise a list containing one empty frozenset per ballot.
        """
        if self.voter_sets is not None:
            return self.voter_sets
        return cast(Any, [EMPTY_VOTER_SET] * len(self))

    def group_rows(
        self, first_rows_idx: NDArray[np.intp], group_ids: NDArray[np.intp]
    ) -> ProfileVoterWeightData:
        """
        Sum weights and union voters for internally computed row groups, resetting labels.

        Args:
            first_rows_idx (NDArray[np.intp]): First source row for each group, in output order.
            group_ids (NDArray[np.intp]): One group ID per source row, indexing first_rows_idx.
                The matrix grouping operation must supply consistent, complete assignments.

        Returns:
            _ProfileVoterWeightData: Owned, read-only grouped columns without source labels.
                Fraction sums remain exact. Empty input produces float64 weights, and absent
                or entirely empty voter sets produce None.

        Raises:
            ProfileError: A grouped floating-point weight overflows to a nonfinite value.
        """
        additional_rows = np.ones(len(self), dtype=bool)
        additional_rows[first_rows_idx] = False
        additional_row_idxs = np.flatnonzero(additional_rows)  # where the mask is True

        if not len(self):
            weights = np.empty(0, dtype=np.float64)

        elif self.weight_kind is WeightKind.FRACTION:
            weights = self.weights[first_rows_idx]
            for row in additional_row_idxs:
                weights[group_ids[row]] += self.weights[row]  # need to add fractions in Python

        else:
            weights = cast(
                NDArray[np.float64],
                np.bincount(group_ids, weights=self.weights, minlength=len(first_rows_idx)),
            )

        voter_sets = None
        if self.voter_sets is not None:
            voter_sets = self.voter_sets[first_rows_idx]
            voter_unions: dict[int, set[str]] = {}

            for row in additional_row_idxs:
                group = int(group_ids[row])
                if group not in voter_unions:
                    voter_unions[group] = set(voter_sets[group])
                voter_unions[group].update(self.voter_sets[row])

            for group, voters in voter_unions.items():
                voter_sets[group] = frozenset(voters)

            if not any(voter_sets):
                voter_sets = None

        return ProfileVoterWeightData(
            weights=weights,
            voter_sets=voter_sets,
            weight_kind=self.weight_kind if len(self) else WeightKind.FLOAT,
        )

    def with_weights(self, new_weights: NDArray) -> Self:
        """
        Replace weights with owned storage while sharing voters and ballot labels.

        The supplied weights follow the same normalization rules as dataframe construction.
        Ballot count and row order remain unchanged. The original object is not modified.

        Args:
            weights (NDArray): One replacement weight per existing ballot, in source row order.

        Returns:
            Self: New data object with independently owned, read-only weights. Existing voter
                and index arrays are shared.

        Raises:
            ProfileError: Weights have invalid dimensions, types, or values, or their count
                differs from the existing ballot count, or total weight exceeds 2**50.
        """

        new_weights, new_weight_kind = _standardize_weights(new_weights)

        if len(new_weights) != len(self):
            raise ProfileError(
                f"Replacement weights must have the same length as existing weights: "
                f"{len(new_weights)} != {len(self)}."
            )

        _validate_total_ballot_weight(new_weights)
        owned_weights = np.array(new_weights, copy=True)
        owned_weights.setflags(write=False)

        data = shallow_copy(self)
        object.__setattr__(data, "weights", owned_weights)
        object.__setattr__(data, "weight_kind", new_weight_kind)

        return data

    def copy(self, *, shallow=False) -> ProfileVoterWeightData:
        """
        Return a shallow copy of the shared-data object with independent array storage.

        The returned object shares no storage with the original, but its arrays are read-only.

        """
        if shallow:
            return shallow_copy(self)

        else:
            return ProfileVoterWeightData(
                weights=self.weights,
                voter_sets=self.voter_sets,
                weight_kind=self.weight_kind,
                ballot_index=self.ballot_index,
            )

    def apply_mask(self, mask: NDArray[np.bool_]) -> ProfileVoterWeightData:
        """
        Apply boolean mask to select rows while retaining their source index labels.

        A selection of validated rows remains valid, so construction is bypassed. Masked arrays are
        new storage and are marked read-only.

        Args:
            mask (NDArray[np.bool_]): Boolean mask with one entry per ballot.

        Returns:
            ProfileVoterWeightData: Selected rows with explicit ballot labels.  A default index
                (stored as None) is materialized as source positions, so the kept rows retain their
                original labels instead of being renumbered from zero.

        Raises:
            ValueError: The mask is not one flag per ballot.
        """
        if mask.shape != (len(self),):
            raise ValueError(f"mask must have shape ({len(self)},), not {mask.shape}.")

        labels = self.ballot_index if self.ballot_index is not None else np.arange(len(self))
        weights = self.weights[mask]
        voter_sets = None if self.voter_sets is None else self.voter_sets[mask]
        ballot_index = labels[mask]

        for column in (weights, voter_sets, ballot_index):
            if column is not None:
                column.setflags(write=False)

        data = shallow_copy(self)
        object.__setattr__(data, "weights", weights)
        object.__setattr__(data, "voter_sets", voter_sets)
        object.__setattr__(data, "ballot_index", ballot_index)

        return data


def _validate_total_ballot_weight(weights: NDArray[np.float64] | NDArray[np.object_]) -> None:
    """
    Reject validated, nonnegative weights whose combined total exceeds 2**50.

    The limit bounds the supported profile size when replacing pandas' compensated
    mentions reduction with direct matrix operations. Near 2**53, naive float summation
    can lose individual votes alongside large weights; 2**50 leaves headroom below that
    consecutive-integer boundary instead of requiring support for such large profiles.
    The cap alone does not preserve small fractional contributions, so float totals and
    mentions still use math.fsum. Fraction totals remain exact and follow the same cap.

    Args:
        weights (NDArray[np.float64] | NDArray[np.object_]): Finite float64 or Fraction weights.

    Raises:
        ProfileError: An individual weight or the total exceeds the supported profile size.
    """
    error_message = f"Total ballot weight must be at most {MAX_TOTAL_BALLOT_WEIGHT} (2**50)."
    if np.any(weights > MAX_TOTAL_BALLOT_WEIGHT):
        raise ProfileError(error_message)

    total_weight = sum(weights) if weights.dtype == object else fsum(weights)
    if total_weight > MAX_TOTAL_BALLOT_WEIGHT:
        raise ProfileError(error_message)


def _non_default_index(index: pd.Index) -> NDArray | None:
    """
    Extract index labels unless they form a default RangeIndex.

    Args:
        index (pd.Index): Source dataframe index.

    Returns:
        NDArray | None: Labels for a nondefault index, or None for a RangeIndex starting at
            zero with step one. The returned array may share storage with the index.
    """
    if isinstance(index, pd.RangeIndex) and index.start == 0 and index.step == 1:
        return None
    return index.to_numpy()


def _standardize_weights(weights: NDArray) -> tuple[NDArray, WeightKind]:
    """
    Validate weights and select float64 or Fraction storage without changing their values.

    Any Fraction selects Fraction storage for the entire array. Other entries must then be
    integers or integral floats and are converted exactly. Without Fractions, every value must
    be exactly representable as float64. Supported floating inputs are Python float and NumPy
    float64; other floating dtypes require explicit caller conversion. Empty input uses FLOAT.

    Args:
        weights (NDArray): One-dimensional array of real, finite, nonnegative weights.
            Booleans are not weights, including in numeric arrays.

    Returns:
        tuple[NDArray, WeightKind]: Normalized weights and their representation. A compatible
            float64 input may be returned without copying; this function does not establish
            ownership.

    Raises:
        ProfileError: Input is not one-dimensional, contains invalid weights, mixes Fractions
            with non-integral floats, or requires lossy conversion to float64.
    """
    if weights.ndim != 1:
        raise ProfileError("weights must be a 1D array.")
    if weights.size == 0:
        return np.empty(0, dtype=np.float64), WeightKind.FLOAT

    if weights.dtype != object:
        _validate_numeric_weights(weights)
        return _numeric_weights_to_float64(weights), WeightKind.FLOAT

    try:
        for weight in weights:
            validate_weight(weight)

        if any(isinstance(weight, Fraction) for weight in weights):
            fractions = np.empty(len(weights), dtype=object)
            fractions[:] = [_as_fraction(weight) for weight in weights]
            return fractions, WeightKind.FRACTION

        return np.array([_to_float_weight(weight) for weight in weights]), WeightKind.FLOAT
    except (TypeError, ValueError) as error:
        raise ProfileError(str(error)) from error


def _numeric_weights_to_float64(weights: NDArray) -> NDArray[np.float64]:
    """
    Retain validated float64 weights or convert integer weights exactly.

    Args:
        weights (NDArray): Finite, nonnegative weights with an integer or float64 dtype.

    Returns:
        NDArray[np.float64]: Exact float64 values; compatible input storage may be reused.

    Raises:
        ProfileError: An integer cannot be represented exactly as float64.
    """
    if weights.dtype == np.float64:
        return weights

    # Smaller nonnegative integers are always exactly representable in float64.
    if weights.dtype.kind in "iu":
        for weight in weights[weights > 2**53]:
            try:
                _to_float_weight(weight)
            except ValueError as error:
                raise ProfileError(str(error)) from error

    return weights.astype(np.float64, copy=False)


def _validate_numeric_weights(weights: NDArray) -> None:
    """
    Validate nonnegative integer or float64 weights, rejecting other numeric dtypes.

    Args:
        weights (NDArray): Array with an integer or float64 dtype.

    Raises:
        ProfileError: The dtype is unsupported or an entry is negative, NaN, or infinite.
    """
    if weights.dtype.kind not in "iuf":
        raise ProfileError("Ballot weights must be real numbers, excluding booleans.")
    if weights.dtype.kind == "f" and weights.dtype.type is not np.float64:
        raise ProfileError(
            "Floating ballot weights must use float64; convert other dtypes explicitly."
        )
    if np.isnan(weights).any():
        raise ProfileError("Ballot weights cannot be NaN.")
    if np.isinf(weights).any():
        raise ProfileError("Ballot weights cannot be infinite.")
    if (weights < 0).any():
        raise ProfileError("Ballot weights cannot be negative.")


def _as_fraction(weight: Any) -> Fraction:
    """
    Retain a Fraction or convert a validated integral weight exactly.

    Args:
        weight (Any): Finite, nonnegative real weight already checked by ``validate_weight``.

    Returns:
        Fraction: Existing Fraction or exact integer weight, without float64 narrowing.

    Raises:
        ValueError: A non-integral float is mixed with Fraction weights.
    """
    if isinstance(weight, Fraction):
        return weight

    integer_weight = int(weight)
    if weight != integer_weight:
        raise ValueError(
            "Cannot mix Fraction weights with non-integral float weights. "
            "Pass Fraction(numerator, denominator) explicitly for fractional weights."
        )
    return Fraction(integer_weight)


def _attach_index_voterset_and_weight_columns(
    df: pd.DataFrame, data: ProfileVoterWeightData, *, rows: NDArray[np.intp] | None = None
) -> pd.DataFrame:
    """
    Append shared ballot columns and restore index labels on a payload dataframe.

    Args:
        df (pd.DataFrame): Payload dataframe to modify in place, already restricted to rows.
        data (_ProfileVoterWeightData): Shared columns and optional source index labels.
        rows (NDArray[np.intp] | None, optional): Positional source rows in output order.
            Defaults to None for all rows; negative indices count from the end.

    Returns:
        pd.DataFrame: The supplied dataframe, with Voter Set and Weight appended in that order
            and its index named Ballot Index. Restored index labels have independent storage.
    """
    if rows is None:
        df[VOTER_SET_COLUMN] = data.voter_set_column()
        df[WEIGHT_COLUMN] = data.weights

        if data.ballot_index is not None:
            df.index = pd.Index(data.ballot_index, copy=True)

    else:
        if data.voter_sets is not None:
            voters = data.voter_sets[rows]
        else:
            voters = np.empty(len(rows), dtype=object if len(data) else float)
            voters[:] = [EMPTY_VOTER_SET] * len(rows)

        df[VOTER_SET_COLUMN] = voters
        df[WEIGHT_COLUMN] = data.weights[rows]

        labels = (
            data.ballot_index[rows]
            if data.ballot_index is not None
            else np.where(rows < 0, rows + len(data), rows)
        )

        df.index = pd.Index(labels, copy=True)

    df.index.name = BALLOT_INDEX_NAME
    return df


def _concatenate_voter_weight_data(
    ballot_data: Sequence[ProfileVoterWeightData],
) -> ProfileVoterWeightData:
    """
    Concatenate aligned weights and voters using the common numeric policy, resetting labels.

    Args:
        ballot_data (Sequence[ProfileVoterWeightData]): Nonempty sequence in source row order.

    Returns:
        ProfileVoterWeightData: Owned weights and voter arrays without source index labels.

    Raises:
        ProfileError: Fraction weights are mixed with non-integral floats.
    """
    weights, weight_kind = _standardize_weights(
        np.concatenate([data.weights for data in ballot_data])
    )
    voters = None

    if any(data.voter_sets is not None for data in ballot_data):
        voters = np.concatenate(
            [np.asarray(data.voter_set_column(), dtype=object) for data in ballot_data]
        )

    return ProfileVoterWeightData(weights, voters, weight_kind)
