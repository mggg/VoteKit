import re
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from typing import Any, Iterator, Protocol, Self, Sequence, cast, runtime_checkable

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from votekit._df_schema import (
    BALLOT_INDEX_NAME,
    RANKING_COLUMN_PREFIX,
    VOTER_SET_COLUMN,
    WEIGHT_COLUMN,
)
from votekit.ballot import _to_float_weight, validate_weight
from votekit.exceptions import ProfileError
from votekit.types import Candidate

EMPTY_VOTER_SET: frozenset[str] = frozenset()
TILDE_SET: frozenset[str] = frozenset({"~"})

# NOTE: Reserved candidate set IDs. Real candidate sets are numbered from 0 in order
# of first appearance, so ``cand_sets[cand_set_id - EMPTY_SET_ID]`` is the set for any ID.
# This access pattern makes the dense array representation easier to think about.
EMPTY_SET_ID = -2
TILDE_SET_ID = -1


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
class _ProfileVoterWeightData:
    """
    Store aligned ballot weights, voter sets, and optional source index labels.

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
            ProfileError: A floating-point weight is negative, NaN, or infinite.
            TypeError: A voter entry cannot be converted to a frozenset.
        """
        self._validate_weight_voterset_index_alignment()

        self._validate_weight_values_and_types()
        owned_weights = np.copy(self.weights)
        owned_weights.setflags(write=False)

        owned_voter_sets = None
        if self.voter_sets is not None:
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

        Weights follow ``_normalize_weights``. A default zero-based, unit-step RangeIndex is
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
        weights, weight_kind = _normalize_weights(df[WEIGHT_COLUMN].to_numpy())

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
        if not isinstance(other, _ProfileVoterWeightData):
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


def _normalize_weights(weights: NDArray) -> tuple[NDArray, WeightKind]:
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


@runtime_checkable
class ProfileMatrix(Protocol):
    """
    Define the shared construction and export interface for profile matrices.

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
    def voter_weight_data(self) -> _ProfileVoterWeightData:
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

    def to_df(self) -> pd.DataFrame:
        """
        Export stored ballot data using the concrete matrix's public dataframe schema.

        Returns:
            pd.DataFrame: Ballot rows in stored order, with decoded candidate data, shared columns,
                and source index labels when retained. Mutating an export affects neither the
                matrix nor other exports. Immutable sets may be shared. Weight dtypes, column
                ordering, and index metadata may differ from the original input.
        """
        ...


def _attach_ballot_columns(df: pd.DataFrame, data: _ProfileVoterWeightData) -> pd.DataFrame:
    """
    Append shared ballot columns and restore index labels on a payload dataframe.

    Args:
        df (pd.DataFrame): Payload dataframe to modify in place. Its rows must align with data.
        data (_ProfileVoterWeightData): Shared columns and optional source index labels.

    Returns:
        pd.DataFrame: The supplied dataframe, with Voter Set and Weight appended in that order
            and its index named Ballot Index. Restored index labels have independent storage.
    """
    df[VOTER_SET_COLUMN] = data.voter_set_column()
    df[WEIGHT_COLUMN] = data.weights
    if data.ballot_index is not None:
        df.index = pd.Index(data.ballot_index, copy=True)
    df.index.name = BALLOT_INDEX_NAME
    return df


@dataclass(frozen=True, slots=True, eq=False)
class RankMatrix:
    """
    Store ranked ballots as candidate-set IDs with their decoding table.

    Construction copies rankings into a read-only, index-sized integer array, freezes candidate
    sets, and retains column and candidate metadata as tuples. The validated shared-data block
    is reused. Candidate names are not normalized here.

    Args:
        voter_weight_data (_ProfileVoterWeightData): Shared data with one entry per ballot.
        rankings (NDArray): Two-dimensional integer array of candidate-set IDs, with one row
            per ballot and one column per ranking position. Empty dimensions are supported.
        cand_sets (tuple[frozenset[Candidate], ...]): Dense decoding table: slot 0 is the empty
            set (ID -2), slot 1 is the tilde set (ID -1), and slot ``cand_set_id + 2`` stores
            each real set. Mutable set entries are accepted and frozen during construction.
        ranking_columns (tuple[str, ...]): Ranking_i labels with unique, increasing numeric
            suffixes. Their count must match the ranking array's column count.
        candidates (tuple[Candidate, ...]): Profile candidate universe, including declared
            candidates no ballot ranks. Candidates occurring only on zero-weight ballots or
            in unused table entries may be absent from this tuple.
    """

    voter_weight_data: _ProfileVoterWeightData
    rankings: NDArray
    cand_sets: tuple[frozenset[Candidate], ...]
    ranking_columns: tuple[str, ...]
    candidates: tuple[Candidate, ...]

    def __post_init__(self) -> None:
        """
        Validate ranking storage and candidate mappings, then retain independently owned data.

        Raises:
            TypeError: Rankings are not an array or candidate sets contain unsupported types.
            ValueError: The dtype, dimensions, ranking labels, candidate sets, or IDs are invalid.
        """
        self._validate_ranking_array()
        self._validate_ranking_columns()

        if any(not isinstance(cand_set, (set, frozenset)) for cand_set in self.cand_sets):
            raise TypeError("Candidate sets must be sets or frozensets.")
        owned_cand_sets = tuple(frozenset(cand_set) for cand_set in self.cand_sets)
        self._validate_candidate_sets(owned_cand_sets)

        if self.rankings.size:
            min_id = int(self.rankings.min())
            max_id = int(self.rankings.max())
            if min_id < EMPTY_SET_ID or max_id >= len(owned_cand_sets) + EMPTY_SET_ID:
                raise ValueError(
                    f"Ranking candidate-set IDs must be between {EMPTY_SET_ID} and "
                    f"{len(owned_cand_sets) + EMPTY_SET_ID - 1}; found {min_id} to {max_id}."
                )

        # Native index width prevents overflow when offsetting narrow integer IDs by two.
        owned_rankings = np.array(self.rankings, dtype=np.intp, copy=True)
        owned_rankings.setflags(write=False)

        object.__setattr__(self, "rankings", owned_rankings)
        object.__setattr__(self, "cand_sets", owned_cand_sets)
        object.__setattr__(self, "ranking_columns", tuple(self.ranking_columns))
        object.__setattr__(self, "candidates", tuple(self.candidates))

    def _validate_ranking_array(self) -> None:
        """
        Check integer ranking IDs and alignment with weights and ranking columns.

        Raises:
            TypeError: Rankings are not a NumPy array.
            ValueError: Rankings are not two-dimensional integers or row/column counts disagree
                with the shared data or ranking labels.
        """
        if not isinstance(self.rankings, np.ndarray):
            raise TypeError(f"rankings must be a numpy array, not {type(self.rankings)}.")
        if self.rankings.ndim != 2:
            raise ValueError(
                f"rankings must be a 2D array, not {self.rankings.ndim}D: {self.rankings.shape}"
            )
        if not np.issubdtype(self.rankings.dtype, np.integer):
            raise ValueError(f"rankings must have an integer dtype, not {self.rankings.dtype}.")

        if self.rankings.shape[0] != len(self.voter_weight_data):
            raise ValueError(
                f"rankings must have the same number of rows as weights: "
                f"{self.rankings.shape[0]} != {len(self.voter_weight_data)}"
            )
        if self.rankings.shape[1] != len(self.ranking_columns):
            raise ValueError(
                f"rankings must have the same number of columns as ranking_columns: "
                f"{self.rankings.shape[1]} != {len(self.ranking_columns)}"
            )

    def _validate_ranking_columns(self) -> None:
        """
        Check that ranking labels have unique numeric suffixes in increasing order.

        Raises:
            ValueError: A label is malformed or its numeric suffix does not follow the previous one.
        """
        ranking_col_re = rf"{re.escape(RANKING_COLUMN_PREFIX)}\d+"
        previous_rank = -1
        for column in self.ranking_columns:
            if not isinstance(column, str) or re.fullmatch(ranking_col_re, column) is None:
                raise ValueError(
                    f"ranking_columns must be of the form '{RANKING_COLUMN_PREFIX}<NUMBER>'. "
                    f"Found: {column!r}"
                )
            rank = int(column.removeprefix(RANKING_COLUMN_PREFIX))
            if rank <= previous_rank:
                raise ValueError("ranking_columns must have unique ranks in increasing order.")
            previous_rank = rank

    def _validate_candidate_sets(self, candidate_sets: tuple[frozenset[Candidate], ...]) -> None:
        """
        Check reserved sets and candidate membership on positive-weight ballots.

        Args:
            candidate_sets (tuple[frozenset[Candidate], ...]): Frozen candidate sets in dense ID
                order, beginning with the empty and tilde sets.

        Raises:
            TypeError: A real set contains a candidate other than a string or non-boolean integer.
            ValueError: Reserved entries are incorrect, a real set is empty or contains tilde,
                or a positive-weight ballot references a candidate outside the profile universe.
        """
        if len(candidate_sets) < 2 or candidate_sets[:2] != (frozenset(), TILDE_SET):
            raise ValueError("Candidate sets must begin with the reserved empty and tilde sets.")

        candidates = set(self.candidates)
        undeclared_set_ids: set[int] = set()
        for cand_set_id, cand_set in enumerate(candidate_sets[2:]):
            if not cand_set or "~" in cand_set:
                raise ValueError("Real candidate sets must be nonempty and cannot contain '~'.")
            if any(
                not isinstance(candidate, (str, int)) or isinstance(candidate, bool)
                for candidate in cand_set
            ):
                raise TypeError("Candidate sets must contain string or integer candidate names.")
            if not cand_set.issubset(candidates):
                undeclared_set_ids.add(cand_set_id)

        if undeclared_set_ids:
            # Profiles may retain candidates outside their universe on zero-weight ballots.
            active_ids = np.unique(self.rankings[self.voter_weight_data.weights > 0])
            if undeclared_set_ids.intersection(active_ids):
                raise ValueError(
                    "Rankings on positive-weight ballots contain candidates outside "
                    "the declared candidates."
                )

    @classmethod
    def from_df(cls, df: pd.DataFrame, candidates: Sequence[Candidate]) -> Self:
        """
        Encode a ranking dataframe as owned ranking IDs and shared ballot data.

        Ranking columns are processed in numeric suffix order. Real IDs are assigned in order of
        first appearance, scanning ballots top to bottom and ranking positions left to right.

        Args:
            df (pd.DataFrame): Profile dataframe with Ranking_i columns of frozenset candidate
                names, plus Weight and Voter Set columns. Names must already be normalized.
            candidates (Sequence[Candidate]): Profile candidate universe in declared order.

        Returns:
            Self: Matrix with an owned ranking array and dense candidate-set lookup table.

        Raises:
            KeyError: A required shared column is missing.
            TypeError: Ranking cells are unhashable or candidate sets contain unsupported types.
            ValueError: Ranking labels, candidate mappings, or shared-data structure are invalid.
            ProfileError: Shared weight validation rejects a value.
        """
        ranking_columns = tuple(
            sorted(
                (col for col in df.columns if col.startswith(RANKING_COLUMN_PREFIX)),
                key=lambda col: int(col.removeprefix(RANKING_COLUMN_PREFIX)),
            )
        )
        cells = df[list(ranking_columns)].to_numpy()

        cand_set_to_id: dict[frozenset[Candidate], int] = {
            EMPTY_VOTER_SET: EMPTY_SET_ID,
            TILDE_SET: TILDE_SET_ID,
        }
        for cand_set in cells.ravel():
            if cand_set not in cand_set_to_id:
                cand_set_to_id[cand_set] = len(cand_set_to_id) + EMPTY_SET_ID

        rankings = np.vectorize(cand_set_to_id.__getitem__, otypes=[int])(cells).reshape(
            cells.shape
        )

        return cls(
            voter_weight_data=_ProfileVoterWeightData.from_df(df),
            rankings=rankings,
            cand_sets=tuple(cand_set_to_id),  # insertion order is ID order
            ranking_columns=ranking_columns,
            candidates=tuple(candidates),
        )

    def __len__(self) -> int:
        """
        Return the number of ballots, including zero-weight ballots.
        """
        return len(self.voter_weight_data)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RankMatrix):
            return NotImplemented
        return (
            self.voter_weight_data == other.voter_weight_data
            and _arrays_equal(self.rankings, other.rankings)
            and self.cand_sets == other.cand_sets
            and self.candidates == other.candidates
        )

    @property
    def num_ranking_positions(self) -> int:
        """
        Return the number of stored ranking columns, including empty and tilde padding.
        """
        return self.rankings.shape[1]

    def get_cand_set_from_id(self, cand_set_id: int) -> frozenset[Candidate]:
        """
        Look up a candidate set by its matrix-local ID, including the two reserved IDs.

        Args:
            cand_set_id (int): Valid ID from EMPTY_SET_ID through ``len(cand_sets) - 3``.
                IDs must come from this matrix; another matrix may assign them differently.
                This accessor assumes a validated ID and does not perform a separate range check.

        Returns:
            frozenset[Candidate]: Stored candidate set; ID -2 returns the empty set and ID -1
                returns the tilde set.
        """
        return self.cand_sets[cand_set_id - EMPTY_SET_ID]

    def iter_candidate_sets(self) -> Iterator[tuple[int, frozenset[Candidate]]]:
        """
        Iterate over (ID, candidate set) pairs in matrix-local ID order, including reserved sets.

        Returns:
            Iterator[tuple[int, frozenset[Candidate]]]: Pairs of candidate-set ID and frozen set,
                beginning with EMPTY_SET_ID (-2) and TILDE_SET_ID (-1).
        """
        return enumerate(self.cand_sets, start=EMPTY_SET_ID)

    def max_candidates_ranked(self) -> int:
        """
        Count the largest number of distinct candidates ranked on any stored ballot.

        Tied candidates each count once; repeated candidates count once per ballot. Empty and
        unranked positions contribute nothing. Zero-weight ballots are included.

        Returns:
            int: Largest distinct-candidate count, or zero when there are no rows or columns.
        """
        if self.rankings.size == 0:
            return 0

        return int(
            np.apply_along_axis(
                lambda row: len(
                    frozenset.union(
                        *(self.get_cand_set_from_id(cand_set_id) for cand_set_id in row)
                    )
                    - TILDE_SET
                ),
                axis=1,
                arr=self.rankings,
            ).max()
        )

    def to_df(self) -> pd.DataFrame:
        """
        Decode ranking IDs into a new dataframe with public candidate-set cells.

        Returns:
            pd.DataFrame: Ranking columns in stored order, followed by Voter Set and Weight.
                Payload arrays are independent; candidate and voter frozensets can be shared.
                The index is named Ballot Index and uses retained labels when present; its label
                storage is independent of the matrix.
        """
        lookup = np.empty(len(self.cand_sets), dtype=object)
        lookup[:] = self.cand_sets
        translated_rankings = lookup[self.rankings - EMPTY_SET_ID]

        df = pd.DataFrame(translated_rankings, columns=list(self.ranking_columns))
        return _attach_ballot_columns(df, self.voter_weight_data)


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
