from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from copy import copy as shallow_copy
from dataclasses import dataclass
from typing import Self

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from votekit._df_schema import RANKING_COLUMN_PREFIX
from votekit.exceptions import ProfileError
from votekit.pref_profile.profile_matrix.shared_data import (
    EMPTY_VOTER_SET,
    ProfileVoterWeightData,
    _arrays_equal,
    _attach_index_voterset_and_weight_columns,
    _concatenate_voter_weight_data,
)
from votekit.types import Candidate
from votekit.utils import sort_candidates_pseudo_lexicographically

END_OF_RANKING_SET: frozenset[str] = frozenset({"~"})

# NOTE: Reserved candidate set IDs. Real candidate sets are numbered from 0 in order
# of first appearance, so ``cand_sets[cand_set_id - EMPTY_SET_ID]`` is the set for any ID. This
# access pattern makes the dense array representation easier to think about.
EMPTY_SET_ID = -2
END_OF_RANKING_ID = -1


@dataclass(frozen=True, slots=True, eq=False)
class RankMatrix:
    """
    Store ranked ballots as candidate-set IDs with their decoding table.

    Construction copies rankings into a read-only, index-sized integer array, freezes candidate
    sets, and retains column and candidate metadata as tuples. The validated shared-data block is
    reused. Candidate names are not normalized here.

    Args:
        voter_weight_data (_ProfileVoterWeightData): Shared data with one entry per ballot.
        rankings (NDArray): Two-dimensional integer array of candidate-set IDs, with one row per
            ballot and one column per ranking position. Tilde ends a ranking, so only tilde may
            follow it. Empty candidate sets are allowed before that sentinel. Empty dimensions are
            supported.
        cand_sets (tuple[frozenset[Candidate], ...]): Dense decoding table: slot 0 is the empty
            set (ID -2), slot 1 is the tilde set (ID -1), and slot ``cand_set_id + 2`` stores each
            real set. Entries must be unique. Mutable set entries are accepted and frozen during
            construction.
        ranking_columns (tuple[str, ...]): Ranking_i labels with unique, increasing numeric
            suffixes. Their count must match the ranking array's column count.
        candidates (tuple[Candidate, ...]): Profile candidate universe, including declared
            candidates no ballot ranks. Candidates occurring only on zero-weight ballots or in
            unused table entries may be absent from this tuple.
    """

    voter_weight_data: ProfileVoterWeightData
    rankings: NDArray
    cand_sets: tuple[frozenset[Candidate], ...]
    ranking_columns: tuple[str, ...]
    candidates: tuple[Candidate, ...]

    def __post_init__(self) -> None:
        """
        Validate ranking storage and candidate mappings, then retain independently owned data.

        Raises:
            TypeError: Rankings are not an array or candidate sets contain unsupported types.
            ValueError: The dtype, dimensions, ranking labels, candidate sets, IDs, or sentinel
                placement are invalid.
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
                with the shared data or ranking labels, or a nonpadding position follows tilde.
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

        if np.any(
            (self.rankings[:, :-1] == END_OF_RANKING_ID)
            & (self.rankings[:, 1:] != END_OF_RANKING_ID)
        ):
            raise ValueError("'~' values can only trail on a ranking.")

    def _validate_ranking_columns(self) -> None:
        """
        Check that ranking labels have unique numeric suffixes in increasing order.

        Raises:
            ValueError: A label is malformed or its numeric suffix does not follow the previous
                one.
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
        Check unique sets, reserved entries, and candidate membership on positive-weight ballots.

        Args:
            candidate_sets (tuple[frozenset[Candidate], ...]): Frozen candidate sets in dense ID
                order, beginning with the empty and tilde sets.

        Raises:
            TypeError: A real set contains a candidate other than a string or non-boolean integer.
            ValueError: Sets are repeated, reserved entries are incorrect, a real set is empty or
                contains tilde, or a positive-weight ballot references a candidate outside the
                profile universe.
        """
        if len(candidate_sets) < 2 or candidate_sets[:2] != (frozenset(), END_OF_RANKING_SET):
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

        if len(set(candidate_sets)) != len(candidate_sets):
            raise ValueError("Candidate sets must be unique.")

        if undeclared_set_ids:
            # Profiles may retain candidates outside their universe on zero-weight ballots.
            active_ids = np.unique(self.rankings[self.voter_weight_data.weights > 0])
            if undeclared_set_ids.intersection(active_ids):
                raise ValueError(
                    "Rankings on positive-weight ballots contain candidates outside "
                    "the declared candidates."
                )

    @classmethod
    def from_candidate_ids(
        cls,
        *,
        rankings: NDArray,
        candidates: Sequence[Candidate],
        voter_weight_data: ProfileVoterWeightData,
    ) -> Self:
        """
        Construct singleton rankings whose candidate IDs follow the supplied candidate order.

        Build the reserved empty and tilde entries followed by one singleton set per candidate.
        Candidate index i is therefore also candidate-set ID i. Ranking columns follow the array
        width, which may differ from the size of the candidate universe. Construction validates
        and copies the rankings while reusing the validated shared-data block.

        Args:
            rankings (NDArray): Two-dimensional integer array of candidate indices. EMPTY_SET_ID
                (-2) denotes an explicit empty position; END_OF_RANKING_ID (-1) ends a ranking and
                may only be followed by that same sentinel. Empty dimensions are supported.
            candidates (Sequence[Candidate]): Unique candidate identities in ID order. Names must
                already be normalized. Tied candidate sets require the general constructor.
            voter_weight_data (ProfileVoterWeightData): Shared data with one entry per ballot.

        Returns:
            Self: Matrix with owned rankings, singleton candidate sets, and Ranking_1 onward
                labels.

        Raises:
            TypeError: Rankings are not a NumPy array or candidate identities have invalid types.
            ValueError: Ranking dimensions, dtype, IDs, row count, sentinel placement, or
                candidate identities are invalid.
        """
        if not isinstance(rankings, np.ndarray):
            raise TypeError(f"rankings must be a numpy array, not {type(rankings)}.")
        if rankings.ndim != 2:
            raise ValueError(f"rankings must be a 2D array, not {rankings.ndim}D: {rankings.shape}")

        candidates = tuple(candidates)
        return cls(
            voter_weight_data=voter_weight_data,
            rankings=rankings,
            cand_sets=(EMPTY_VOTER_SET, END_OF_RANKING_SET, *(frozenset({c}) for c in candidates)),
            ranking_columns=tuple(
                f"{RANKING_COLUMN_PREFIX}{i + 1}" for i in range(rankings.shape[1])
            ),
            candidates=candidates,
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
            END_OF_RANKING_SET: END_OF_RANKING_ID,
        }
        for cand_set in cells.ravel():
            if cand_set not in cand_set_to_id:
                cand_set_to_id[cand_set] = len(cand_set_to_id) + EMPTY_SET_ID

        rankings = np.vectorize(cand_set_to_id.__getitem__, otypes=[int])(cells).reshape(
            cells.shape
        )

        return cls(
            voter_weight_data=ProfileVoterWeightData.from_df(df),
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
            cand_set_id (int): Valid ID from EMPTY_SET_ID through ``len(cand_sets) - 3``. IDs must
                come from this matrix; another matrix may assign them differently. This accessor
                assumes a validated ID and does not perform a separate range check.

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
                beginning with EMPTY_SET_ID (-2) and END_OF_RANKING_ID (-1).
        """
        return enumerate(self.cand_sets, start=EMPTY_SET_ID)

    def has_rankings(self) -> NDArray[np.bool_]:
        """
        Identify rows containing at least one ranking position; includes empty set rankings.

        Example:
            [-2, -1, -1, -1, -1] has rankings, but [-1, -1, -1, -1, -1] does not.
            (-2 means empty set, -1 means tilde padding.)

        Returns:
            NDArray[np.bool_]: One flag per source row, including zero-weight rows. Empty ranking
                sets count as positions; entirely padded rows and rows with no columns are false.
        """
        return np.any(self.rankings != END_OF_RANKING_ID, axis=1)

    def has_ranked_candidates(self) -> NDArray[np.bool_]:
        """
        Identify rows ranking at least one candidate, excluding empty sets and tilde padding.

        Returns:
            NDArray[np.bool_]: One flag per source row, including zero-weight rows. Rows
                containing only empty sets or padding, and rows with no columns, are false.
        """
        return np.any(self.rankings >= 0, axis=1)

    def has_ties(self) -> NDArray[np.bool_]:
        """
        Identify rows containing a candidate set with more than one member.

        Returns:
            NDArray[np.bool_]: One flag per source row, including zero-weight rows. Empty sets,
                padding, and repeated singleton candidates do not count as ties.
        """
        set_sizes = np.fromiter(map(len, self.cand_sets), dtype=np.intp)
        return np.any(set_sizes[self.rankings - EMPTY_SET_ID] > 1, axis=1)

    def get_ranking(self, row: int) -> tuple[frozenset[Candidate], ...]:
        """
        Decode one ranking, removing trailing padding and preserving empty ranking positions.

        Args:
            row (int): Positional row index, with negative indices counted from the end.

        Returns:
            tuple[frozenset[Candidate], ...]: Ranking sets in stored order. An entirely padded row
                returns an empty tuple. The immutable candidate sets may be shared.

        Raises:
            IndexError: The row is outside the matrix.
        """
        return tuple(
            self.get_cand_set_from_id(int(set_id))
            for set_id in self.rankings[row]
            if set_id != END_OF_RANKING_ID
        )

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
                    - END_OF_RANKING_SET
                ),
                axis=1,
                arr=self.rankings,
            ).max()
        )

    def to_df(self, *, rows: NDArray[np.intp] | None = None) -> pd.DataFrame:
        """
        Decode ranking IDs into a new dataframe with public candidate-set cells.

        Args:
            rows (NDArray[np.intp] | None, optional): Positional row indices in output order.
                Repeated and negative indices are supported. Defaults to None for all rows. Source
                ballot labels are preserved, including implicit labels.

        Returns:
            pd.DataFrame: Ranking columns in stored order, followed by Voter Set and Weight.
                Payload arrays are independent; candidate and voter frozensets can be shared. The
                index is named Ballot Index and uses retained labels when present; its label
                storage is independent of the matrix.
        """
        lookup = np.empty(len(self.cand_sets), dtype=object)
        lookup[:] = self.cand_sets
        rankings = self.rankings if rows is None else self.rankings[rows]
        translated_rankings = lookup[rankings - EMPTY_SET_ID]

        df = pd.DataFrame(translated_rankings, columns=list(self.ranking_columns))
        return _attach_index_voterset_and_weight_columns(df, self.voter_weight_data, rows=rows)

    def get_candidates_cast(self) -> frozenset[Candidate]:
        """
        Return the set of candidates that appear on any ballot with nonzero weight.

        Returns:
            frozenset[Candidate]: Candidates ranked on at least one ballot.
        """
        active_candidate_set_ids = np.unique(self.rankings[self.voter_weight_data.weights > 0])

        return frozenset().union(
            *(
                self.get_cand_set_from_id(cand_set_id)
                for cand_set_id in active_candidate_set_ids
                if cand_set_id >= 0
            )
        )

    def copy(self, *, shallow: bool = False) -> RankMatrix:
        """
        Return a copy of the matrix with independent arrays and tuples.

        Args:
            shallow (bool, optional): If True, return a shallow copy with shared arrays and
                tuples. Defaults to False.

        Returns:
            RankMatrix: New matrix with the same data.
        """
        if shallow:
            return shallow_copy(self)

        return RankMatrix(
            voter_weight_data=self.voter_weight_data.copy(),
            rankings=self.rankings,
            cand_sets=self.cand_sets,
            ranking_columns=self.ranking_columns,
            candidates=self.candidates,
        )

    def with_weights(self, weights: NDArray) -> RankMatrix:
        """
        Replace weights while sharing immutable rankings, voters, and source labels.

        Args:
            weights (NDArray): One replacement weight per source row.

        Returns:
            RankMatrix: Matrix with owned, validated weights. A nonempty candidate universe is
                preserved; an empty universe is inferred from the newly active rows.

        Raises:
            ProfileError: Weights are invalid or activate candidates outside a nonempty universe.
        """
        data = self.voter_weight_data.with_weights(weights)
        matrix = shallow_copy(self)

        object.__setattr__(matrix, "voter_weight_data", data)

        active_candidates = matrix.get_candidates_cast()
        if not self.candidates:
            candidates = tuple(sort_candidates_pseudo_lexicographically(active_candidates))
            object.__setattr__(matrix, "candidates", candidates)

        elif not active_candidates.issubset(self.candidates):
            raise ProfileError(
                "Candidates cast are not a subset of candidates list. The following "
                " candidates are in candidates_cast but not candidates: "
                f"{set(active_candidates) - set(self.candidates)}."
            )

        return matrix

    def group_ballots(self) -> RankMatrix:
        """
        Combine equal rankings in first-appearance order without dataframe conversion.

        Returns:
            RankMatrix: Grouped matrix with owned, read-only arrays. An empty result uses float64
                weights. Immutable candidate and voter sets may be shared.

        Raises:
            ProfileError: A grouped floating-point weight overflows to a nonfinite value.
        """
        first_rows_idx, group_ids = _group_ranking_rows(self.rankings)
        grouped_data = self.voter_weight_data.group_rows(first_rows_idx, group_ids)

        return RankMatrix(
            voter_weight_data=grouped_data,
            rankings=self.rankings[first_rows_idx],
            cand_sets=self.cand_sets,
            ranking_columns=self.ranking_columns,
            candidates=self.candidates,
        )


def _group_ranking_rows(rankings: NDArray[np.intp]) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """
    Identify equal rankings, ordering groups by their first source row.

    Candidate-set IDs must already be canonical, so equal sets have equal IDs. Empty positions and
    tilde padding remain distinct.

    Args:
        rankings (NDArray[np.intp]): Two-dimensional canonical ranking IDs, left unchanged.

    Returns:
        tuple[NDArray[np.intp], NDArray[np.intp]]: Representative source rows and the group
            assigned to each source row. Both arrays are empty for an empty matrix.
    """
    if not len(rankings):
        return np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp)

    if not rankings.shape[1]:
        return np.array([0], dtype=np.intp), np.zeros(len(rankings), dtype=np.intp)

    rows = np.ascontiguousarray(rankings)  # each row continuous in memory

    row_dtype = np.dtype(
        (np.void, rows.dtype.itemsize * rows.shape[1])
    )  # size of each row in bytes

    # view interprets each row is a single void element (fixed size block of raw bytes) so that we
    # may use it as a key for np.unique.
    row_keys = rows.view(row_dtype).reshape(-1)

    _, first_rows_idx, group_ids = np.unique(
        row_keys,
        return_index=True,
        return_inverse=True,
    )

    group_order = np.argsort(first_rows_idx)
    reordered_ids = np.empty(len(group_order), dtype=np.intp)
    reordered_ids[group_order] = np.arange(len(group_order))

    return first_rows_idx[group_order], reordered_ids[group_ids]


def _concatenate_rank_matrices(
    matrices: Sequence[RankMatrix],
    candidates: tuple[Candidate, ...],
    max_ranking_length: int,
) -> RankMatrix:
    """
    Concatenate rows, remapping candidate-set IDs and padding shorter rankings.

    Args:
        matrices (Sequence[RankMatrix]): Nonempty sequence of matrices in source row order.
        candidates (tuple[Candidate, ...]): Combined candidate universe.
        max_ranking_length (int): Output width, at least each source matrix's width.

    Returns:
        RankMatrix: Owned combined arrays with a canonical candidate-set table and reset labels.
    """
    data = _concatenate_voter_weight_data([matrix.voter_weight_data for matrix in matrices])
    rankings = np.full((len(data), max_ranking_length), END_OF_RANKING_ID, dtype=np.intp)
    candidate_set_ids = {frozenset(): EMPTY_SET_ID, END_OF_RANKING_SET: END_OF_RANKING_ID}

    offset = 0

    for matrix in matrices:
        remapping = np.empty(len(matrix.cand_sets), dtype=np.intp)
        for slot, candidate_set in enumerate(matrix.cand_sets):
            remapping[slot] = candidate_set_ids.setdefault(
                candidate_set, len(candidate_set_ids) + EMPTY_SET_ID
            )

        end = offset + len(matrix)
        rankings[offset:end, : matrix.rankings.shape[1]] = remapping[matrix.rankings - EMPTY_SET_ID]
        offset = end

    return RankMatrix(
        voter_weight_data=data,
        rankings=rankings,
        cand_sets=tuple(candidate_set_ids),
        ranking_columns=tuple(f"{RANKING_COLUMN_PREFIX}{i + 1}" for i in range(max_ranking_length)),
        candidates=tuple(candidates),
    )
