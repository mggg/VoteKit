from __future__ import annotations

import csv
import io
from collections.abc import Iterator
from functools import cached_property
from os import PathLike
from pathlib import Path
from typing import Optional, Self, Sequence, Tuple, Union, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from votekit._df_schema import (
    BALLOT_INDEX_NAME,
    RANKING_COLUMN_PREFIX,
    VOTER_SET_COLUMN,
    WEIGHT_COLUMN,
)
from votekit.ballot import Ballot, RankBallot
from votekit.exceptions import ProfileError
from votekit.pref_profile.csv_utils import (
    _parse_ballot_from_rank_csv,
    _parse_profile_data_from_rank_csv,
    _validate_rank_csv_format,
)
from votekit.pref_profile.pref_profile._candidate_names import (
    _normalize_candidate_name,
    _normalize_unique_candidates,
)
from votekit.pref_profile.pref_profile.preference_profile import PreferenceProfile
from votekit.pref_profile.profile_matrix import END_OF_RANKING_SET, RankMatrix
from votekit.pref_profile.utils import _sum_rank_profiles
from votekit.types import Candidate
from votekit.utils import (
    _validate_candidate_names,
    sort_candidates_pseudo_lexicographically,
)


def _normalize_rank_df_candidates(df: pd.DataFrame, context: str = "ranking cells") -> pd.DataFrame:
    """
    Normalize distinct ranking sets without modifying the supplied dataframe.

    Equal sets reuse normalized names. Candidate types are checked before reusing an equal set
    because boolean and floating-point values can compare equal to integer candidates.

    Args:
        df (pd.DataFrame): Dataframe with ranking columns containing candidate sets.
        context (str, optional): Source name used in validation errors. Defaults to "ranking cells".

    Returns:
        pd.DataFrame: Copy of the dataframe with normalized candidate names in ranking columns.
            Empty sets and literal tilde padding sets are preserved; spelling variants merge.

    Raises:
        TypeError: A ranking cell is not a set or a candidate has an unsupported type.
        ValueError: A normalized candidate name contains a prohibited value or character.
    """
    normalized_df = df.copy()
    ranking_cols = [
        column
        for column in df.columns
        if isinstance(column, str) and column.startswith(RANKING_COLUMN_PREFIX)
    ]

    # Include types in the keys so an invalid boolean cannot disappear behind an integer.
    candidate_names: dict[tuple[type, Candidate], None] = {}
    normalized_sets: dict[
        frozenset[Candidate], tuple[frozenset[Candidate], frozenset[Candidate]]
    ] = {}

    def normalize_ranking_set(candidate_set: frozenset[Candidate]) -> frozenset[Candidate]:
        if not isinstance(candidate_set, (set, frozenset)):
            raise TypeError("Ranking cells must contain sets or frozensets of candidates.")
        if candidate_set == END_OF_RANKING_SET:
            return END_OF_RANKING_SET

        frozen_set = frozenset(candidate_set)
        cached = normalized_sets.get(frozen_set)
        if cached is not None:
            original_set, normalized_set = cached
            if frozen_set is not original_set:
                invalid_candidates = [
                    candidate
                    for candidate in frozen_set
                    if not isinstance(candidate, (str, int)) or isinstance(candidate, bool)
                ]
                if invalid_candidates:
                    _validate_candidate_names(invalid_candidates, context)
            return normalized_set

        normalized = tuple(_normalize_candidate_name(candidate) for candidate in frozen_set)
        candidate_names.update(((type(candidate), candidate), None) for candidate in normalized)
        normalized_set = frozenset(normalized)
        normalized_sets[frozen_set] = (frozen_set, normalized_set)
        return normalized_set

    normalized_df[ranking_cols] = df[ranking_cols].map(normalize_ranking_set)
    _validate_candidate_names([candidate for _, candidate in candidate_names], context)
    return normalized_df


class RankProfile(PreferenceProfile):
    max_ranking_length: int
    _matrix: RankMatrix

    def __new__(cls, **kwargs) -> Self:
        """
        Preserve the explicitly requested profile type, including subclasses.
        """
        return cast(Self, super().__new__(cls, **kwargs))

    def __init__(
        self,
        *,
        ballots: Sequence[Ballot] = tuple(),
        candidates: Sequence[Candidate] = tuple(),
        max_ranking_length: Optional[int] = None,
        df: pd.DataFrame = pd.DataFrame(),
        include_zero_score: None = None,
    ):
        for index, ballot in enumerate(ballots):
            if not isinstance(ballot, RankBallot):
                raise ProfileError(
                    f"RankProfile requires RankBallots; ballot at index {index} "
                    f"is {type(ballot).__name__}."
                )

        self.candidates = _normalize_unique_candidates(
            candidates, f"{type(self).__name__}.candidates"
        )

        self.max_ranking_length = 0 if max_ranking_length is None else max_ranking_length

        if df.equals(pd.DataFrame()):
            rank_df, self.candidates_cast = self._init_from_rank_ballots(
                cast(Sequence[RankBallot], ballots)
            )

        else:
            rank_df, self.candidates_cast = self._init_from_rank_df(df)

        if self.candidates == tuple():
            self.candidates = _normalize_unique_candidates(
                self.candidates_cast, f"{type(self).__name__}.candidates"
            )

        self._validate_candidates()
        self._matrix = RankMatrix.from_df(rank_df, self.candidates)

        self.max_ranking_length = self._find_max_ranking_length()

        if self.max_ranking_length > 0:
            if self.max_candidates_ranked > self.max_ranking_length:
                msg = (
                    "This preference profile contains ballot(s) whose number of candidates "
                    "exceeds the length of the longest ranking. "
                    "It's likely that you need to clean the profile to remove ties. "
                    "If the profile is exactly as you expect, "
                    f"pass max_ranking_length={self.max_candidates_ranked} "
                    "to RankProfile()."
                )
                raise ValueError(msg)

        self._finalize_profile()

    def __update_ballot_ranking_data(
        self,
        rank_ballot_data: dict[str, list],
        idx: int,
        rank_ballot: RankBallot,
        candidates_cast: list[Candidate],
        num_ballots: int,
    ):
        """
        Update the ranking data from a ballot.

        Args:
            rank_ballot_data (dict[str, list]): Per-column ballot lists, updated in place.
            idx (int): Index of ballot.
            rank_ballot (RankBallot): Ballot.
            candidates_cast (list[Candidate]): Candidates cast, updated in place.
                Only positive-weight ballots contribute. Names can be strings or integers.
            num_ballots (int): Total number of ballots.
        """

        if rank_ballot.ranking is None:
            return

        for j, cand_set in enumerate(rank_ballot.ranking):
            for c in cand_set:
                if self.candidates != tuple():
                    if c not in self.candidates:
                        raise ProfileError(
                            f"Candidate {c} found in ballot {rank_ballot} but not in "
                            f"candidate list {self.candidates}."
                        )
                if rank_ballot.weight > 0 and c not in candidates_cast:
                    candidates_cast.append(c)

            if f"{RANKING_COLUMN_PREFIX}{j + 1}" not in rank_ballot_data:
                assert self.max_ranking_length is not None

                if self.max_ranking_length > 0:
                    raise ProfileError(
                        f"Max ranking length {self.max_ranking_length} given but "
                        f"ballot {rank_ballot} has length at least {j + 1}."
                    )
                rank_ballot_data[f"{RANKING_COLUMN_PREFIX}{j + 1}"] = [
                    END_OF_RANKING_SET
                ] * num_ballots
            rank_ballot_data[f"{RANKING_COLUMN_PREFIX}{j + 1}"][idx] = cand_set

    def __update_rank_ballot_data_attrs(
        self,
        rank_ballot_data: dict[str, list],
        idx: int,
        rank_ballot: RankBallot,
        candidates_cast: list[Candidate],
        num_ballots: int,
    ):
        """
        Update ballot data from a rank ballot.

        Args:
            rank_ballot_data (dict[str, list]): Per-column ballot lists, updated in place.
            idx (int): Index of ballot.
            rank_ballot (RankBallot): Ballot.
            candidates_cast (list[Candidate]): Candidates cast, updated in place.
                Only positive-weight ballots contribute. Names can be strings or integers.
            num_ballots (int): Total number of ballots.
        """
        rank_ballot_data[WEIGHT_COLUMN][idx] = rank_ballot.weight

        if rank_ballot.voter_set != frozenset():
            rank_ballot_data[VOTER_SET_COLUMN][idx] = rank_ballot.voter_set

        if rank_ballot.ranking is not None:
            self.__update_ballot_ranking_data(
                rank_ballot_data=rank_ballot_data,
                idx=idx,
                rank_ballot=rank_ballot,
                candidates_cast=candidates_cast,
                num_ballots=num_ballots,
            )

    def __init_rank_ballot_data(
        self,
        rank_ballots: Sequence[RankBallot],
    ) -> Tuple[int, dict[str, list]]:
        """
        Create the ballot data objects.

        Args:
            rank_ballots (Sequence[RankBallot]): Ballots in dataframe row order.

        Returns:
            Tuple[int, dict[str, list]]: Ballot count and allocated shared/ranking column lists.
                Ranking slots begin with tilde markers for unranked positions.
        """
        num_ballots = len(rank_ballots)

        rank_ballot_data: dict[str, list] = {
            WEIGHT_COLUMN: [np.nan] * num_ballots,
            VOTER_SET_COLUMN: [set()] * num_ballots,
        }
        assert self.max_ranking_length is not None

        if self.max_ranking_length > 0:
            rank_ballot_data.update(
                {
                    f"{RANKING_COLUMN_PREFIX}{i + 1}": [END_OF_RANKING_SET] * num_ballots
                    for i in range(self.max_ranking_length)
                }
            )
        return num_ballots, rank_ballot_data

    def __init_formatted_rank_df(
        self,
        rank_ballot_data: dict[str, list],
    ) -> pd.DataFrame:
        """
        Create a pandas dataframe from the ballot data.

        Args:
            rank_ballot_data (dict[str, list]): Dictionary storing ballot data.

        Returns:
            pd.DataFrame: Dataframe of profile.
        """
        df = pd.DataFrame(rank_ballot_data)
        col_order = [c for c in df.columns if RANKING_COLUMN_PREFIX in c] + [
            VOTER_SET_COLUMN,
            WEIGHT_COLUMN,
        ]

        df = df.loc[:, col_order]
        df.index.name = BALLOT_INDEX_NAME
        return df

    def _init_from_rank_ballots(
        self, ballots: Sequence[RankBallot]
    ) -> tuple[pd.DataFrame, tuple[Candidate, ...]]:
        """
        Build candidate-set dataframe rows and collect candidates on positive-weight ballots.

        Args:
            ballots (Sequence[RankBallot]): Ballots in the order their rows should appear.

        Returns:
            tuple[pd.DataFrame, tuple[Candidate, ...]]: Ranking dataframe and candidates cast.
                Dataframe cells contain candidate sets; matrix construction assigns their IDs.

        Raises:
            ProfileError: A ballot contains an undeclared candidate or exceeds a supplied maximum
                ranking length.
        """
        # `rank_ballot_data` sends {Weight, Voter Set} keys to a list to be
        # indexed in the same order as the output df containing information
        # for each ballot. So rank_ballot_data[<weight>][<index>] is the weight value for
        # the ballot at index <index> in the df.
        num_ballots, rank_ballot_data = self.__init_rank_ballot_data(ballots)

        candidates_cast: list[Candidate] = []

        for i, b in enumerate(ballots):
            self.__update_rank_ballot_data_attrs(
                rank_ballot_data=rank_ballot_data,
                idx=i,
                rank_ballot=b,
                candidates_cast=candidates_cast,
                num_ballots=num_ballots,
            )

        df = self.__init_formatted_rank_df(
            rank_ballot_data=rank_ballot_data,
        )

        return df, tuple(candidates_cast)

    def __validate_init_rank_df_params(self, df: pd.DataFrame) -> None:
        """
        Validate that the correct params were passed to the init method when constructing
        from a dataframe.

        Args:
            df (pd.DataFrame): Dataframe representation of ballots.

        Raises:
            ProfileError: max_ranking_length must be provided.
            ProfileError: Candidates must be provided.
        """
        boiler_plate = "When providing a dataframe and no ballot list to the init method, "
        if len(df) == 0:
            return

        if self.max_ranking_length == 0:
            raise ProfileError(
                boiler_plate + "max_ranking_length must be provided and be non-zero."
            )

    def __validate_init_rank_df(self, df: pd.DataFrame) -> None:
        """
        Validate that the df passed to the init method is of valid type.

        Args:
            df (pd.DataFrame): Dataframe representation of ballots.

        Raises:
            ProfileError: A ranking column is missing, unexpected, or duplicated.
            ProfileError: Weight column is missing.
            ProfileError: Voter set column is missing.
            ProfileError: Index column is misformatted.

        """
        if WEIGHT_COLUMN not in df.columns:
            raise ProfileError(f"Weight column not in dataframe: {df.columns}")
        if VOTER_SET_COLUMN not in df.columns:
            raise ProfileError(f"Voter Set column not in dataframe: {df.columns}")
        if df.index.name != BALLOT_INDEX_NAME:
            raise ProfileError(f"Index not named 'Ballot Index': {df.index.name}")
        expected_columns = {
            f"{RANKING_COLUMN_PREFIX}{rank}" for rank in range(1, self.max_ranking_length + 1)
        }
        for rank in range(1, self.max_ranking_length + 1):
            column = f"{RANKING_COLUMN_PREFIX}{rank}"
            if column not in df.columns:
                raise ProfileError(f"Ranking column '{column}' not in dataframe: {df.columns}")

        unexpected_columns = set(df.columns) - expected_columns - {WEIGHT_COLUMN, VOTER_SET_COLUMN}
        if unexpected_columns:
            raise ProfileError(f"Unexpected columns in ranked dataframe: {unexpected_columns}")
        if not df.columns.is_unique:
            raise ProfileError("Ranked dataframe columns must be unique.")

    def __find_candidates_cast_from_init_rank_df(self, df: pd.DataFrame) -> tuple[Candidate, ...]:
        """
        Find candidates ranked on positive-weight dataframe rows, excluding tilde markers.

        Args:
            df (pd.DataFrame): Profile dataframe with ranking columns of candidate sets and weights.

        Returns:
            tuple[Candidate, ...]: Distinct candidates in pseudo-lexicographic order when names
                support sorting. Invalid names are retained for subsequent candidate validation.
        """

        mask = df[WEIGHT_COLUMN] > 0

        candidates_cast: set[Candidate] = set()

        ranking_cols = [c for c in df.columns if c.startswith(RANKING_COLUMN_PREFIX)]
        sets = df.loc[mask, ranking_cols].to_numpy().ravel()
        candidates_cast |= set().union(*sets)

        candidates_cast.discard("~")

        try:
            candidates_cast_list = sort_candidates_pseudo_lexicographically(candidates_cast)
        except TypeError:
            # catch when an invalid candidate is casted, defer error handling to validation
            candidates_cast_list = list(candidates_cast)

        return tuple(candidates_cast_list)

    def _init_from_rank_df(self, df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[Candidate, ...]]:
        """
        Normalize a ranking dataframe and identify candidates on positive-weight rows.

        An empty dataframe resets the profile's maximum ranking length to zero.

        Args:
            df (pd.DataFrame): Profile dataframe with candidate-set ranking columns, shared columns,
                and an index named Ballot Index.

        Returns:
            tuple[pd.DataFrame, tuple[Candidate, ...]]: A copy with normalized ranking sets and
                candidates cast. Candidate-set IDs are encoded during matrix construction.

        Raises:
            ProfileError: Required construction parameters, columns, or the index name are invalid.
            TypeError: Ranking cells or candidate names have unsupported types.
            ValueError: Raw or normalized candidate names contain prohibited values or characters.
        """
        self.__validate_init_rank_df_params(df)
        self.__validate_init_rank_df(df)
        df = _normalize_rank_df_candidates(df, f"{type(self).__name__}.candidates_cast")
        candidates_cast = self.__find_candidates_cast_from_init_rank_df(df)

        if len(df) == 0:
            self.max_ranking_length = 0

        return df, candidates_cast

    def _find_max_ranking_length(self) -> int:
        """
        Compute and set the maximum ranking length of the profile.

        Returns:
            int: Max ranking length.

        """
        if self.max_ranking_length == 0 or self.max_ranking_length is None:
            return self._matrix.num_ranking_positions

        return self.max_ranking_length

    @cached_property
    def max_candidates_ranked(self) -> int:
        """
        The maximum number of candidates ranked on any ballot in the profile.

        Can be longer than max_ranking_length if a ballot has ties.
        E.g., a ballot that ranks two candidates tied for first and ranks no other candidates
        has length 1, but ranks 2 candidates in total.
        """
        return self._matrix.max_candidates_ranked()

    def get_ballot(self, row: int) -> RankBallot:
        """
        Materialize one ballot without constructing or caching the full ballot tuple.

        Args:
            row (int): Positional row index, with negative indices counted from the end.
                Source dataframe labels do not affect positional lookup.

        Returns:
            RankBallot: Ballot with the stored weight and voters, including zero-weight rows.

        Raises:
            IndexError: The row is outside the profile.
        """
        ranking = self._matrix.get_ranking(row)
        ballot_data = self._matrix.voter_weight_data
        voters = ballot_data.voter_sets[row] if ballot_data.voter_sets is not None else frozenset()
        return RankBallot(
            ranking=ranking or None, weight=ballot_data.weights[row], voter_set=voters
        )

    def iter_ballots(self) -> Iterator[RankBallot]:
        """
        Yield ballots in source row order without materializing or caching the full tuple.

        Yields:
            RankBallot: One ballot at a time, including zero-weight rows. Each iteration builds
                fresh ballot objects from the matrix; an existing ballot cache is not used.
        """
        for row in range(len(self._matrix)):
            yield self.get_ballot(row)

    @cached_property
    def ballots(self) -> tuple[RankBallot, ...]:
        """
        Materialize and cache ballots directly from matrix rows in source order.

        Returns:
            tuple[RankBallot, ...]: Ballots with weights and voters, including zero-weight rows.
        """
        return tuple(self.iter_ballots())

    def __add__(self, other) -> RankProfile:
        """
        Add two PreferenceProfiles by combining their ballot lists.
        """
        return _sum_rank_profiles([self, other])

    def group_ballots(self) -> RankProfile:
        """
        Combine equal rankings in first-appearance order, summing weights and unioning voters.

        Declared candidates and the ranking limit are preserved. Ballot indices reset to a
        default index. Fraction sums remain exact; float sums use NumPy accumulation.

        Returns:
            RankProfile: New profile with one ballot per distinct ranking.

        Raises:
            ProfileError: A grouped floating-point weight overflows to a nonfinite value.
        """
        return RankProfile._from_matrix(
            self._matrix.group_ballots(),
            max_ranking_length=self.max_ranking_length,
        )

    def copy(self) -> RankProfile:
        """
        Returns a shallow copy of the immutable RankProfile.

        Returns:
            RankProfile: New RankProfile object
        """
        return RankProfile._from_matrix(
            self._matrix,
            max_ranking_length=self.max_ranking_length,
        )

    def __eq__(self, other):
        if not isinstance(other, RankProfile):
            return False

        if self.max_ranking_length != other.max_ranking_length:
            return False

        return super().__eq__(other)

    def __str__(self) -> str:
        repr_str = "RankProfile\n"
        repr_str += f"Maximum ranking length: {self.max_ranking_length}\n"

        repr_str += (
            f"Candidates: {self.candidates}\n"
            f"Candidates who received votes: {self.candidates_cast}\n"
            f"Total number of Ballot objects: {self.num_ballots}\n"
            f"Total weight of Ballot objects: {self.total_ballot_wt}\n"
        )

        return repr_str

    __repr__ = __str__

    def __to_rank_csv_header(
        self, candidate_mapping: dict[Candidate, str], include_voter_set: bool
    ) -> list[list]:
        """
        Construct the header rows for the PrefProfile a custom CSV format.

        Args:
            candidate_mapping (dict[Candidate, str]): Mapping of candidate to ID.
                Candidates can be strings, integers, or mix of both.
            include_voter_set (bool): Whether or not to include the voter set of each
                ballot.
        """
        header = [
            ["VoteKit RankProfile", "v2"],
            ["Candidates"],
            [
                f"({str(c)}:{type(c).__name__}:{cand_label})"
                for c, cand_label in candidate_mapping.items()
            ],
        ]
        header += [["Max Ranking Length"], [str(self.max_ranking_length)]]
        header += [["Includes Voter Set"], [str(include_voter_set)]]
        header += [["="] * 10]

        return header

    def __to_rank_csv_ranking_list(
        self, rank_ballot: RankBallot, candidate_mapping: dict[Candidate, str]
    ) -> list:
        """
        Create the list of ranking data for a ballot in the profile.

        Args:
            rank_ballot (RankBallot): Ballot.
            candidate_mapping (dict[Candidate, str]): Mapping candidate names to IDs.
                Candidates can be strings, integers, or mix of both.

        """
        assert self.max_ranking_length is not None

        if rank_ballot.ranking is not None:
            ranking_list = [
                (set([candidate_mapping[c] for c in cand_set]) if cand_set != frozenset() else "{}")
                for cand_set in rank_ballot.ranking
            ]
            if len(ranking_list) != self.max_ranking_length:
                ranking_list += [""] * (self.max_ranking_length - len(ranking_list))

            return ranking_list

        return [""] * self.max_ranking_length

    def __to_rank_csv_ballot_row(
        self,
        ballot: RankBallot,
        include_voter_set: bool,
        candidate_mapping: dict[Candidate, str],
        weight_precision: int,
    ) -> list[list]:
        """
        Create the row for a ballot in the profile.

        Args:
            ballot (Ballot): Ballot.
            include_voter_set (bool): Whether or not to include the voter set of each
                ballot.
            candidate_mapping (dict[Candidate, int]): Mapping candidate names to
                integers. Candidates can be strings, integers, or mix of both.
            weight_precision (int): Number of decimals to round float weights to.

        """
        row = self.__to_rank_csv_ranking_list(ballot, candidate_mapping)
        row += ["&"]

        row += [round(ballot.weight, weight_precision), "&"]

        if include_voter_set:
            row += [v for v in sorted(ballot.voter_set)]

        return row

    def __to_rank_csv_data_column_names(
        self, include_voter_set: bool, candidate_mapping: dict[Candidate, str]
    ) -> list:
        """
        Create the data column header.

        Args:
            include_voter_set (bool): Whether or not to include the voter set of each
                ballot.
            candidate_mapping (dict[Candidate, str]): Maps candidate names to
                prefixes. Candidates can be strings, integers, or mix of both.
        """
        assert self.max_ranking_length is not None
        data_col_names = list(self._matrix.ranking_columns)
        # NOTE: CSV layout puts Weight before Voter Set, the reverse of df. CSV does not affect the
        # df column order. Don't reorder here: from_csv reads Weight by position, so existing CSVs
        # will break if the order is changed.
        data_col_names += ["&", WEIGHT_COLUMN, "&"]

        if include_voter_set:
            data_col_names += [VOTER_SET_COLUMN]

        return data_col_names

    def to_csv(
        self,
        fpath: str | PathLike | Path | None = None,
        include_voter_set: bool = False,
        weight_precision: int = 2,
    ) -> str | None:
        """
        Saves PreferenceProfile to a custom CSV format.

        Args:
            fpath (str | PathLike | Path | None, optional): Path to the saved csv. If None,
                returns the CSV content as a string instead of writing to disk. Defaults to None.
            include_voter_set (bool, optional): Whether or not to include the voter set of each
                ballot. Defaults to False.
            weight_precision (int): Number of decimals to round float weights to. Defaults to 2.

        Returns:
            str | None: The CSV content as a string when ``fpath`` is None, otherwise None.

        Raises:
            ProfileError: Cannot write a profile with no ballots to a csv.
            ValueError: File path must be provided.
        """
        if fpath is not None and str(fpath) == "":
            raise ValueError("File path must be provided.")

        if len(self.ballots) == 0:
            raise ProfileError("Cannot write a profile with no ballots to a csv.")

        if self._matrix.voter_weight_data.has_fraction_weights:
            raise ValueError(
                "RankProfile CSV does not support rational weights. Convert a copied profile "
                "to float first if a lossy export is acceptable."
            )

        candidate_mapping = {c: str(i) for i, c in enumerate(self.candidates)}

        header = self.__to_rank_csv_header(candidate_mapping, include_voter_set)
        data_col_names = self.__to_rank_csv_data_column_names(include_voter_set, candidate_mapping)
        ballot_rows = [
            self.__to_rank_csv_ballot_row(b, include_voter_set, candidate_mapping, weight_precision)
            for b in self.ballots
        ]
        rows = header + [data_col_names] + ballot_rows

        if fpath is None:
            sio = io.StringIO(newline="")
            writer = csv.writer(sio)
            writer.writerows(rows)
            return sio.getvalue()

        with open(
            str(fpath),
            "w",
            newline="",
            encoding="utf-8",
        ) as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(rows)
        return None

    @classmethod
    def from_csv(cls, fpath: Union[str, PathLike, Path]) -> PreferenceProfile:
        """
        Creates a PreferenceProfile from a csv, formatted from the ``to_csv`` method.

        Args:
            fpath (Union[str, PathLike, Path]): Path to csv.

        Raises:
            ValueError: If csv is improperly formatted for VoteKit.
            ProfileError: If read profile has no rankings or scores.

        """
        with open(str(fpath), "r") as file:
            reader = csv.reader(file)
            csv_data = list(reader)

        _validate_rank_csv_format(csv_data)

        (
            inv_candidate_mapping,
            max_ranking_length,
            includes_voter_set,
            break_indices,
        ) = _parse_profile_data_from_rank_csv(csv_data)

        ballots = [
            _parse_ballot_from_rank_csv(
                row,
                includes_voter_set,
                break_indices,
                inv_candidate_mapping,
            )
            for row in csv_data[9:]
        ]

        return cls(
            ballots=tuple(ballots),
            candidates=tuple(inv_candidate_mapping.values()),
            max_ranking_length=max_ranking_length,
        )

    # Static because we intend to bypass the constructor
    @staticmethod
    def _from_matrix(matrix: RankMatrix, *, max_ranking_length: int) -> RankProfile:
        """
        Create a RankProfile from a RankMatrix.

        Args:
            RankMatrix (RankMatrix): A RankMatrix object.

        Returns:
            RankProfile: A RankProfile object.
        """
        profile = object.__new__(RankProfile)
        profile._matrix = matrix
        profile.candidates = matrix.candidates
        profile.max_ranking_length = max_ranking_length
        profile.candidates_cast = tuple(
            sort_candidates_pseudo_lexicographically(matrix.get_candidates_cast())
        )
        profile._finalize_profile()
        return profile

    def to_matrix(self) -> RankMatrix:
        """
        Returns a deep copy of the RankMatrix representation of the RankProfile.

        Returns:
            RankMatrix: A RankMatrix object.
        """
        return self._matrix.copy()

    def _with_weights(self, weights: NDArray) -> RankProfile:
        """
        Replace ballot weights while sharing immutable rankings, voters, and index labels.

        Args:
            weights (NDArray): One replacement weight per ballot, in source row order.

        Returns:
            RankProfile: Plain rank profile with owned weights and recomputed totals and active
                candidates. An empty candidate universe is inferred from the new active rows.

        Raises:
            ProfileError: Replacement weights are invalid or newly active candidates fall outside
                a nonempty declared candidate universe.
        """
        new_matrix = self._matrix.with_weights(weights)
        profile = RankProfile._from_matrix(new_matrix, max_ranking_length=self.max_ranking_length)
        profile._validate_candidates()

        return profile
