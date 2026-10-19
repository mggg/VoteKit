from __future__ import annotations

import csv
import io
from functools import cached_property
from os import PathLike
from pathlib import Path
from typing import Optional, Self, Sequence, Tuple, Union, cast

import numpy as np
import pandas as pd

from votekit._df_schema import (
    BALLOT_INDEX_NAME,
    VOTER_SET_COLUMN,
    WEIGHT_COLUMN,
)
from votekit.ballot import Ballot, ScoreBallot
from votekit.exceptions import ProfileError
from votekit.pref_profile.csv_utils import (
    _parse_ballot_from_score_csv,
    _parse_profile_data_from_score_csv,
    _validate_score_csv_format,
)
from votekit.pref_profile.pref_profile._candidate_names import _normalize_unique_candidates
from votekit.pref_profile.pref_profile.preference_profile import PreferenceProfile
from votekit.pref_profile.profile_matrix import ScoreMatrix
from votekit.pref_profile.utils import _sum_score_profiles
from votekit.types import Candidate
from votekit.utils import sort_candidates_pseudo_lexicographically


def _normalize_score_df_candidates(
    df: pd.DataFrame, context: str = "score columns"
) -> pd.DataFrame:
    """
    Normalize score-column names without modifying the supplied dataframe.

    Args:
        df (pd.DataFrame): Dataframe with candidate-labelled score columns and shared columns.
        context (str, optional): Source name used in validation errors. Defaults to "score columns".

    Returns:
        pd.DataFrame: Copy with normalized candidate-column names in their original order.

    Raises:
        ProfileError: Normalized score names collide with each other or a shared column name.
        TypeError: A candidate column has an unsupported label type.
        ValueError: A normalized label contains a prohibited value or character.
    """
    shared_columns = {WEIGHT_COLUMN, VOTER_SET_COLUMN}
    score_columns = [column for column in df.columns if column not in shared_columns]
    normalized_columns = _normalize_unique_candidates(score_columns, context)
    if shared_columns.intersection(normalized_columns):
        raise ProfileError("Candidate names cannot normalize to reserved dataframe column names.")

    column_mapping = dict(zip(score_columns, normalized_columns, strict=True))
    return df.rename(columns=column_mapping)


class ScoreProfile(PreferenceProfile):
    _matrix: ScoreMatrix

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
    ):
        for index, ballot in enumerate(ballots):
            if not isinstance(ballot, ScoreBallot):
                raise ProfileError(
                    f"ScoreProfile requires ScoreBallots; ballot at index {index} "
                    f"is {type(ballot).__name__}."
                )

        self.candidates = _normalize_unique_candidates(
            candidates, f"{type(self).__name__}.candidates"
        )

        if df.equals(pd.DataFrame()):
            score_df, self.candidates_cast = self._init_from_score_ballots(
                cast(Sequence[ScoreBallot], ballots)
            )

        else:
            score_df, self.candidates_cast = self._init_from_score_df(df)

        if self.candidates == tuple():
            self.candidates = _normalize_unique_candidates(
                self.candidates_cast, f"{type(self).__name__}.candidates"
            )

        self._validate_candidates()

        self._matrix = ScoreMatrix.from_df(score_df, self.candidates)

        self.max_ranking_length = None
        self._finalize_profile()

    def __update_ballot_scores_data(
        self,
        score_ballot_data: dict[str | int, list],
        idx: int,
        ballot: ScoreBallot,
        candidates_cast: list[Candidate],
        num_ballots: int,
        seen_candidates: list[Candidate],
    ) -> None:
        """
        Update the score data from a ballot.

        Args:
            score_ballot_data (dict[str | int, list]): Per-column ballot lists, updated in place.
                Dictionary keys represent the column names.
            idx (int): Index of ballot.
            ballot (ScoreBallot): Ballot.
            candidates_cast (list[Candidate]): Candidates cast, updated in place.
                Only positive-weight ballots contribute. Names can be strings or integers.
            num_ballots (int): Total number of ballots.
            seen_candidates (list[Candidate]): Encountered candidates, updated in place.
                Starts with declared candidates and appends new candidates in encounter order.
        """
        if ballot.scores is None:
            return

        for c, score in ballot.scores.items():
            if c not in seen_candidates:
                if self.candidates:
                    raise ProfileError(
                        f"Candidate {c} found in ballot {ballot} but not in "
                        f"candidate list {self.candidates}."
                    )
                seen_candidates.append(c)
            if ballot.weight > 0 and c not in candidates_cast:
                candidates_cast.append(c)
            if c not in score_ballot_data:
                score_ballot_data[c] = [np.nan] * num_ballots
            score_ballot_data[c][idx] = score

    def __update_score_ballot_data_attrs(
        self,
        score_ballot_data: dict[str | int, list],
        idx: int,
        ballot: ScoreBallot,
        candidates_cast: list[Candidate],
        num_ballots: int,
        seen_candidates: list[Candidate],
    ) -> None:
        """
        Update all ballot data from a ballot.

        Args:
            score_ballot_data (dict[Candidate, list]): Per-column ballot lists, updated in place.
                Dictionary keys represent the column names.
                Candidates can be strings, integers, or mix of both.
            idx (int): Index of ballot.
            ballot (ScoreBallot): Ballot.
            candidates_cast (list[Candidate]): Candidates cast, updated in place.
                Only positive-weight ballots contribute. Names can be strings or integers.
            num_ballots (int): Total number of ballots.
            seen_candidates (list[Candidate]): Candidates encountered so far, in column order.
        """
        score_ballot_data[WEIGHT_COLUMN][idx] = ballot.weight

        if ballot.voter_set != frozenset():
            score_ballot_data[VOTER_SET_COLUMN][idx] = ballot.voter_set

        if ballot.scores is not None:
            self.__update_ballot_scores_data(
                score_ballot_data=score_ballot_data,
                idx=idx,
                ballot=ballot,
                candidates_cast=candidates_cast,
                num_ballots=num_ballots,
                seen_candidates=seen_candidates,
            )

    def __init_score_ballot_data(
        self, ballots: Sequence[ScoreBallot]
    ) -> Tuple[int, dict[str | int, list]]:
        """
        Create the ballot data objects.

        Args:
            ballots (Sequence[ScoreBallot]): Ballots in dataframe row order.

        Returns:
            Tuple[int, dict[str | int, list]]: Ballot count and allocated column lists.
                Declared candidate columns begin with NaN for unscored entries.
        """
        num_ballots = len(ballots)

        score_ballot_data: dict[str | int, list] = {
            WEIGHT_COLUMN: [np.nan] * num_ballots,
            VOTER_SET_COLUMN: [set()] * num_ballots,
        }

        if self.candidates != tuple():
            score_ballot_data.update({cand: [np.nan] * num_ballots for cand in self.candidates})

        return num_ballots, score_ballot_data

    def __init_formatted_score_df(
        self,
        score_ballot_data: dict[str | int, list],
        candidates_cast: list[Candidate],
        seen_candidates: list[Candidate],
    ) -> pd.DataFrame:
        """
        Create a pandas dataframe from the ballot data.

        Args:
            score_ballot_data (dict[str | int, list]): Per-column ballot lists keyed by candidate or
                shared-column name.
            candidates_cast (list[Candidate]): Candidates on positive-weight ballots.
                Candidates can be strings, integers, or mix of both.
            seen_candidates (list[Candidate]): Candidates in column order.
                Candidates can be strings, integers, or mix of both.

        Returns:
            pd.DataFrame: Candidate-labelled scores, followed by Voter Set and Weight.
                Columns follow declared candidate order, or pseudo-lexicographic order when
                candidates are inferred from ballots.
        """
        df = pd.DataFrame(score_ballot_data)
        temp_col_order = [
            VOTER_SET_COLUMN,
            WEIGHT_COLUMN,
        ]

        col_order = list(seen_candidates) + temp_col_order

        if self.candidates == tuple():
            remaining_cands = set(candidates_cast) - set(df.columns)
            empty_df_cols = np.full((len(df), len(remaining_cands)), np.nan)
            df[list(remaining_cands)] = empty_df_cols
            col_order = (
                list(sort_candidates_pseudo_lexicographically(seen_candidates)) + temp_col_order
            )

        df = df.loc[:, col_order]
        df.index.name = BALLOT_INDEX_NAME
        return df

    def _init_from_score_ballots(
        self, ballots: Sequence[ScoreBallot]
    ) -> tuple[pd.DataFrame, tuple[Candidate, ...]]:
        """
        Build candidate-labelled score rows and collect candidates on positive-weight ballots.

        Args:
            ballots (Sequence[ScoreBallot]): Ballots in the order their rows should appear.

        Returns:
            tuple[pd.DataFrame, tuple[Candidate, ...]]: Score dataframe and candidates cast.
                Declared candidate order is retained; otherwise columns use pseudo-lexicographic
                order.

        Raises:
            ProfileError: A ballot scores a candidate outside an explicitly supplied candidate list.
        """
        # `score_ballot_data` sends {Weight, Voter Set} keys to a list to be
        # indexed in the same order as the output df containing information
        # for each ballot. So ballot_data[<weight>][<index>] is the weight value for
        # the ballot at index <index> in the df.
        num_ballots, score_ballot_data = self.__init_score_ballot_data(ballots)

        candidates_cast: list[Candidate] = []
        seen_candidates: list[Candidate] = list(self.candidates)

        for i, b in enumerate(ballots):
            self.__update_score_ballot_data_attrs(
                score_ballot_data=score_ballot_data,
                idx=i,
                ballot=b,
                candidates_cast=candidates_cast,
                num_ballots=num_ballots,
                seen_candidates=seen_candidates,
            )

        df = self.__init_formatted_score_df(
            score_ballot_data=score_ballot_data,
            candidates_cast=candidates_cast,
            seen_candidates=seen_candidates,
        )
        return df, tuple(candidates_cast)

    def __validate_init_score_df_params(self, df: pd.DataFrame) -> None:
        """
        Validate that the correct params were passed to the init method when constructing
        from a dataframe.

        Args:
            df (pd.DataFrame): Dataframe representation of ballots.

        Raises:
            ValueError: One of contains_rankings and contains_scores must be True.
            ValueError: If contains_rankings is True, max_ranking_length must be provided.
            ValueError: Candidates must be provided.
        """
        boiler_plate = "When providing a dataframe and no ballot list to the init method, "
        if len(df) == 0:
            return

        if self.candidates == tuple():
            raise ProfileError(boiler_plate + "candidates must be provided.")

    def __validate_init_score_df(self, df: pd.DataFrame) -> None:
        """
        Validate shared columns and the index before normalizing score-column names.

        Args:
            df (pd.DataFrame): Dataframe representation of ballots.

        Raises:
            ProfileError: Weight or voter-set columns are missing, or the index name is invalid.

        """
        if WEIGHT_COLUMN not in df.columns:
            raise ProfileError(f"Weight column not in dataframe: {df.columns}")
        if VOTER_SET_COLUMN not in df.columns:
            raise ProfileError(f"Voter Set column not in dataframe: {df.columns}")
        if df.index.name != BALLOT_INDEX_NAME:
            raise ProfileError(f"Index not named 'Ballot Index': {df.index.name}")

    def __find_candidates_cast_from_init_score_df(self, df: pd.DataFrame) -> tuple[Candidate, ...]:
        """
        Compute which candidates received votes from the df and set the candidates_cast and
        candidates attr.

        Args:
            df (pd.DataFrame): Dataframe representation of ballots.

        Returns:
            tuple[str | int]: Candidates cast.
        """

        mask = df[WEIGHT_COLUMN] > 0

        candidates_cast: set[Candidate] = set()
        cand_cols = df.columns.difference([VOTER_SET_COLUMN, WEIGHT_COLUMN])
        positive = df.loc[mask, list(cand_cols)].gt(0).any()
        # .any() applies along the columns, so we get a boolean series where the
        # value is True the candidate has any positive score the column
        candidates_cast |= set(positive[positive].index)
        try:
            candidates_cast_list = sort_candidates_pseudo_lexicographically(candidates_cast)
        except TypeError:
            # catch when an invalid candidate is casted, defer error handling to validation
            candidates_cast_list = list(candidates_cast)

        return tuple(candidates_cast_list)

    def _init_from_score_df(self, df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[Candidate, ...]]:
        """
        Normalize a score dataframe and find candidates with positive scores and ballot weights.

        Args:
            df (pd.DataFrame): Profile dataframe with candidate-labelled scores, shared columns,
                and an index named Ballot Index.

        Returns:
            tuple[pd.DataFrame, tuple[Candidate, ...]]: A copy with normalized score-column names
                and candidates cast.

        Raises:
            ProfileError: Required candidates or columns are missing, the index name is invalid,
                or normalized score names collide with each other or a shared column name.
            TypeError: Candidate-column labels have unsupported types.
            ValueError: Raw or normalized candidate names contain prohibited values or characters.
        """
        self.__validate_init_score_df_params(df)
        self.__validate_init_score_df(df)
        df = _normalize_score_df_candidates(df, f"{type(self).__name__}.candidates_cast")

        for candidate in self.candidates:
            if candidate not in df.columns:
                raise ProfileError(f"Candidate column '{candidate}' not in dataframe: {df.columns}")

        candidates_cast = self.__find_candidates_cast_from_init_score_df(df)
        return df, candidates_cast

    @cached_property
    def ballots(self: ScoreProfile) -> tuple[ScoreBallot, ...]:
        """
        Materialize and cache ballots directly from matrix rows in source order.

        Returns:
            tuple[ScoreBallot, ...]: Ballots with weights and voters, including zero-weight rows.
        """
        matrix = self._matrix
        ballot_data = matrix.voter_weight_data
        column_positions = {
            candidate: column for column, candidate in enumerate(matrix.score_columns)
        }
        candidate_columns = [
            (candidate, column_positions[candidate])
            for candidate in self.candidates
            if candidate in column_positions
        ]

        ballots: list[ScoreBallot] = []
        for row_index, score_row in enumerate(matrix.scores):
            scores = {
                candidate: score_row[column]
                for candidate, column in candidate_columns
                if not np.isnan(score_row[column])
            }
            voters = (
                ballot_data.voter_sets[row_index]
                if ballot_data.voter_sets is not None
                else frozenset()
            )
            ballots.append(
                ScoreBallot(
                    scores=scores or None, weight=ballot_data.weights[row_index], voter_set=voters
                )
            )

        return tuple(ballots)

    def __add__(self, other):
        """
        Add two PreferenceProfiles by combining their ballot lists.
        """
        return _sum_score_profiles([self, other])

    def group_ballots(self) -> ScoreProfile:
        """
        Combine equal scores in first-appearance order, summing weights and unioning voters.

        Declared candidates and score-column order are preserved. Ballot indices reset to a
        default index. Fraction sums remain exact; float sums use NumPy accumulation.
        Unscored entries compare equally but remain distinct from explicit zero scores.

        Returns:
            ScoreProfile: New profile with one ballot per distinct score row.

        Raises:
            ProfileError: A grouped floating-point weight overflows to a nonfinite value.
        """
        return ScoreProfile._from_matrix(self._matrix.group_ballots())

    def copy(self) -> ScoreProfile:
        """
        Return a new profile sharing the immutable score matrix.

        Returns:
            ScoreProfile: New ScoreProfile object
        """
        return ScoreProfile._from_matrix(
            matrix=self._matrix,
        )

    def __eq__(self, other):
        if not isinstance(other, ScoreProfile):
            return False

        return super().__eq__(other)

    def __str__(self) -> str:
        repr_str = "ScoreProfile\n"

        repr_str += (
            f"Candidates: {self.candidates}\n"
            f"Candidates who received votes: {self.candidates_cast}\n"
            f"Total number of Ballot objects: {self.num_ballots}\n"
            f"Total weight of Ballot objects: {self.total_ballot_wt}\n"
        )

        return repr_str

    __repr__ = __str__

    def __to_score_csv_header(
        self, candidate_mapping: dict[Candidate, str], include_voter_set: bool
    ) -> list[list]:
        """
        Construct the header rows for the PrefProfile a custom CSV format.

        Args:
            candidate_mapping (dict[Candidate, str]): Candidate name mapped to integer IDs.
                integer IDs are cast to strings for csv.
                Candidates can be strings, integers, or mix of both.
            include_voter_set (bool): Whether or not to include the voter set of each
                ballot.
        """
        header = [
            ["VoteKit ScoreProfile", "v2"],
            ["Candidates"],
            [
                f"({c}:{type(c).__name__}:{cand_label})"
                for c, cand_label in candidate_mapping.items()
            ],
        ]
        header += [["Includes Voter Set"], [str(include_voter_set)]]
        header += [["="] * 10]

        return header

    def __to_score_csv_score_list(self, score_ballot: ScoreBallot) -> list:
        """
        Create the list of score data for a ballot in the profile.

        Args:
            ballot (Ballot): Ballot.

        """
        if score_ballot.scores is not None:
            return [float(score_ballot.scores.get(c, 0)) for c in self.candidates]

        return [""] * len(self.candidates)

    def __to_score_csv_ballot_row(
        self,
        score_ballot: ScoreBallot,
        include_voter_set: bool,
        weight_precision: int,
    ) -> list[list]:
        """
        Create the row for a ballot in the profile.

        Args:
            score_ballot (ScoreBallot): Ballot.
            include_voter_set (bool): Whether or not to include the voter set of each
                ballot.
            weight_precision (int): Number of decimals to round float weights to.


        """
        row = self.__to_score_csv_score_list(score_ballot)
        row += ["&"]

        row += [round(score_ballot.weight, weight_precision), "&"]

        if include_voter_set:
            row += [v for v in sorted(score_ballot.voter_set)]

        return row

    def __to_score_csv_data_column_names(
        self, include_voter_set: bool, candidate_mapping: dict[Candidate, str]
    ) -> list:
        """
        Create the data column header.

        Args:
            include_voter_set (bool): Whether or not to include the voter set of each
                ballot.
            candidate_mapping (dict[Candidate, str]): Maps candidate names to IDs.
                Candidates can be strings, integers, or mix of both.
        """
        data_col_names = [f"{cand_label}" for cand_label in candidate_mapping.values()]
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
            ValueError: The file path is empty or the profile contains Fraction weights.
        """
        if fpath is not None and str(fpath) == "":
            raise ValueError("File path must be provided.")

        if len(self.ballots) == 0:
            raise ProfileError("Cannot write a profile with no ballots to a csv.")

        if self._matrix.voter_weight_data.has_fraction_weights:
            raise ValueError(
                "ScoreProfile CSV does not support rational weights. Convert a copied profile "
                "to float first if a lossy export is acceptable."
            )

        candidate_mapping = {c: str(i) for i, c in enumerate(self.candidates)}

        header = self.__to_score_csv_header(candidate_mapping, include_voter_set)
        data_col_names = self.__to_score_csv_data_column_names(include_voter_set, candidate_mapping)
        ballot_rows = [
            self.__to_score_csv_ballot_row(b, include_voter_set, weight_precision)
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
            fpath (str): Path to csv.

        Raises:
            ValueError: If csv is improperly formatted for VoteKit.
            ProfileError: If read profile has no rankings or scores.

        """
        with open(str(fpath), "r") as file:
            reader = csv.reader(file)
            csv_data = list(reader)

        _validate_score_csv_format(csv_data)

        (
            inv_candidate_mapping,
            includes_voter_set,
            break_indices,
        ) = _parse_profile_data_from_score_csv(csv_data)

        ballots = [
            _parse_ballot_from_score_csv(
                row,
                includes_voter_set,
                break_indices,
                inv_candidate_mapping,
            )
            for row in csv_data[7:]
        ]

        return cls(
            ballots=tuple(ballots),
            candidates=tuple(inv_candidate_mapping.values()),
        )

    @staticmethod
    def _from_matrix(matrix: ScoreMatrix) -> ScoreProfile:
        """
        Create a ScoreProfile from a ScoreMatrix.

        Args:
            matrix (ScoreMatrix): Validated, owned score matrix to share with the new profile.

        Returns:
            ScoreProfile: Ordinary profile with derived metadata and frozen attributes.
        """
        profile = object.__new__(ScoreProfile)
        profile._matrix = matrix
        profile.candidates = matrix.candidates
        profile.candidates_cast = tuple(
            sort_candidates_pseudo_lexicographically(matrix.get_candidates_cast())
        )
        profile._finalize_profile()
        return profile

    def to_matrix(self) -> ScoreMatrix:
        """
        Export the score matrix with independent, read-only array storage.

        Returns:
            ScoreMatrix: New matrix preserving scores, weights, voters, labels, and candidate
                metadata. Immutable candidate names and voter sets may be shared.
        """
        return self._matrix.copy()
