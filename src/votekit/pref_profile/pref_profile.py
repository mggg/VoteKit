from __future__ import annotations

import csv
import io
import os
import pickle
import re
import urllib.request
import warnings
from functools import cached_property
from os import PathLike
from pathlib import Path
from typing import Optional, Self, Sequence, Tuple, Union, cast, overload

import numpy as np
import pandas as pd

from votekit._df_schema import (
    BALLOT_INDEX_NAME,
    RANKING_COLUMN_PREFIX,
    VOTER_SET_COLUMN,
    WEIGHT_COLUMN,
)
from votekit.ballot import Ballot, RankBallot, ScoreBallot
from votekit.exceptions import ProfileError
from votekit.pref_profile.csv_utils import (
    _parse_ballot_from_rank_csv,
    _parse_ballot_from_score_csv,
    _parse_profile_data_from_rank_csv,
    _parse_profile_data_from_score_csv,
    _validate_rank_csv_format,
    _validate_score_csv_format,
)
from votekit.pref_profile.profile_matrix import TILDE_SET, ProfileMatrix, RankMatrix, ScoreMatrix
from votekit.pref_profile.utils import (
    _sum_rank_profiles,
    _sum_score_profiles,
    convert_row_to_rank_ballot,
    convert_row_to_score_ballot,
)
from votekit.types import Candidate, Numeric
from votekit.utils import (
    _validate_candidate_names,
    sort_candidates_pseudo_lexicographically,
)


def _infer_profile_type(ballots: Sequence[Ballot], df: pd.DataFrame) -> type[PreferenceProfile]:
    """
    Select the profile type for a direct PreferenceProfile construction.

    A dataframe with a ranking-prefixed column selects RankProfile; other dataframes
    select ScoreProfile. Without a dataframe, all ballots must have the same type.
    Empty inputs retain the base PreferenceProfile type.

    Args:
        ballots (Sequence[Ballot]): Ballots used when no dataframe is supplied.
        df (pd.DataFrame): Input dataframe, or a default empty dataframe when absent.

    Returns:
        type[PreferenceProfile]: The class to allocate and initialize.

    Raises:
        ProfileError: The ballot sequence does not consist entirely of rank or score ballots.
    """
    if not df.equals(pd.DataFrame()):
        if any(c.startswith(RANKING_COLUMN_PREFIX) for c in df.columns if isinstance(c, str)):
            return RankProfile
        return ScoreProfile

    if len(ballots) == 0:
        return PreferenceProfile
    if all(isinstance(ballot, RankBallot) for ballot in ballots):
        return RankProfile
    if all(isinstance(ballot, ScoreBallot) for ballot in ballots):
        return ScoreProfile

    score_count = sum(isinstance(ballot, ScoreBallot) for ballot in ballots)
    rank_count = sum(isinstance(ballot, RankBallot) for ballot in ballots)
    raise ProfileError(
        "Profile cannot contain RankBallots and ScoreBallots. There are "
        f"{score_count} ScoreBallots and {rank_count} RankBallots."
    )


class PreferenceProfile:
    """
    PreferenceProfile class, contains ballots and candidates for a given election.
    This is a frozen class, so you need to create a new PreferenceProfile any time
    you want to edit the ballots, candidates, etc.

    Calling PreferenceProfile directly infers RankProfile or ScoreProfile from the inputs.
    With no ballots or dataframe, it creates an empty PreferenceProfile. Explicit calls to
    RankProfile, ScoreProfile, or their subclasses preserve the requested type and reject
    incompatible ballot types.

    Args:
        ballots (Sequence[Ballot], optional): Tuple of ``Ballot`` objects. Defaults to empty tuple.
        candidates (tuple[Candidate], optional): Tuple of candidates.
            Candidate can be a str or int. Defaults to empty tuple.
            If empty, computes this from any candidate listed on a ballot with positive weight.
            Candidates can be strings, integers, or mix of both.
        max_ranking_length (int, optional): The length of the longest allowable ballot, i.e., how
            many candidates are allowed to be ranked in an election. Defaults to longest observed
            ballot.
        df (pd.DataFrame, optional): Input ballot dataframe. Defaults to an empty dataframe.

    Parameters:
        ballots (Sequence[Ballot]): Tuple of ``Ballot`` objects.
        candidates (tuple[Candidate]): Tuple of candidates.
            Candidates can be strings, integers, or mix of both.
        max_ranking_length (int): The length of the longest allowable ballot, i.e., how
            many candidates are allowed to be ranked in an election.
        df (pandas.DataFrame): Detached dataframe export of the ballots. Each access creates
            a new dataframe; editing it does not change the profile.
        candidates_cast (tuple[Candidate]): Tuple of candidates who appear on any ballot with
            positive weight, either in the ranking or in the score dictionary.
            Candidates can be strings, integers, or mix of both.
        total_ballot_wt (Numeric): Sum of ballot weights.
        num_ballots (int): Length of ballot list.

    Raises:
        ProfileError: a data frame and ballot list are passed to the init method.
        ProfileError: max_ranking_length is set but a ballot ranking excedes the length.
        ProfileError: a candidate is found on a ballot that is not listed on a provided
            candidate list.
        ProfileError: candidates must be unique.
        ProfileError: candidates must not have names matching ranking columns.

    """

    _is_frozen: bool = False
    _matrix: ProfileMatrix | None = None

    @overload
    def __new__(
        cls,
        *,
        ballots: Sequence[RankBallot],
        candidates: Sequence[Candidate] = tuple(),
        max_ranking_length: int = 0,
        df: pd.DataFrame = pd.DataFrame(),
        **kwargs,
    ) -> RankProfile: ...

    @overload
    def __new__(
        cls,
        *,
        ballots: Sequence[ScoreBallot],
        candidates: Sequence[Candidate] = tuple(),
        max_ranking_length: int = 0,
        df: pd.DataFrame = pd.DataFrame(),
        **kwargs,
    ) -> ScoreProfile: ...

    @overload
    def __new__(
        cls,
        *,
        ballots: Sequence[Ballot] = tuple(),
        candidates: Sequence[Candidate] = tuple(),
        max_ranking_length: int = 0,
        df: pd.DataFrame = pd.DataFrame(),
        **kwargs,
    ) -> Self: ...

    def __new__(
        cls,
        *,
        ballots: Sequence[Ballot] = tuple(),
        candidates: Sequence[Candidate] = tuple(),
        max_ranking_length: int = 0,
        df: pd.DataFrame = pd.DataFrame(),
        **kwargs,
    ) -> PreferenceProfile:
        if not df.equals(pd.DataFrame()) and len(ballots) != 0:
            raise ProfileError(
                "Cannot pass a dataframe and a ballot list to profile init method. Must pick one."
            )

        if cls is not PreferenceProfile:
            return super().__new__(cls)

        profile_type = _infer_profile_type(ballots, df)
        return object.__new__(profile_type)

    def __init__(
        self,
        *,
        ballots: Sequence[Ballot] = tuple(),
        candidates_cast: Sequence[Candidate] = tuple(),
        candidates: Sequence[Candidate] = tuple(),
        max_ranking_length: Optional[int] = None,
        df: pd.DataFrame = pd.DataFrame(),
        include_zero_score: Optional[bool] = None,
    ):
        self.candidates_cast = _normalize_unique_candidates(
            candidates_cast, f"{type(self).__name__}.candidates_cast"
        )
        self.candidates = _normalize_unique_candidates(
            candidates, f"{type(self).__name__}.candidates"
        )
        self.max_ranking_length = max_ranking_length

        self._validate_candidates()
        self._finalize_profile()

    def _finalize_profile(self) -> None:
        """
        Compute matrix-derived metadata and freeze the initialized profile.
        """
        self.total_ballot_wt = self._find_total_ballot_wt()
        self.num_ballots = self._find_num_ballots()
        self._is_frozen = True

    def to_df(self) -> pd.DataFrame:
        """
        Export a detached dataframe; changes to it do not affect the profile.

        Each call constructs a fresh dataframe from the authoritative matrix. Retain the returned
        dataframe locally when performing several pandas operations on the same export.

        Returns:
            pd.DataFrame: Ballot data and index labels with independent array storage.
                Immutable candidate and voter sets may be shared with the matrix.
        """
        if self._matrix is None:
            raise NotImplementedError
        return self._matrix.to_df()

    @property
    def df(self) -> pd.DataFrame:
        """
        Deprecated alias for to_df(), returning a fresh dataframe on each access.

        Deprecated as of VoteKit 3.6.0 and scheduled for removal in the next major release.
        Use to_df() to make dataframe construction explicit, and retain its result locally
        when performing several pandas operations on the same export.

        Returns:
            pd.DataFrame: Detached snapshot of the profile's ballot data and index labels.
        """
        warnings.warn(
            "PreferenceProfile.df is deprecated as of VoteKit 3.6.0 and will be removed "
            "in the next major release. Use to_df() instead: profiles now construct a fresh "
            "dataframe on each call. Save the result locally when reusing it.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.to_df()

    def _find_num_ballots(self) -> int:
        """
        Compute and set the number of ballots.

        Returns:
            int: num ballots
        """
        return len(self._matrix) if self._matrix is not None else 0

    def _find_total_ballot_wt(self) -> Numeric:
        """
        Sum the weights stored in the authoritative matrix.

        Returns:
            Numeric: Total ballot weight, or zero when no matrix is attached.
        """
        if self._matrix is None:
            return 0

        # Object-array weights are validated Fraction values by construction.
        return cast(Numeric, np.sum(self._matrix.voter_weight_data.weights))

    def _validate_candidates(self) -> None:
        """
        Check reserved ranking-column names and membership after candidate normalization.

        Raises:
            ProfileError: A candidate name is reserved or candidates cast are undeclared.
        """
        for candidate in self.candidates:
            if isinstance(candidate, str) and re.fullmatch(
                rf"{re.escape(RANKING_COLUMN_PREFIX)}\d+", candidate
            ):
                raise ProfileError(
                    f"{candidate} is a name reserved for ranking columns, it cannot be used as a "
                    "candidate name."
                )

        if not set(self.candidates_cast).issubset(self.candidates):
            raise ProfileError(
                "Candidates cast are not a subset of candidates list. The following "
                " candidates are in candidates_cast but not candidates: "
                f"{set(self.candidates_cast) - set(self.candidates)}."
            )

    def __setattr__(self, name, value):
        if getattr(self, "_is_frozen", False):
            raise AttributeError(f"Cannot modify frozen instance: tried to set '{name}'")
        super().__setattr__(name, value)

    def __delattr__(self, name):
        if getattr(self, "_is_frozen", False):
            raise AttributeError(f"Cannot delete attribute '{name}' from frozen instance")
        super().__delattr__(name)

    def __eq__(self, other):
        if not isinstance(other, PreferenceProfile):
            return False
        if set(self.candidates) != set(other.candidates):
            return False
        if set(self.candidates_cast) != set(other.candidates_cast):
            return False
        if self.total_ballot_wt != other.total_ballot_wt:
            return False

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pp_1 = self.group_ballots()
            pp_2 = other.group_ballots()
        for b in pp_1.ballots:
            if b not in pp_2.ballots:
                return False
        for b in pp_2.ballots:
            if b not in pp_1.ballots:
                return False
        return True

    def __str__(self) -> str:
        repr_str = "PrefenceProfile"
        repr_str += (
            f"Candidates: {self.candidates}\n"
            f"Candidates who received votes: {self.candidates_cast}\n"
            f"Total number of Ballot objects: {self.num_ballots}\n"
            f"Total weight of Ballot objects: {self.total_ballot_wt}\n"
        )

        return repr_str

    __repr__ = __str__

    def group_ballots(self) -> Self:
        raise NotImplementedError

    def copy(self) -> Self:
        raise NotImplementedError

    @property
    def ballots(self) -> tuple[Ballot, ...]:
        raise NotImplementedError

    def to_pickle(self, fpath: Union[str, PathLike, Path]):
        """
        Saves profile to pickle file.

        Args:
            fpath (Union[str, os.PathLike, pathlib.Path]): File path to save profile to.

        Raises:
            ValueError: File path must be provided.
        """
        if fpath == "":
            raise ValueError("File path must be provided.")
        with open(str(fpath), "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_pickle(cls, fpath: Union[str, PathLike, Path]) -> PreferenceProfile:
        """
        Reads profile from pickle file.

        Args:
            fpath (Union[str, os.PathLike, pathlib.Path]): File path to profile. Can be a url.
        """
        fpath = str(fpath)
        if not os.path.isfile(fpath):
            with urllib.request.urlopen(fpath) as response:
                data = pickle.loads(response.read())
        else:
            with open(str(fpath), "rb") as f:
                data = pickle.load(f)
        assert isinstance(data, PreferenceProfile)
        return data


def _normalize_candidate_name(candidate: Candidate) -> Candidate:
    """
    Normalize a candidate name by stripping whitespace if it's a string.
    """
    return candidate.strip() if isinstance(candidate, str) else candidate


def _normalize_unique_candidates(
    candidates: Sequence[Candidate], context: str = "candidates"
) -> tuple[Candidate, ...]:
    """
    Normalize candidate names while preserving order and rejecting collisions.

    Args:
        candidates (Sequence[Candidate]): Candidate names to normalize.
        context (str, optional): Source name used in validation errors. Defaults to "candidates".

    Returns:
        tuple[Candidate, ...]: Normalized candidate names.

    Raises:
        ProfileError: Two entries have the same identity after normalization.
        TypeError: Candidates is a bare string or a candidate has an unsupported type.
        ValueError: A normalized name contains a prohibited value or character.
    """
    if isinstance(candidates, str):
        raise TypeError("Candidates cannot be a string. Wrap string in list.")

    normalized_candidates = tuple(_normalize_candidate_name(c) for c in candidates)
    _validate_candidate_names(normalized_candidates, context)

    if len(set(normalized_candidates)) != len(normalized_candidates):
        raise ProfileError(
            "All candidates must be unique. "
            "Two or more candidates have the same identity after stripping whitespace."
        )

    return normalized_candidates


def _normalize_rank_df_candidates(df: pd.DataFrame, context: str = "ranking cells") -> pd.DataFrame:
    """
    Normalize candidate names in ranking cells without modifying the supplied dataframe.

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

    def normalize_ranking_set(candidate_set: frozenset[Candidate]) -> frozenset[Candidate]:
        if not isinstance(candidate_set, (set, frozenset)):
            raise TypeError("Ranking cells must contain sets or frozensets of candidates.")
        if candidate_set == TILDE_SET:
            return TILDE_SET

        normalized = tuple(_normalize_candidate_name(candidate) for candidate in candidate_set)
        candidate_names.update(((type(candidate), candidate), None) for candidate in normalized)
        return frozenset(normalized)

    normalized_df[ranking_cols] = df[ranking_cols].map(normalize_ranking_set)
    _validate_candidate_names([candidate for _, candidate in candidate_names], context)
    return normalized_df


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
                rank_ballot_data[f"{RANKING_COLUMN_PREFIX}{j + 1}"] = [TILDE_SET] * num_ballots
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
                    f"{RANKING_COLUMN_PREFIX}{i + 1}": [TILDE_SET] * num_ballots
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
            return len(self._matrix.ranking_columns)

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

    @cached_property
    def ballots(self: RankProfile) -> tuple[RankBallot, ...]:
        """
        Compute the ballot tuple as a cached property.
        """
        assert self.max_ranking_length is not None

        return tuple(
            convert_row_to_rank_ballot(row, self.max_ranking_length)
            for _, row in self.to_df().iterrows()
        )

    def __add__(self, other) -> RankProfile:
        """
        Add two PreferenceProfiles by combining their ballot lists.
        """
        return _sum_rank_profiles([self, other])

    def group_ballots(self) -> RankProfile:
        """
        Groups ballots by rankings and updates weights. Retains voter sets, but
        loses ballot indices.

        Returns:
            RankProfile: A RankProfile object with grouped ballot list.
        """
        if self.num_ballots == 0:
            return RankProfile(
                candidates=self.candidates,
                max_ranking_length=self.max_ranking_length,
            )

        df = self.to_df()
        ranking_cols = list(self._matrix.ranking_columns)
        group_df = df.groupby(ranking_cols, dropna=False)
        new_df = group_df.aggregate(
            {
                WEIGHT_COLUMN: "sum",
                VOTER_SET_COLUMN: (lambda sets: set().union(*sets)),
            }
        ).reset_index()

        new_df.index.name = BALLOT_INDEX_NAME

        return RankProfile(
            df=new_df,
            candidates=self.candidates,
            max_ranking_length=self.max_ranking_length,
        )

    def copy(self) -> RankProfile:
        """
        Returns a copy of a RankProfile

        Returns:
            RankProfile: New RankProfile object
        """
        return RankProfile(
            candidates=self.candidates,
            df=self.to_df(),
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
        data_col_names = [f"{RANKING_COLUMN_PREFIX}{i + 1}" for i in range(self.max_ranking_length)]
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
        Compute the ballot tuple as a cached property.
        """
        return tuple(
            convert_row_to_score_ballot(row, tuple(self.candidates))
            for _, row in self.to_df().iterrows()
        )

    def __add__(self, other):
        """
        Add two PreferenceProfiles by combining their ballot lists.
        """
        return _sum_score_profiles([self, other])

    def group_ballots(self) -> ScoreProfile:
        """
        Groups ballots by scores and updates weights. Retains voter sets, but
        loses ballot ids.

        Returns:
            ScoreProfile: A ScoreProfile object with grouped ballot list.
        """
        if self.num_ballots == 0:
            return ScoreProfile(
                candidates=self.candidates,
            )

        df = self.to_df()
        cand_cols = list(self._matrix.score_columns)
        group_df = df.groupby(cand_cols, dropna=False)
        new_df = group_df.aggregate(
            {
                WEIGHT_COLUMN: "sum",
                VOTER_SET_COLUMN: (lambda sets: set().union(*sets)),
            }
        ).reset_index()

        new_df.index.name = BALLOT_INDEX_NAME

        return ScoreProfile(
            df=new_df,
            candidates=self.candidates,
        )

    def copy(self) -> ScoreProfile:
        """
        Returns a copy of a ScoreProfile

        Returns:
            ScoreProfile: New ScoreProfile object
        """
        return ScoreProfile(
            df=self.to_df(),
            candidates=self.candidates,
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
