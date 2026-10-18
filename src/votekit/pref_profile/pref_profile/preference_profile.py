from __future__ import annotations

import os
import pickle
import re
import urllib.request
import warnings
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Self, Sequence, Union, cast, overload

import numpy as np
import pandas as pd

from votekit._df_schema import RANKING_COLUMN_PREFIX
from votekit.ballot import Ballot, RankBallot, ScoreBallot
from votekit.exceptions import ProfileError
from votekit.pref_profile.pref_profile._candidate_names import _normalize_unique_candidates
from votekit.pref_profile.profile_matrix import ProfileMatrix
from votekit.types import Candidate, Numeric

if TYPE_CHECKING:
    from votekit.pref_profile.pref_profile.rank_profile import RankProfile
    from votekit.pref_profile.pref_profile.score_profile import ScoreProfile


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
    from votekit.pref_profile.pref_profile.rank_profile import RankProfile
    from votekit.pref_profile.pref_profile.score_profile import ScoreProfile

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
