from __future__ import annotations
import csv
import pandas as pd
from ..ballot import Ballot, ScoreBallot, RankBallot
from .utils import convert_row_to_rank_ballot
from .csv_utils import (
    _validate_rank_csv_format,
    _parse_profile_data_from_rank_csv,
    _parse_ballot_from_rank_csv,
)
import numpy as np
from typing import Optional, Tuple, Sequence, Union
import warnings
import pickle
from .profile_error import ProfileError
from functools import cached_property
from pathlib import Path
from os import PathLike


class PreferenceProfile:
    """
    PreferenceProfile class, contains ballots and candidates for a given election.
    This is a frozen class, so you need to create a new PreferenceProfile any time
    you want to edit the ballots, candidates, etc.

    Args:
        ballots (Sequence[Ballot], optional): Tuple of ``Ballot`` objects. Defaults to empty tuple.
        candidates (tuple[str], optional): Tuple of candidate strings. Defaults to empty tuple.
            If empty, computes this from any candidate listed on a ballot with positive weight.
        max_ranking_length (int, optional): The length of the longest allowable ballot, i.e., how
            many candidates are allowed to be ranked in an election. Defaults to longest observed
            ballot.
        contains_rankings (bool, optional): Whether or not the profile contains ballots with
            rankings. If no boolean is provided, then the appropriate boolean value will be
            interpreted from the input preference profile (i.e. if some ballot in the profile has
            a ranking, then this will be set  to `True`).
        contains_scores (bool, optional): Whether or not the profile contains ballots with
            scores. If no boolean is provided, then the appropriate boolean value will be
            interpreted from the input preference profile (i.e. if some ballot in the profile has
            a ranking, then this will be set  to `True`).

    Parameters:
        ballots (Sequence[Ballot]): Tuple of ``Ballot`` objects.
        candidates (tuple[str]): Tuple of candidate strings.
        max_ranking_length (int): The length of the longest allowable ballot, i.e., how
            many candidates are allowed to be ranked in an election.
        df (pandas.DataFrame): Data frame view of the ballots.
        candidates_cast (tuple[str]): Tuple of candidates who appear on any ballot with positive
            weight, either in the ranking or in the score dictionary.
        total_ballot_wt (float): Sum of ballot weights.
        num_ballots (int): Length of ballot list.
        contains_rankings (bool): Whether or not the profile contains ballots with
            rankings.
        contains_scores (bool): Whether or not the profile contains ballots with
            scores.

    Raises:
        ProfileError: a data frame and ballot list are passed to the init method.
        ProfileError: contains_rankings is set to False but a ballot contains a ranking.
        ProfileError: contains_rankings is set to True but no ballot contains a ranking.
        ProfileError: contains_scores is set to False but a ballot contains a score.
        ProfileError: contains_scores is set to True but no ballot contains a score.
        ProfileError: max_ranking_length is set but a ballot ranking excedes the length.
        ProfileError: a candidate is found on a ballot that is not listed on a provided
            candidate list.
        ProfileError: candidates must be unique.
        ProfileError: candidates must not have names matching ranking columns.

    """

    _is_frozen: bool = False

    def __new__(
        cls,
        *,
        ballots: Sequence[Ballot] = tuple(),
        candidates: Sequence[str] = tuple(),
        max_ranking_length: int = 0,
        df: pd.DataFrame = pd.DataFrame(),
    ):

        if not df.equals(pd.DataFrame()) and ballots != tuple():
            raise ProfileError(
                "Cannot pass a dataframe and a ballot list to profile init method. Must pick one."
            )

        elif df.equals(pd.DataFrame()):
            if all(isinstance(b, RankBallot) for b in ballots):
                return super().__new__(RankProfile)

            elif all(isinstance(b, ScoreBallot) for b in ballots):
                return super().__new__(ScoreProfile)

            else:
                score_idxs = [
                    idx for idx, b in enumerate(ballots) if isinstance(b, ScoreBallot)
                ]
                rank_idxs = [
                    idx for idx, b in enumerate(ballots) if isinstance(b, RankBallot)
                ]

                raise ProfileError(
                    "Profile cannot contain RankBallots and ScoreBallots. ScoreBallots"
                    f" appear at indices {score_idxs}, RankBallots appear at indices"
                    f" {rank_idxs}."
                )

        if any(c.startswith("Ranking_") for c in df.columns):
            return super().__new__(RankProfile)

        return super().__new__(ScoreProfile)

    def __init__(
        self,
        *,
        ballots: Sequence[Ballot] = tuple(),
        candidates: Sequence[str] = tuple(),
        max_ranking_length: Optional[int] = None,
        df: pd.DataFrame = pd.DataFrame(),
    ):
        self.total_ballot_wt = self._find_total_ballot_wt()
        self.num_ballots = self._find_num_ballots()

        self._is_frozen = True

    def _find_num_ballots(self) -> int:
        """
        Compute and set the number of ballots.

        Returns:
            int: num ballots
        """
        return len(self.df)

    def _find_total_ballot_wt(self) -> float:
        """
        Compute and set the total ballot weight.

        Returns:
            float: total ballot weight.
        """
        total_weight = 0
        try:
            total_weight = self.df["Weight"].sum()
        except KeyError:
            pass
        return total_weight

    def _validate_and_set_candidates(self) -> None:
        """
        Ensure that the candidate names are not equal to the ranking column names, that they are
        unique, and strips whitespace from candidates.

        Raises:
            ProfileError: Candidate names must not be the same as "Ranking_i".
            ProfileError: Candidate names must be unique.
        """
        if not len(set(self.candidates)) == len(self.candidates):
            raise ProfileError("All candidates must be unique.")

        if not set(self.candidates_cast).issubset(self.candidates):
            raise ValueError(
                "Candidates cast are not a subset of candidates list. The following "
                " candidates are in candidates_cast but not candidates: "
                f"{set(self.candidates_cast)-set(self.candidates)}."
            )

        self.candidates = tuple([c.strip() for c in self.candidates])
        self.candidates_cast = tuple([c.strip() for c in self.candidates_cast])

    def __setattr__(self, name, value):
        if getattr(self, "_is_frozen", False):
            raise AttributeError(
                f"Cannot modify frozen instance: tried to set '{name}'"
            )
        super().__setattr__(name, value)

    def __delattr__(self, name):
        if getattr(self, "_is_frozen", False):
            raise AttributeError(
                f"Cannot delete attribute '{name}' from frozen instance"
            )
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

    def to_pickle(self, fpath: str):
        """
        Saves profile to pickle file.

        Args:
            fpath (str): File path to save profile to.

        Raises:
            ValueError: File path must be provided.
        """
        if fpath == "":
            raise ValueError("File path must be provided.")
        with open(fpath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_pickle(cls, fpath: str) -> PreferenceProfile:
        """
        Reads profile from pickle file.

        Args:
            fpath (str): File path to profile.
        """

        with open(fpath, "rb") as f:
            data = pickle.load(f)
        assert isinstance(data, PreferenceProfile)
        return data


class RankProfile(PreferenceProfile):

    def __init__(
        self,
        *,
        ballots: Sequence[Ballot] = tuple(),
        candidates: Sequence[str] = tuple(),
        max_ranking_length: Optional[int] = None,
        df: pd.DataFrame = pd.DataFrame(),
    ):

        self.candidates = tuple(candidates)
        self.max_ranking_length = (
            0 if max_ranking_length is None else max_ranking_length
        )

        if df.equals(pd.DataFrame()):
            (
                self.df,
                self.candidates_cast,
            ) = self._init_from_rank_ballots(ballots)
            if self.candidates == tuple():
                self.candidates = self.candidates_cast

        else:
            self.df, self.candidates_cast = self._init_from_rank_df(df)

        self.max_ranking_length = self._find_max_ranking_length()
        self._validate_and_set_candidates_from_rankings()

        super().__init__()

    def __update_ballot_ranking_data(
        self,
        rank_ballot_data: dict[str, list],
        idx: int,
        rank_ballot: RankBallot,
        candidates_cast: list[str],
        num_ballots: int,
    ):
        """
        Update the ranking data from a ballot.

        Args:
            rank_ballot_data (dict[str, list]): Dictionary storing ballot data.
            idx (int): Index of ballot.
            rank_ballot (RankBallot): Ballot.
            candidates_cast (list[str]): List of candidates who have received votes.
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
            if f"Ranking_{j+1}" not in rank_ballot_data:
                if self.max_ranking_length > 0:
                    raise ProfileError(
                        f"Max ranking length {self.max_ranking_length} given but "
                        f"ballot {rank_ballot} has length at least {j+1}."
                    )
                rank_ballot_data[f"Ranking_{j+1}"] = [frozenset("~")] * num_ballots

            rank_ballot_data[f"Ranking_{j+1}"][idx] = cand_set

    def __update_rank_ballot_data_attrs(
        self,
        rank_ballot_data: dict[str, list],
        idx: int,
        rank_ballot: RankBallot,
        candidates_cast: list[str],
        num_ballots: int,
    ):
        """
        Update ballot data from a rank ballot.

        Args:
            rank_ballot_data (dict[str, list]): Dictionary storing ballot data.
            idx (int): Index of ballot.
            rank_ballot (RankBallot): Ballot.
            candidates_cast (list[str]): List of candidates who have received votes.
            num_ballots (int): Total number of ballots.
        """
        rank_ballot_data["Weight"][idx] = rank_ballot.weight

        if rank_ballot.voter_set != frozenset():
            rank_ballot_data["Voter Set"][idx] = rank_ballot.voter_set

        if rank_ballot.ranking is not None:
            self.__update_ballot_ranking_data(
                rank_ballot_data=rank_ballot_data,
                idx=idx,
                rank_ballot=rank_ballot,
                candidates_cast=candidates_cast,
                num_ballots=num_ballots,
            )

    def __init_rank_ballot_data(
        self, rank_ballots: Sequence[RankBallot]
    ) -> Tuple[int, dict[str, list]]:
        """
        Create the ballot data objects.

        Args:
            rank_ballots (Sequence[RankBallot,...]): Tuple of ballots.

        Returns:
            Tuple[int, dict[str, list]]: num_ballots, rank_ballot_data

        """
        num_ballots = len(rank_ballots)

        rank_ballot_data: dict[str, list] = {
            "Weight": [np.nan] * num_ballots,
            "Voter Set": [set()] * num_ballots,
        }

        if self.max_ranking_length > 0:
            rank_ballot_data.update(
                {
                    f"Ranking_{i+1}": [frozenset("~")] * num_ballots
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
        col_order = [c for c in df.columns if "Ranking_" in c] + [
            "Voter Set",
            "Weight",
        ]

        df = df[col_order]
        df.index.name = "Ballot Index"
        return df

    def _init_from_rank_ballots(
        self, ballots: Sequence[RankBallot]
    ) -> tuple[pd.DataFrame, tuple[str, ...]]:
        """
        Create the pandas dataframe representation of the profile.

        Args:
            ballots (Sequence[RankBallot,...]): Sequence of ballots.

        Returns:
            tuple[pd.DataFrame, tuple[str, ...]]: df, candidates_cast

        """
        # `rank_ballot_data` sends {Weight, Voter Set} keys to a list to be
        # indexed in the same order as the output df containing information
        # for each ballot. So rank_ballot_data[<weight>][<index>] is the weight value for
        # the ballot at index <index> in the df.
        num_ballots, rank_ballot_data = self.__init_rank_ballot_data(ballots)

        candidates_cast: list[str] = []

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

        return (
            df,
            tuple(candidates_cast),
        )

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
        boiler_plate = (
            "When providing a dataframe and no ballot list to the init method, "
        )
        if len(df) == 0:
            return

        if self.max_ranking_length == 0:
            raise ProfileError(
                boiler_plate + "max_ranking_length must be provided and be non-zero."
            )

        if self.candidates == tuple():
            raise ProfileError(boiler_plate + "candidates must be provided.")

    def __validate_init_rank_df(self, df: pd.DataFrame) -> None:
        """
        Validate that the df passed to the init method is of valid type.

        Args:
            df (pd.DataFrame): Dataframe representation of ballots.

        Raises:
            ProfileError: Ranking column is missing.
            ProfileError: Weight column is missing.
            ProfileError: Voter set column is missing.
            ProfileError: Index column is misformatted.

        """
        if "Weight" not in df.columns:
            raise ProfileError(f"Weight column not in dataframe: {df.columns}")
        if "Voter Set" not in df.columns:
            raise ProfileError(f"Voter Set column not in dataframe: {df.columns}")
        if df.index.name != "Ballot Index":
            raise ProfileError(f"Index not named 'Ballot Index': {df.index.name}")
        if any(
            f"Ranking_{i+1}" not in df.columns for i in range(self.max_ranking_length)
        ):
            for i in range(self.max_ranking_length):
                if f"Ranking_{i+1}" not in df.columns:
                    raise ProfileError(
                        f"Ranking column 'Ranking_{i+1}' not in dataframe: {df.columns}"
                    )

    def __find_candidates_cast_from_init_rank_df(
        self, df: pd.DataFrame
    ) -> tuple[str, ...]:
        """
        Compute which candidates received votes from the df and set the candidates_cast and
        candidates attr.

        Args:
            df (pd.DataFrame): Dataframe representation of ballots.

        Returns:
            tuple[str]: Candidates cast.
        """

        mask = df["Weight"] > 0

        candidates_cast: set[str] = set()

        ranking_cols = [c for c in df.columns if c.startswith("Ranking_")]
        sets = df.loc[mask, ranking_cols].to_numpy().ravel()
        candidates_cast |= set().union(*sets)

        candidates_cast.discard("~")
        return tuple(candidates_cast)

    def _init_from_rank_df(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, tuple[str, ...]]:
        """
        Validate the dataframe and determine the candidates cast.

        Args:
            df (pd.DataFrame): Dataframe representation of ballots.

        Returns
            tuple[pd.DataFrame, tuple[str]]: df, candidates_cast
        """
        self.__validate_init_rank_df_params(df)
        self.__validate_init_rank_df(df)
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
        if self.max_ranking_length == 0:
            return len([c for c in self.df.columns if "Ranking_" in c])

        return self.max_ranking_length

    def _validate_and_set_candidates_from_rankings(self) -> None:
        """
        Ensure that the candidate names are not equal to the ranking column names, that they are
        unique, and strips whitespace from candidates.

        Raises:
            ProfileError: Candidate names must not be the same as "Ranking_i".
            ProfileError: Candidate names must be unique.
        """
        for cand in self.candidates:
            if any(f"Ranking_{i}" == cand for i in range(len(self.candidates))):
                raise ProfileError(
                    (
                        f"Candidate {cand} must not share name with"
                        " ranking columns: Ranking_i."
                    )
                )

        super()._validate_and_set_candidates()

    @cached_property
    def ballots(self: PreferenceProfile) -> tuple[RankBallot, ...]:
        """
        Compute the ballot tuple as a cached property.
        """

        # TODO do this with map?
        computed_ballots = [RankBallot()] * len(self.df)
        for i, (_, b_row) in enumerate(self.df.iterrows()):
            computed_ballots[i] = convert_row_to_rank_ballot(
                b_row,
                self.max_ranking_length,
            )
        return tuple(computed_ballots)

    def __add__(self, other) -> RankProfile:
        """
        Add two PreferenceProfiles by combining their ballot lists.
        """
        # TODO do with df ?
        if isinstance(other, RankProfile):
            ballots = self.ballots + other.ballots
            max_ranking_length = max(
                [self.max_ranking_length, other.max_ranking_length]
            )
            candidates = set(self.candidates).union(other.candidates)
            return RankProfile(
                ballots=ballots,
                max_ranking_length=max_ranking_length,
                candidates=candidates,
            )

        raise TypeError("Unsupported operand type. Must be an instance of RankProfile.")

    def group_ballots(self) -> RankProfile:
        """
        Groups ballots by rankings and updates weights. Retains voter sets, but
        loses ballot indices.

        Returns:
            RankProfile: A RankProfile object with grouped ballot list.
        """
        empty_df = pd.DataFrame(columns=["Voter Set", "Weight"], dtype=np.float64)
        empty_df.index.name = "Ballot Index"

        if len(self.df) == 0:
            return RankProfile(
                candidates=self.candidates,
                max_ranking_length=self.max_ranking_length,
            )

        ranking_cols = [c for c in self.df.columns if "Ranking_" in c]
        group_df = self.df.groupby(ranking_cols, dropna=False)
        new_df = group_df.aggregate(
            {
                "Weight": "sum",
                "Voter Set": (lambda sets: set().union(*sets)),
            }
        ).reset_index()

        new_df.index.name = "Ballot Index"

        return RankProfile(
            df=new_df,
            candidates=self.candidates,
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
        self, candidate_mapping: dict[str, str], include_voter_set: bool
    ) -> list[list]:
        """
        Construct the header rows for the PrefProfile a custom CSV format.

        Args:
            include_voter_set (bool): Whether or not to include the voter set of each
                ballot.
        """
        header = [
            ["VoteKit RankProfile"],
            ["Candidates"],
            [f"({c}:{cand_label})" for c, cand_label in candidate_mapping.items()],
        ]
        header += [["Max Ranking Length"], [str(self.max_ranking_length)]]
        header += [["Includes Voter Set"], [str(include_voter_set)]]
        header += [["="] * 10]

        return header

    def __to_rank_csv_ranking_list(
        self, rank_ballot: RankBallot, candidate_mapping: dict[str, str]
    ) -> list:
        """
        Create the list of ranking data for a ballot in the profile.

        Args:
            rank_ballot (RankBallot): Ballot.
            candidate_mapping (dict[str, int]): Mapping candidate names to integers.

        """
        if rank_ballot.ranking is not None:
            ranking_list = [
                (
                    set([candidate_mapping[c] for c in cand_set])
                    if cand_set != frozenset()
                    else "{}"
                )
                for cand_set in rank_ballot.ranking
            ]
            if len(ranking_list) != self.max_ranking_length:
                ranking_list += [""] * (self.max_ranking_length - len(ranking_list))

            return ranking_list

        return [""] * self.max_ranking_length

    def __to_rank_csv_ballot_row(
        self,
        ballot: Ballot,
        include_voter_set: bool,
        candidate_mapping: dict[str, str],
        weight_precision: int,
    ) -> list[list]:
        """
        Create the row for a ballot in the profile.

        Args:
            ballot (Ballot): Ballot.
            include_voter_set (bool): Whether or not to include the voter set of each
                ballot.
            candidate_mapping (dict[str, int]): Mapping candidate names to integers.
            weight_precision (int): Number of decimals to round float weights to. Defaults to 2.

        """
        row = self.__to_rank_csv_ranking_list(ballot, candidate_mapping)
        row += ["&"]

        row += [round(ballot.weight, weight_precision), "&"]

        if include_voter_set:
            row += [v for v in sorted(ballot.voter_set)]

        return row

    def __to_rank_csv_data_column_names(
        self, include_voter_set: bool, candidate_mapping: dict[str, str]
    ) -> list:
        """
        Create the data column header.

        Args:
            include_voter_set (bool): Whether or not to include the voter set of each
                ballot.
            candidate_mapping (dict[str, str]): Maps candidate names to prefixes.
        """
        data_col_names = [f"Ranking_{i+1}" for i in range(self.max_ranking_length)]
        data_col_names += ["&", "Weight", "&"]

        if include_voter_set:
            data_col_names += ["Voter Set"]

        return data_col_names

    def to_csv(
        self,
        fpath: Union[str, PathLike, Path],
        include_voter_set: bool = False,
        weight_precision: int = 2,
    ):
        """
        Saves PreferenceProfile to a custom CSV format.

        Args:
            fpath (Union[str, PathLike, Path]): Path to the saved csv.
            include_voter_set (bool, optional): Whether or not to include the voter set of each
                ballot. Defaults to False.
            weight_precision (int): Number of decimals to round float weights to. Defaults to 2.
        Raises:
            ProfileError: Cannot write a profile with no ballots to a csv.
            ValueError: File path must be provided.
        """
        if fpath == "":
            raise ValueError("File path must be provided.")

        if len(self.ballots) == 0:
            raise ProfileError("Cannot write a profile with no ballots to a csv.")

        prefix_idx = 1
        candidate_mapping = {c: c[:prefix_idx] for c in self.candidates}
        while len(set(candidate_mapping.values())) < len(candidate_mapping.values()):
            prefix_idx += 1
            candidate_mapping = {c: c[:prefix_idx] for c in self.candidates}

        header = self.__to_rank_csv_header(candidate_mapping, include_voter_set)
        data_col_names = self.__to_rank_csv_data_column_names(
            include_voter_set, candidate_mapping
        )
        ballot_rows = [
            self.__to_rank_csv_ballot_row(
                b, include_voter_set, candidate_mapping, weight_precision
            )
            for b in self.ballots
        ]
        rows = header + [data_col_names] + ballot_rows

        with open(
            fpath,
            "w",
            newline="",
            encoding="utf-8",
        ) as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(rows)

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

    def __init__():
        pass
