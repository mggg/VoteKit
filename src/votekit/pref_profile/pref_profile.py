from __future__ import annotations
import csv
from fractions import Fraction
import pandas as pd
from pydantic import ConfigDict, field_validator, model_validator
from ..ballot import Ballot
from .utils import _df_to_ballot_tuple
from pydantic.dataclasses import dataclass
from typing_extensions import Self
from dataclasses import field
import numpy as np
from typing import Optional, Tuple
from functools import partial
import warnings
import pickle


@dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class PreferenceProfile:
    """
    PreferenceProfile class, contains ballots and candidates for a given election.
    This is a frozen class, so you need to create a new PreferenceProfile any time
    you want to edit the ballots, candidates, etc.

    Args:
        ballots (tuple[Ballot], optional): Tuple of ``Ballot`` objects. Defaults to empty tuple.
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
        ballots (tuple[Ballot]): Tuple of ``Ballot`` objects.
        candidates (tuple[str]): Tuple of candidate strings.
        max_ranking_length (int): The length of the longest allowable ballot, i.e., how
            many candidates are allowed to be ranked in an election.
        df (pandas.DataFrame): Data frame view of the ballots.
        candidates_cast (tuple[str]): Tuple of candidates who appear on any ballot with positive
            weight, either in the ranking or in the score dictionary.
        total_ballot_wt (Fraction): Sum of ballot weights.
        num_ballots (int): Length of ballot list.
        contains_rankings (bool): Whether or not the profile contains ballots with
            rankings.
        contains_scores (bool): Whether or not the profile contains ballots with
            scores.

    Raises:
        ValueError: contains_rankings is set to False but a ballot contains a ranking.
        ValueError: contains_rankings is set to True but no ballot contains a ranking.
        ValueError: contains_scores is set to False but a ballot contains a score.
        ValueError: contains_scores is set to True but no ballot contains a score.
        ValueError: max_ranking_length is set but a ballot ranking excedes the length.
        ValueError: a candidate is found on a ballot that is not listed on a provided
            candidate list.
        ValueError: candidates must be unique.

    Warns:
        UserWarning: max_ranking_length is set but contains_rankings is False.
            Sets max_ranking_length to 0.

    """

    ballots: tuple[Ballot, ...] = field(default_factory=tuple)
    candidates: tuple[str, ...] = field(default_factory=tuple)
    max_ranking_length: int = 0
    df: pd.DataFrame = field(default_factory=pd.DataFrame)
    candidates_cast: tuple[str, ...] = field(default_factory=tuple)
    num_ballots: int = 0
    total_ballot_wt: Fraction = Fraction(0)
    contains_rankings: Optional[bool] = None
    contains_scores: Optional[bool] = None

    @field_validator("candidates")
    @classmethod
    def cands_must_be_unique(
        cls, candidates: Optional[tuple[str, ...]]
    ) -> Optional[tuple[str, ...]]:
        if candidates:
            if not len(set(candidates)) == len(candidates):
                raise ValueError("All candidates must be unique.")
        return candidates

    def __update_ballot_scores_data(
        self,
        ballot_data: dict[str, list],
        idx: int,
        ballot: Ballot,
        candidates_cast: list[str],
        num_ballots: int,
    ) -> None:
        if ballot.scores:
            if self.contains_scores is False:
                raise ValueError(
                    (
                        f"Ballot {ballot} has scores {ballot.scores} but contains_scores is "
                        "set to False."
                    )
                )

            for c, score in ballot.scores.items():
                if ballot.weight > 0 and c not in candidates_cast:
                    candidates_cast.append(c)

                if c not in ballot_data:
                    if self.candidates:
                        raise ValueError(
                            f"Candidate {c} found in ballot {ballot} but not in "
                            f"candidate list {self.candidates}."
                        )
                    ballot_data[c] = [np.nan] * num_ballots
                ballot_data[c][idx] = score

    def __update_ballot_rankings_data(
        self,
        ballot_data: dict[str, list],
        idx: int,
        ballot: Ballot,
        candidates_cast: list[str],
        num_ballots: int,
    ) -> None:

        if ballot.ranking:
            if self.contains_rankings is False:
                raise ValueError(
                    (
                        f"Ballot {ballot} has ranking {ballot.ranking} but contains_rankings is"
                        " set to False."
                    )
                )

            for j, cand_set in enumerate(ballot.ranking):
                for c in cand_set:
                    if self.candidates:
                        if c not in self.candidates:
                            raise ValueError(
                                f"Candidate {c} found in ballot {ballot} but not in "
                                f"candidate list {self.candidates}."
                            )
                    if ballot.weight > 0 and c not in candidates_cast:
                        candidates_cast.append(c)
                if f"Ranking_{j+1}" not in ballot_data:
                    if self.max_ranking_length:
                        raise ValueError(
                            f"Max ballot length {self.max_ranking_length} given but "
                            "ballot {b} has length at least {j+1}."
                        )
                    ballot_data[f"Ranking_{j+1}"] = [np.nan] * num_ballots

                ballot_data[f"Ranking_{j+1}"][idx] = cand_set

    def __update_ballot_data_attrs(
        self,
        ballot_data: dict[str, list],
        idx: int,
        ballot: Ballot,
        candidates_cast: list[str],
        num_ballots: int,
    ) -> None:
        ballot_data["Weight"][idx] = ballot.weight

        if ballot.voter_set:
            ballot_data["Voter Set"][idx] = ballot.voter_set

        if ballot.scores:
            self.__update_ballot_scores_data(
                ballot_data=ballot_data,
                idx=idx,
                ballot=ballot,
                candidates_cast=candidates_cast,
                num_ballots=num_ballots,
            )

        if ballot.ranking:
            self.__update_ballot_rankings_data(
                ballot_data=ballot_data,
                idx=idx,
                ballot=ballot,
                candidates_cast=candidates_cast,
                num_ballots=num_ballots,
            )

    def __init_ballot_data(self) -> Tuple[int, dict[str, list]]:
        num_ballots = len(self.ballots)

        ballot_data: dict[str, list] = {
            "Weight": [np.nan] * num_ballots,
            "Voter Set": [set()] * num_ballots,
        }

        if self.candidates:
            ballot_data.update({c: [np.nan] * num_ballots for c in self.candidates})

        if self.max_ranking_length:
            ballot_data.update(
                {
                    f"Ranking_{i+1}": [np.nan] * num_ballots
                    for i in range(self.max_ranking_length)
                }
            )
        return num_ballots, ballot_data

    def __init_formatted_df(
        self,
        ballot_data: dict[str, list],
        contains_scores_indicator: bool,
    ):
        df = pd.DataFrame(ballot_data)
        temp_col_order = [c for c in df.columns if "Ranking_" in c] + [
            "Voter Set",
            "Weight",
        ]

        if self.candidates and contains_scores_indicator:
            col_order = list(self.candidates) + temp_col_order
        elif contains_scores_indicator:
            col_order = (
                sorted([c for c in df.columns if c not in temp_col_order])
                + temp_col_order
            )
        else:
            col_order = temp_col_order
        df = df[col_order]
        df.index.name = "Ballot Index"
        return df

    def __set_class_attrs_from_df(
        self,
        df: pd.DataFrame,
        candidates_cast: list[str],
        contains_rankings_indicator: bool,
        contains_scores_indicator: bool,
    ) -> Self:
        object.__setattr__(self, "df", df)
        object.__setattr__(self, "candidates_cast", tuple(candidates_cast))
        if not self.candidates:
            object.__setattr__(self, "candidates", tuple(candidates_cast))

        if self.contains_rankings is None:
            object.__setattr__(self, "contains_rankings", contains_rankings_indicator)
        elif self.contains_rankings and not contains_rankings_indicator:
            raise ValueError(
                "contains_rankings is True but we found no ballots with rankings."
            )

        if self.contains_scores is None:
            object.__setattr__(self, "contains_scores", contains_scores_indicator)
        elif self.contains_scores and not contains_scores_indicator:
            raise ValueError(
                "contains_scores is True but we found no ballots with scores."
            )

        return self

    @model_validator(mode="after")
    def ballot_list_to_df(self) -> Self:
        # `ballot_data` sends {Weight, Voter Set} keys to a list to be
        # indexed in the same order as the output df containing information
        # for each ballot. So ballot_data[<weight>][<index>] is the weight value for
        # the ballot at index <index> in the df.
        num_ballots, ballot_data = self.__init_ballot_data()

        candidates_cast: list[str] = []
        contains_rankings_indicator = False
        contains_scores_indicator = False

        for i, b in enumerate(self.ballots):
            contains_scores_indicator = contains_scores_indicator or (
                b.scores is not None
            )
            contains_rankings_indicator = contains_rankings_indicator or (
                b.ranking is not None
            )

            self.__update_ballot_data_attrs(
                ballot_data=ballot_data,
                idx=i,
                ballot=b,
                candidates_cast=candidates_cast,
                num_ballots=num_ballots,
            )

        df = self.__init_formatted_df(
            ballot_data=ballot_data,
            contains_scores_indicator=contains_scores_indicator,
        )

        return self.__set_class_attrs_from_df(
            df=df,
            candidates_cast=candidates_cast,
            contains_rankings_indicator=contains_rankings_indicator,
            contains_scores_indicator=contains_scores_indicator,
        )

    @model_validator(mode="after")
    def find_max_ranking_length(self) -> Self:
        if self.max_ranking_length > 0 and not self.contains_rankings:
            warnings.warn(
                "Profile does not contain rankings but "
                f"max_ranking_length={self.max_ranking_length}. Setting max_ranking_length"
                " to 0."
            )

            object.__setattr__(self, "max_ranking_length", 0)

        elif not self.max_ranking_length and self.contains_rankings:
            max_ranking_length = len([c for c in self.df.columns if "Ranking_" in c])
            object.__setattr__(self, "max_ranking_length", max_ranking_length)

        return self

    @model_validator(mode="after")
    def find_num_ballots(self) -> Self:
        object.__setattr__(self, "num_ballots", len(self.df))
        return self

    @model_validator(mode="after")
    def find_total_ballot_wt(self) -> Self:
        object.__setattr__(self, "total_ballot_wt", self.df["Weight"].sum())

        return self

    def __add__(self, other):
        """
        Add two PreferenceProfiles by combining their ballot lists.
        """
        if isinstance(other, PreferenceProfile):
            ballots = self.ballots + other.ballots
            max_ranking_length = max(
                [self.max_ranking_length, other.max_ranking_length]
            )
            candidates = set(self.candidates).union(other.candidates)
            return PreferenceProfile(
                ballots=ballots,
                max_ranking_length=max_ranking_length,
                candidates=candidates,
            )

        else:
            raise TypeError(
                "Unsupported operand type. Must be an instance of PreferenceProfile."
            )

    def group_ballots(self) -> PreferenceProfile:
        """
        Groups ballots by rankings and scores and updates weights. Retains voter sets, but
        loses ballot ids.

        Returns:
            PreferenceProfile: A PreferenceProfile object with grouped ballot list.
        """
        non_group_cols = ["Weight", "Voter Set"]
        ranking_cols = [c for c in self.df.columns if "Ranking_" in c]
        cand_cols = [
            c for c in self.df.columns if c not in non_group_cols + ranking_cols
        ]

        group_df = self.df.groupby(cand_cols + ranking_cols, dropna=False)
        new_df = group_df.aggregate(
            {
                "Weight": "sum",
                "Voter Set": (lambda sets: set().union(*sets)),
            }
        ).reset_index()

        new_ballots = _df_to_ballot_tuple(
            new_df,
            candidates=self.candidates,
        )

        return PreferenceProfile(
            ballots=new_ballots,
            candidates=self.candidates,
            max_ranking_length=self.max_ranking_length,
        )

    def __eq__(self, other):
        if not isinstance(other, PreferenceProfile):
            return False
        if set(self.candidates) != set(other.candidates):
            return False
        if set(self.candidates_cast) != set(other.candidates_cast):
            return False
        if self.total_ballot_wt != other.total_ballot_wt:
            return False
        if self.max_ranking_length != other.max_ranking_length:
            return False
        if self.contains_rankings != other.contains_rankings:
            return False
        if self.contains_scores != other.contains_scores:
            return False

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

        repr_str = f"Profile contains rankings: {self.contains_rankings}\n"
        if self.contains_rankings:
            repr_str += f"Maximum ranking length: {self.max_ranking_length}\n"

        repr_str += f"Profile contains scores: {self.contains_scores}\n"

        repr_str += f"Candidates: {self.candidates}\n"
        repr_str += f"Candidates who received votes: {self.candidates_cast}\n"

        repr_str += f"Total number of Ballot objects: {self.num_ballots}\n"
        repr_str += f"Total weight of Ballot objects: {self.total_ballot_wt}\n"

        return repr_str

    __repr__ = __str__

    def to_csv(self, fpath: str):
        """
        Saves PreferenceProfile to CSV.

        Args:
            fpath (str): Path to the saved csv.
        """
        with open(fpath, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)

            rows = [["Candidates"], self.candidates]
            writer.writerows(rows)

        cols = ["Weight", "Voter Set"]
        if self.contains_rankings:
            cols = [c for c in self.df.columns if "Ranking_" in c] + cols
        if self.contains_scores:
            cols = [c for c in self.df.columns if c in self.candidates] + cols

        self.df[cols].to_csv(
            path_or_buf=fpath,
            mode="a",
            encoding="utf-8",
        )

    # Peter, think we can rig this so that it "absorbs" the load_cvr function?
    # this csv is very specially formatted to deal with data types, so maybe not?
    @classmethod
    def from_csv(cls, fpath: str) -> PreferenceProfile:
        """
        Creates a PreferenceProfile from a csv, formatted from the ``to_csv`` method.
        """
        with open(fpath, "r") as file:
            reader = csv.reader(file)
            candidates = tuple(list(reader)[1])

        df = pd.read_csv(fpath, skiprows=[0, 1])
        dtype = {c: "float64" for c in candidates if c in df.columns}
        df.astype(dtype)

        ranking_cols = [c for c in df.columns if "Ranking_" in c]

        def _str_to_fraction(s: str) -> Fraction:
            if "/" in s:
                numerator, denominator = [int(x) for x in s.split("/")]
                return Fraction(numerator, denominator)

            return Fraction(int(s))

        if df.dtypes["Weight"] == "object":
            df["Weight"] = df["Weight"].apply(_str_to_fraction)  # type: ignore

        def _str_to_set(s: str | float, frozen: bool) -> frozenset | float | set:
            if pd.isna(s):
                return np.nan
            elif s == "frozenset()" or s == "set()":
                return frozenset() if frozen else set()

            strip_str = "frozenset({})" if frozen else "{}"
            if isinstance(s, str):
                s = s.strip(strip_str)
                contents = [c.strip("'") for c in s.split(", ")]

            return frozenset(contents) if frozen else set(contents)

        for c in ranking_cols:
            df[c] = df[c].apply(partial(_str_to_set, frozen=True))  # type: ignore

        df["Voter Set"] = df["Voter Set"].apply(partial(_str_to_set, frozen=False))  # type: ignore

        return cls(
            ballots=_df_to_ballot_tuple(df, candidates=candidates),
            candidates=candidates,
        )

    def to_pickle(self, fpath: str):
        """
        Saves profile to pickle file.

        Args:
            fpath (str): File path to save profile to.
        """

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
