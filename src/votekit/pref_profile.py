from __future__ import annotations
import csv
from fractions import Fraction
import pandas as pd
from pydantic import ConfigDict, field_validator, model_validator
from .ballot import Ballot
from pydantic.dataclasses import dataclass
from typing_extensions import Self
from dataclasses import field
import numpy as np
from typing import Optional
from functools import partial
import warnings
import pickle

def _convert_ranking_cols_to_ranking(
    row: pd.Series, ranking_cols: list[str]
) -> Optional[tuple[frozenset, ...]]:
    """
    Convert the ranking cols to a ranking tuple.

    """
    ranking = []
    for i, col in enumerate(ranking_cols):
        if pd.isna(row[col]):
            if not all(pd.isna(row[c]) for c in ranking_cols[i:]):
                raise ValueError(
                    f"Row {row} has NaN values between valid ranking positions. "
                    "NaN values can only trail on a ranking."
                )

            break

        ranking.append(row[col])

    return tuple(ranking) if ranking else None


def _convert_row_to_ballot(
    row: pd.Series,
    ranking_cols: list[str],
    weight_col: str,
    id_col: str,
    voter_set_col: str,
    candidates: list[str],
) -> Ballot:
    """
    Convert a row of a properly formatted df to a Ballot.
    """
    ranking = _convert_ranking_cols_to_ranking(row, ranking_cols)
    scores = {c: row[c] for c in candidates if c in row and not pd.isna(row[c])}
    id = row[id_col] if not pd.isna(row[id_col]) else None
    voter_set = row[voter_set_col]
    weight = row[weight_col]

    return Ballot(
        ranking=ranking,
        scores=scores if scores else None,
        weight=weight,
        id=id,
        voter_set=voter_set,
    )


def _df_to_ballot_tuple(
    df: pd.DataFrame,
    candidates: list[str],
    ranking_cols: list[str] = [],
    weight_col: str = "Weight",
    id_col: str = "ID",
    voter_set_col: str = "Voter Set",
) -> tuple[Ballot]:
    """
    Convert a df into a list of ballots.
    """
    return tuple(
        df.apply(
            partial(
                _convert_row_to_ballot,
                ranking_cols=ranking_cols,
                weight_col=weight_col,
                id_col=id_col,
                candidates=candidates,
                voter_set_col=voter_set_col,
            ),
            axis="columns",
        )
    )


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
            rankings. If not provided, will set itself to correct value given the ballot list.
        contains_scores (bool, optional): Whether or not the profile contains ballots with
            scores. If not provided, will set itself to correct value given the ballot list.

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

    @model_validator(mode="after")
    def ballot_list_to_df(self) -> Self:
        num_ballots = len(self.ballots)

        ballot_data = {
            "Weight": [np.nan] * num_ballots,
            "ID": [np.nan] * num_ballots,
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

        candidates_cast = []
        contains_rankings_indicator = False
        contains_scores_indicator = False

        for i, b in enumerate(self.ballots):
            ballot_data["Weight"][i] = b.weight

            if b.id:
                ballot_data["ID"][i] = b.id
            if b.voter_set:
                ballot_data["Voter Set"][i] = b.voter_set

            if b.scores:
                if self.contains_scores is False:
                    raise ValueError(
                        (
                            f"Ballot {b} has scores {b.scores} but contains_scores is "
                            "set to False."
                        )
                    )
                contains_scores_indicator = True

                for c, score in b.scores.items():
                    if b.weight > 0 and c not in candidates_cast:
                        candidates_cast.append(c)

                    if c not in ballot_data:
                        if self.candidates:
                            raise ValueError(
                                f"Candidate {c} found in ballot {b} but not in "
                                f"candidate list {self.candidates}."
                            )
                        ballot_data[c] = [np.nan] * num_ballots
                    ballot_data[c][i] = score

            if b.ranking:
                if self.contains_rankings is False:
                    raise ValueError(
                        (
                            f"Ballot {b} has ranking {b.ranking} but contains_rankings is"
                            " set to False."
                        )
                    )
                contains_rankings_indicator = True

                for j, cand_set in enumerate(b.ranking):
                    for c in cand_set:
                        if self.candidates:
                            if c not in self.candidates:
                                raise ValueError(
                                    f"Candidate {c} found in ballot {b} but not in "
                                    f"candidate list {self.candidates}."
                                )
                        if b.weight > 0 and c not in candidates_cast:
                            candidates_cast.append(c)
                    if f"Ranking_{j+1}" not in ballot_data:
                        if self.max_ranking_length:
                            raise ValueError(
                                f"Max ballot length {self.max_ranking_length} given but "
                                "ballot {b} has length at least {j+1}."
                            )
                        ballot_data[f"Ranking_{j+1}"] = [np.nan] * num_ballots

                    ballot_data[f"Ranking_{j+1}"][i] = cand_set

        df = pd.DataFrame(ballot_data)
        temp_col_order = [c for c in df.columns if "Ranking" in c] + [
            "Weight",
            "ID",
            "Voter Set",
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
    def find_max_ranking_length(self) -> Self:
        if self.max_ranking_length >0 and not self.contains_rankings:
            warnings.warn("Profile does not contain rankings but "
                          f"max_ranking_length={self.max_ranking_length}. Setting max_ranking_length"
                          " to 0.")
            
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
            max_ranking_length = max([self.max_ranking_length, other.max_ranking_length])
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
        non_group_cols = ["Weight", "ID", "Voter Set"]
        ranking_cols = [c for c in self.df.columns if "Ranking_" in c]
        cand_cols = [
            c for c in self.df.columns if c not in non_group_cols + ranking_cols
        ]

        group_df = self.df.groupby(cand_cols + ranking_cols, dropna=False)
        new_df = group_df.aggregate(
            {
                "Weight": "sum",
                "Voter Set": (lambda sets: set().union(*sets)),
                "ID": lambda x: x.iloc[0] if len(x) == 1 else np.nan,
            }
        ).reset_index()

        new_ballots = _df_to_ballot_tuple(
            new_df, candidates=self.candidates, ranking_cols=ranking_cols
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
            repr_str+=f"Maximum ranking length: {self.max_ranking_length}\n"

        repr_str+=f"Profile contains scores: {self.contains_scores}\n"
        
        repr_str+=f"Candidates: {self.candidates}\n"
        repr_str+=f"Candidates who received votes: {self.candidates_cast}\n"

        repr_str+=f"Total number of Ballot objects: {self.num_ballots}\n"
        repr_str+=f"Total weight of Ballot objects: {self.total_ballot_wt}\n"

        return repr_str

    __repr__ = __str__

    # def to_ballot_dict(self, standardize: bool = False) -> dict[Ballot, Fraction]:
    #     """
    #     Converts profile to dictionary with keys = ballots and
    #     values = corresponding total weights.

    #     Args:
    #         standardize (bool, optional): If True, divides the weight of each ballot by the total
    #             weight. Defaults to False.

    #     Returns:
    #         dict[Ballot, Fraction]:
    #             A dictionary with ballots (keys) and corresponding total weights (values).
    #     """
    #     tot_weight = self.total_ballot_wt
    #     di: dict = {}
    #     for ballot in self.ballots:
    #         weightless_ballot = Ballot(ranking=ballot.ranking, scores=ballot.scores, id =ballot.id, voter_set=ballot.voter_set)
    #         weight = ballot.weight
    #         if standardize:
    #             weight/= tot_weight

    #         if weightless_ballot not in di.keys():
    #             di[weightless_ballot] = weight
    #         else:
    #             di[weightless_ballot] += weight
    #     return di

    # def to_ranking_dict(
    #     self, standardize: bool = False
    # ) -> dict[tuple[frozenset[str], ...], Fraction]:
    #     """
    #     Converts profile to dictionary with keys = rankings and
    #     values = corresponding total weights.

    #     Args:
    #         standardize (bool, optional): If True, divides the weight of each ballot by the total
    #             weight. Defaults to False.

    #     Returns:
    #         dict[tuple[frozenset[str],...], Fraction]:
    #             A dictionary with rankings (keys) and corresponding total weights (values).
    #     """

    #     if not self.contains_rankings:
    #         warnings.warn(("You are trying to convert a profile that contains no rankings to a "
    #                        "ranking_dict."))
    #     tot_weight = self.total_ballot_wt
    #     di: dict = {}
    #     for ballot in self.ballots:
    #         ranking = ballot.ranking
    #         weight = ballot.weight
    #         if standardize:
    #              weight/= tot_weight

    #         if ranking not in di.keys():
    #             di[ranking] = weight
    #         else:
    #             di[ranking] += weight
    #     return di

    # def to_scores_dict(
    #     self, standardize: bool = False
    # ) -> dict[tuple[str, Fraction], Fraction]:
    #     """
    #     Converts profile to dictionary with keys = scores and
    #     values = corresponding total weights.

    #     Args:
    #         standardize (bool, optional): If True, divides the weight of each ballot by the total
    #             weight. Defaults to False.

    #     Returns:
    #         dict[tuple[str, Fraction], Fraction]:
    #             A dictionary with scores (keys) and corresponding total weights (values).
    #     """
    #     if not self.contains_scores:
    #         warnings.warn(("You are trying to convert a profile that contains no scores to a "
    #                        "scores_dict."))

    #     tot_weight = self.total_ballot_wt
    #     di: dict = {}
    #     for ballot in self.ballots:
    #         scores = tuple(ballot.scores.items()) if ballot.scores else None
    #         weight = ballot.weight 
    #         if standardize:
    #             weight /= tot_weight

    #         if scores not in di.keys():
    #             di[scores] = weight
    #         else:
    #             di[scores] += weight
    #     return di


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
        

        cols = ["Weight", "ID", "Voter Set"]
        if self.contains_rankings:
            cols = [c for c in self.df.columns if "Ranking_" in c] + cols
        if self.contains_scores:
            cols = [c for c in self.df.columns if c in self.candidates] + cols

        self.df[cols].to_csv(path_or_buf=fpath, mode='a', encoding="utf-8",)
    
    @classmethod
    def from_csv(cls, fpath:str)-> PreferenceProfile:
        """
        Creates a PreferenceProfile from a csv, formatted from the ``to_csv`` method.
        """
        with open(fpath, 'r') as file:
            reader = csv.reader(file)
            candidates =list(reader)[1]

        df = pd.read_csv(fpath, skiprows=[0,1])
        dtype = {c: 'float64' for c in candidates if c in df.columns}
        dtype.update({"ID": 'str'})
        df.astype(dtype)

        ranking_cols = [c for c in df.columns if "Ranking_" in c]

        def _str_to_fraction(s: str) -> Fraction:
            if "/" in s:
                numerator, denominator = [int(x) for x in s.split("/")]
                return Fraction(numerator, denominator)

            return Fraction(int(s))

        if df.dtypes["Weight"] == "object":
            df["Weight"] = df["Weight"].apply(_str_to_fraction)

        def _str_to_set(s: str | float, frozen: bool )-> frozenset | float | set:
            if pd.isna(s):
                return np.nan
            elif s == "frozenset()" or s=="set()":
                return frozenset() if frozen else set()

            strip_str = "frozenset({})" if frozen else "{}"
            s = s.strip(strip_str)
            contents = [c.strip("'") for c in s.split(", ")]

            return frozenset(contents) if frozen else set(contents)
        
        for c in ranking_cols:
            df[c] = df[c].apply(partial(_str_to_set, frozen = True))

        df["Voter Set"] = df["Voter Set"].apply(partial(_str_to_set, frozen = False))
        
        return cls(ballots = _df_to_ballot_tuple(df, candidates=candidates, ranking_cols=ranking_cols), candidates = candidates)
    
    def to_pickle(self, fpath:str):
        """
        Saves profile to pickle file.

        Args:
            fpath (str): File path to save profile to.
        """

        with open(fpath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_pickle(cls, fpath:str)-> PreferenceProfile:
        """
        Reads profile from pickle file.

        Args:
            fpath (str): File path to profile.
        """

        with open(fpath, "rb") as f:
            data = pickle.load(f)
        assert isinstance(data, PreferenceProfile)
        return data
        
    # move these ideas to another module, about preferenceprofile summary stats
    # move all print stuff 
    # def head(
    #     self,
    #     n: int,
    #     sort_by_weight: Optional[bool] = True,
    #     percents: Optional[bool] = False,
    #     totals: Optional[bool] = False,
    # ) -> pd.DataFrame:
    #     """
    #     Displays top-n ballots in profile.

    #     Args:
    #         n (int): Number of ballots to view.
    #         sort_by_weight (bool, optional): If True, rank ballot from most to least votes.
    #             Defaults to True.
    #         percents (bool, optional): If True, show voter share for a given ballot.
    #             Defaults to False.
    #         totals (bool, optional): If True, show total values for Percent and Weight.
    #             Defaults to False.

    #     Returns:
    #         pandas.DataFrame: A dataframe with top-n ballots.
    #     """
    #     if sort_by_weight:
    #         df = (
    #             self.df.sort_values(by="Weight", ascending=False)
    #             .head(n)
    #             .reset_index(drop=True)
    #         )
    #     else:
    #         df = self.df.head(n).reset_index(drop=True)

    #     if totals:
    #         df = self._sum_row(df)

    #     if not percents:
    #         return df.drop(columns="Percent")

    #     return df

    # def tail(
    #     self,
    #     n: int,
    #     sort_by_weight: Optional[bool] = True,
    #     percents: Optional[bool] = False,
    #     totals: Optional[bool] = False,
    # ) -> pd.DataFrame:
    #     """
    #     Displays bottom-n ballots in profile.

    #     Args:
    #         n (int): Number of ballots to view.
    #         sort_by_weight (bool, optional): If True, rank ballot from least to most votes.
    #             Defaults to True.
    #         percents (bool, optional): If True, show voter share for a given ballot.
    #             Defaults to False.
    #         totals (bool, optional): If True, show total values for Percent and Weight.
    #             Defaults to False.

    #     Returns:
    #         pandas.DataFrame: A data frame with bottom-n ballots.
    #     """
    #     if sort_by_weight:
    #         df = self.df.sort_values(by="Weight", ascending=True)
    #         df["New Index"] = [x for x in range(len(self.df) - 1, -1, -1)]
    #         df = df.set_index("New Index").head(n)
    #         df.index.name = None

    #     else:
    #         df = self.df.iloc[::-1].head(n)

    #     if totals:
    #         df = self._sum_row(df)

    #     if not percents:
    #         return df.drop(columns="Percent")

    #     return df

    # def _sum_row(self, df: pd.DataFrame) -> pd.DataFrame:
    #     """
    #     Computes sum total for weight and percent column
    #     """

    #     def format_as_float(percent_str):
    #         return float(percent_str.split("%")[0])

    #     sum_row = {
    #         "Ranking": "",
    #         "Scores": "",
    #         "Weight": f'{df["Weight"].sum()} out of {self.total_ballot_wt}',
    #         "Percent": f'{df["Percent"].apply(format_as_float).sum():.2f} out of 100%',
    #     }

    #     df.loc["Totals"] = sum_row  # type: ignore

    #     return df.fillna("")
