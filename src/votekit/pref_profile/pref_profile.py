from __future__ import annotations
import csv
from fractions import Fraction
import pandas as pd
from pydantic import ConfigDict, field_validator, model_validator
from ..ballot import Ballot
from pydantic.dataclasses import dataclass
from typing_extensions import Self
from dataclasses import field
import numpy as np


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
        max_ballot_length (int, optional): The length of the longest allowable ballot, i.e., how
            many candidates are allowed to be ranked in an election. Defaults to longest observed
            ballot.

    Parameters:
        ballots (tuple[Ballot]): Tuple of ``Ballot`` objects.
        candidates (tuple[str]): Tuple of candidate strings.
        max_ballot_length (int): The length of the longest allowable ballot, i.e., how
            many candidates are allowed to be ranked in an election.
        df (pandas.DataFrame): Data frame view of the ballots.
        candidates_cast (tuple[str]): Tuple of candidates who appear on any ballot with positive
            weight, either in the ranking or in the score dictionary.
        total_ballot_wt (Fraction): Sum of ballot weights.
        num_ballots (int): Length of ballot list.

    """

    ballots: tuple[Ballot, ...] = field(default_factory=tuple)
    candidates: tuple[str, ...] = field(default_factory=tuple)
    max_ballot_length: int = 0
    df: pd.DataFrame = field(default_factory=pd.DataFrame)
    candidates_cast: tuple[str, ...] = field(default_factory=tuple)
    num_ballots: int = 0
    total_ballot_wt: Fraction = Fraction(0)

    # to and from csv, different type options for scoring, ranking, both
    # encoding specification

    @model_validator(mode="after")
    def ballot_list_to_df(self) -> Self:
        """
        After init, convert ballot list to df.
        """

        num_ballots = len(self.ballots)

        ballot_data = {
            "weight": [np.nan] * num_ballots,
            "id": [np.nan] * num_ballots,
            "voter_set": [np.nan] * num_ballots,
        }

        if self.candidates:
            ballot_data.update({c: [np.nan] * num_ballots for c in self.candidates})

        if self.max_ballot_length:
            ballot_data.update(
                {
                    f"ranking_{i}": [np.nan] * num_ballots
                    for i in range(self.max_ballot_length)
                }
            )

        for i, b in enumerate(self.ballots):
            ballot_data["weight"][i] = b.weight

            if b.id:
                ballot_data["id"][i] = b.id
            if b.voter_set:
                ballot_data["voter_set"][i] = b.voter_set

            if b.scores:
                for c, score in b.scores.items():
                    if c not in ballot_data:
                        if self.candidates:
                            raise ValueError(
                                f"Candidate {c} found in ballot {b} but not in candidate list {self.candidates}."
                            )
                        ballot_data[c] = [np.nan] * num_ballots
                    ballot_data[c][i] = score

            if b.ranking:
                for j, cand_set in enumerate(b.ranking):
                    if f"ranking_{j}" not in ballot_data:
                        if self.max_ballot_length:
                            raise ValueError(
                                f"Max ballot length {self.max_ballot_length} given but ballot {b} has length at least {j}."
                            )
                        ballot_data[f"ranking_{j}"] = [np.nan] * num_ballots

                    ballot_data[f"ranking_{j}"][i] = cand_set

        df = pd.DataFrame(ballot_data)
        temp_col_order = [c for c in df.columns if "ranking" in c] + [
            "weight",
            "id",
            "voter_set",
        ]
        col_order = (
            self.candidates + temp_col_order
            if self.candidates
            else sorted([c for c in df.columns if c not in temp_col_order])
            + temp_col_order
        )
        df = df[col_order]
        df.index.name = "Ballot Index"

        object.__setattr__(self, "df", df)
        return self

    @field_validator("candidates")
    @classmethod
    def cands_must_be_unique(
        cls, candidates: Optional[tuple[str, ...]]
    ) -> Optional[tuple[str, ...]]:
        # TODO think about how to do this with DF operations
        # or how to validate unique candidates given a ballot list or df

        # if candidates:
        #     if not len(set(candidates)) == len(candidates):
        #         raise ValueError("All candidates must be unique.")
        return candidates

    @model_validator(mode="after")
    def find_candidates_cast(self) -> Self:
        pass
        # TODO do this using df operations

        # candidates_cast: set = set()
        # for ballot in self.ballots:
        #     if ballot.weight > 0:
        #         if ballot.ranking:
        #             candidates_cast.update(*ballot.ranking)
        #         if ballot.scores:
        #             candidates_cast.update(ballot.scores.keys())

        # object.__setattr__(self, "candidates_cast", tuple(candidates_cast))
        # if not self.candidates:
        #     object.__setattr__(self, "candidates", tuple(candidates_cast))
        return self

    @model_validator(mode="after")
    def check_ballot_length(self) -> Self:
        # TODO rethink where this happens in the logic, maybe only run if given
        # a ballot list
        # if self.max_ballot_length == 0:
        #     for ballot in self.ballots:
        #         ballot_len = len(ballot.ranking) if ballot.ranking else 0

        #         if ballot_len > self.max_ballot_length:
        #             object.__setattr__(self, "max_ballot_length", ballot_len)

        # else:
        #     for ballot in self.ballots:
        #         if ballot.ranking:
        #             if len(ballot.ranking) > self.max_ballot_length:
        #                 raise ValueError(
        #                     "Profile contains a ballot with a ranking that is too long."
        #                 )

        return self

    @model_validator(mode="after")
    def find_num_ballots(self) -> Self:
        # TODO do this with df operations
        object.__setattr__(self, "num_ballots", len(self.ballots))
        return self

    @model_validator(mode="after")
    def find_total_ballot_wt(self) -> Self:
        # TODO do this with df operations
        total_ballot_wt = Fraction(0)
        for ballot in self.ballots:
            total_ballot_wt += ballot.weight

        object.__setattr__(self, "total_ballot_wt", total_ballot_wt)

        return self

    @model_validator(mode="after")
    def create_df(self) -> Self:
        """
        Convert a list of ballots into a pandas dataframe.

        """
        weights = []
        rankings = []
        scores: list[tuple[str, ...]] = []
        for ballot in self.ballots:
            part = []
            if ballot.ranking:
                for ranking in ballot.ranking:
                    if len(ranking) == 1:
                        part.append(list(ranking)[0])

                    else:
                        part.append(f"{set(ranking)} (Tie)")
            score_list: list[str] = []
            if ballot.scores:
                for c, score in ballot.scores.items():
                    score_list.append(f"{c}:{float(score):.2f}")

            rankings.append(tuple(part))
            weights.append(ballot.weight)
            scores.append(tuple(score_list))

        df = pd.DataFrame({"Ranking": rankings, "Scores": scores, "Weight": weights})

        try:
            df["Percent"] = df["Weight"] / df["Weight"].sum()
        except ZeroDivisionError:
            df["Percent"] = 0.0

        def format_as_percent(frac):
            return f"{float(frac):.2%}"

        df["Percent"] = df["Percent"].apply(format_as_percent)
        object.__setattr__(self, "df", df)
        return self

    def to_ballot_dict(self, standardize: bool = False) -> dict[Ballot, Fraction]:
        """
        Converts profile to dictionary with keys = ballots and
        values = corresponding total weights.

        Args:
            standardize (bool, optional): If True, divides the weight of each ballot by the total
                weight. Defaults to False.

        Returns:
            dict[Ballot, Fraction]:
                A dictionary with ballots (keys) and corresponding total weights (values).
        """
        tot_weight = self.total_ballot_wt
        di: dict = {}
        for ballot in self.ballots:
            weightless_ballot = Ballot(ranking=ballot.ranking, scores=ballot.scores)
            if standardize:
                weight = ballot.weight / tot_weight
            else:
                weight = ballot.weight
            if weightless_ballot not in di.keys():
                di[weightless_ballot] = weight
            else:
                di[weightless_ballot] += weight
        return di

    def to_ranking_dict(
        self, standardize: bool = False
    ) -> dict[tuple[frozenset[str], ...], Fraction]:
        """
        Converts profile to dictionary with keys = rankings and
        values = corresponding total weights.

        Args:
            standardize (bool, optional): If True, divides the weight of each ballot by the total
                weight. Defaults to False.

        Returns:
            dict[tuple[frozenset[str],...], Fraction]:
                A dictionary with rankings (keys) and corresponding total weights (values).
        """
        tot_weight = self.total_ballot_wt
        di: dict = {}
        for ballot in self.ballots:
            ranking = ballot.ranking

            if not ranking:
                ranking = (frozenset(),)
            if standardize:
                weight = ballot.weight / tot_weight
            else:
                weight = ballot.weight

            if ranking not in di.keys():
                di[ranking] = weight
            else:
                di[ranking] += weight
        return di

    def to_scores_dict(
        self, standardize: bool = False
    ) -> dict[tuple[str, Fraction], Fraction]:
        """
        Converts profile to dictionary with keys = scores and
        values = corresponding total weights.

        Args:
            standardize (bool, optional): If True, divides the weight of each ballot by the total
                weight. Defaults to False.

        Returns:
            dict[tuple[str, Fraction], Fraction]:
                A dictionary with scores (keys) and corresponding total weights (values).
        """
        tot_weight = self.total_ballot_wt
        di: dict = {}
        for ballot in self.ballots:
            if ballot.scores:
                scores = tuple([(c, score) for c, score in ballot.scores.items()])
            else:
                scores = tuple()
            if standardize:
                weight = ballot.weight / tot_weight
            else:
                weight = ballot.weight

            if scores not in di.keys():
                di[scores] = weight
            else:
                di[scores] += weight
        return di

    def to_csv(self, fpath: str):
        """
        Saves PreferenceProfile to CSV.

        Args:
            fpath (str): Path to the saved csv.
        """
        with open(fpath, "w", newline="") as csvfile:
            fieldnames = ["weight", "ranking", "scores"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for ballot in self.ballots:
                if ballot.ranking:
                    ranking = tuple([set(s) for s in ballot.ranking])
                else:
                    ranking = tuple()

                if ballot.scores:
                    scores = tuple(
                        [(c, float(score)) for c, score in ballot.scores.items()]
                    )
                else:
                    scores = tuple()
                writer.writerow(
                    {
                        "weight": float(ballot.weight),
                        "ranking": ranking,
                        "scores": scores,
                    }
                )

    def head(
        self,
        n: int,
        sort_by_weight: Optional[bool] = True,
        percents: Optional[bool] = False,
        totals: Optional[bool] = False,
    ) -> pd.DataFrame:
        """
        Displays top-n ballots in profile.

        Args:
            n (int): Number of ballots to view.
            sort_by_weight (bool, optional): If True, rank ballot from most to least votes.
                Defaults to True.
            percents (bool, optional): If True, show voter share for a given ballot.
                Defaults to False.
            totals (bool, optional): If True, show total values for Percent and Weight.
                Defaults to False.

        Returns:
            pandas.DataFrame: A dataframe with top-n ballots.
        """
        if sort_by_weight:
            df = (
                self.df.sort_values(by="Weight", ascending=False)
                .head(n)
                .reset_index(drop=True)
            )
        else:
            df = self.df.head(n).reset_index(drop=True)

        if totals:
            df = self._sum_row(df)

        if not percents:
            return df.drop(columns="Percent")

        return df

    def tail(
        self,
        n: int,
        sort_by_weight: Optional[bool] = True,
        percents: Optional[bool] = False,
        totals: Optional[bool] = False,
    ) -> pd.DataFrame:
        """
        Displays bottom-n ballots in profile.

        Args:
            n (int): Number of ballots to view.
            sort_by_weight (bool, optional): If True, rank ballot from least to most votes.
                Defaults to True.
            percents (bool, optional): If True, show voter share for a given ballot.
                Defaults to False.
            totals (bool, optional): If True, show total values for Percent and Weight.
                Defaults to False.

        Returns:
            pandas.DataFrame: A data frame with bottom-n ballots.
        """
        if sort_by_weight:
            df = self.df.sort_values(by="Weight", ascending=True)
            df["New Index"] = [x for x in range(len(self.df) - 1, -1, -1)]
            df = df.set_index("New Index").head(n)
            df.index.name = None

        else:
            df = self.df.iloc[::-1].head(n)

        if totals:
            df = self._sum_row(df)

        if not percents:
            return df.drop(columns="Percent")

        return df

    def __str__(self) -> str:
        # Displays top 15 cast ballots or entire profile

        if len(self.df) < 15:
            return self.head(n=len(self.df), sort_by_weight=True).to_string(
                index=False, justify="justify"
            )

        print(
            f"PreferenceProfile too long, only showing 15 out of {len(self.df) } rows."
        )
        return self.head(n=15, sort_by_weight=True).to_string(
            index=False, justify="justify"
        )

    # set repr to print outputs
    __repr__ = __str__

    def group_ballots(self) -> PreferenceProfile:
        """
        Groups ballots by rankings and scores and updates weights. Retains voter sets, but
        loses ballot ids.

        Returns:
            PreferenceProfile: A PreferenceProfile object with grouped ballot list.
        """

        seen_ballots = {}

        for ballot in self.ballots:
            weightless_ballot = Ballot(
                ranking=ballot.ranking, scores=ballot.scores, weight=Fraction(0)
            )

            if weightless_ballot not in seen_ballots:
                seen_ballots[weightless_ballot] = {
                    "weight": ballot.weight,
                    "voter_set": ballot.voter_set,
                }

            else:
                seen_ballots[weightless_ballot]["weight"] += ballot.weight  # type: ignore[operator]
                seen_ballots[weightless_ballot]["voter_set"].update(
                    ballot.voter_set
                )  # type: ignore[attr-defined]

        new_ballots = [Ballot()] * len(seen_ballots)

        for i, (ballot, ballot_dict) in enumerate(seen_ballots.items()):
            new_ballots[i] = Ballot(
                ranking=ballot.ranking,
                scores=ballot.scores,
                weight=ballot_dict["weight"],  # type: ignore[arg-type]
                voter_set=ballot_dict["voter_set"],  # type: ignore[arg-type]
            )

        return PreferenceProfile(
            ballots=tuple(new_ballots),
            candidates=self.candidates,
            max_ballot_length=self.max_ballot_length,
        )

    def __eq__(self, other):
        # TODO this should check the df, the max ballot length, the cands, etc
        if not isinstance(other, PreferenceProfile):
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

    def _sum_row(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes sum total for weight and percent column
        """

        def format_as_float(percent_str):
            return float(percent_str.split("%")[0])

        sum_row = {
            "Ranking": "",
            "Scores": "",
            "Weight": f'{df["Weight"].sum()} out of {self.total_ballot_wt}',
            "Percent": f'{df["Percent"].apply(format_as_float).sum():.2f} out of 100%',
        }

        df.loc["Totals"] = sum_row  # type: ignore

        return df.fillna("")

    def __add__(self, other):
        """
        Add two PreferenceProfiles by combining their ballot lists.
        """
        if isinstance(other, PreferenceProfile):
            ballots = self.ballots + other.ballots
            pp = PreferenceProfile(ballots=ballots)
            pp.group_ballots()
            return pp
        else:
            raise TypeError(
                "Unsupported operand type. Must be an instance of PreferenceProfile."
            )
