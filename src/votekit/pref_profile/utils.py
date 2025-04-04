from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pref_profile import PreferenceProfile

from ..ballot import Ballot
from fractions import Fraction
from typing import Optional
import pandas as pd
from functools import partial


def _convert_ranking_cols_to_ranking(
    row: pd.Series
) -> Optional[tuple[frozenset, ...]]:
    """
    Convert the ranking cols to a ranking tuple in profile.df.

    Args:
        row (pd.Series): Row of a profile.df.

    Returns:
        Optional[tuple[frozenset, ...]]: Ranking of ballot.

    """
    ranking = []
    ranking_cols = [c for c in row.index if "Ranking_" in c]
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


def convert_row_to_ballot(
    row: pd.Series,
    candidates: tuple[str, ...],
) -> Ballot:
    """
    Convert a row of a properly formatted profile.df to a Ballot.

    Args:
        row (pd.Series): Row of a profile.df.
        candidates (tuple[str,...]): The name of the candidates.

    Returns:
        Ballot: Ballot corresponding to the row of the df.
    """
    ranking = _convert_ranking_cols_to_ranking(row)
    scores = {c: row[c] for c in candidates if c in row and not pd.isna(row[c])}
    id = row["ID"] if not pd.isna(row["ID"]) else None
    voter_set = row["Voter Set"]
    weight = row["Weight"]

    return Ballot(
        ranking=ranking,
        scores=scores if scores else None,
        weight=weight,
        id=id,
        voter_set=voter_set,
    )


def _df_to_ballot_tuple(
    df: pd.DataFrame,
    candidates: tuple[str, ...],
) -> tuple[Ballot]:
    """
    Convert a properly formatted profile.df into a list of ballots.

    Args:
        df (pd.DataFrame): A profile.df.
        candidates (tuple[str,...]): The candidates.

    Returns:
        tuple[Ballot]: The tuple of ballots.
    """
    if df.empty:
        return tuple()
    
    return tuple(
        df.apply(
            partial(
                convert_row_to_ballot,
                candidates=candidates,
            ),
            axis="columns",
        )
    )


def profile_to_ballot_dict(
    profile: PreferenceProfile, standardize: bool = False
) -> dict[Ballot, Fraction]:
    """
    Converts profile to dictionary with keys = ballots and
    values = corresponding total weights.

    Args:
        profile (PreferenceProfile): Profile to convert.
        standardize (bool, optional): If True, divides the weight of each ballot by the total
            weight. Defaults to False.

    Returns:
        dict[Ballot, Fraction]:
            A dictionary with ballots (keys) and corresponding total weights (values).
    """
    tot_weight = profile.total_ballot_wt
    di: dict = {}
    for ballot in profile.ballots:
        weightless_ballot = Ballot(
            ranking=ballot.ranking,
            scores=ballot.scores,
            id=ballot.id,
            voter_set=ballot.voter_set,
        )
        weight = ballot.weight
        if standardize:
            weight /= tot_weight

        if weightless_ballot not in di.keys():
            di[weightless_ballot] = weight
        else:
            di[weightless_ballot] += weight
    return di


def profile_to_ranking_dict(
    profile: PreferenceProfile, standardize: bool = False
) -> dict[tuple[frozenset[str], ...], Fraction]:
    """
    Converts profile to dictionary with keys = rankings and
    values = corresponding total weights.

    Args:
        profile (PreferenceProfile): Profile to convert.
        standardize (bool, optional): If True, divides the weight of each ballot by the total
            weight. Defaults to False.

    Returns:
        dict[tuple[frozenset[str],...], Fraction]:
            A dictionary with rankings (keys) and corresponding total weights (values).

    Raises:
        ValueError: Profile must contain rankings.
    """

    if not profile.contains_rankings:
        raise ValueError(
            (
                "You are trying to convert a profile that contains no rankings to a "
                "ranking_dict."
            )
        )
    tot_weight = profile.total_ballot_wt
    di: dict = {}
    for ballot in profile.ballots:
        ranking = ballot.ranking
        weight = ballot.weight
        if standardize:
            weight /= tot_weight

        if ranking not in di.keys():
            di[ranking] = weight
        else:
            di[ranking] += weight
    return di


def profile_to_scores_dict(
    profile: PreferenceProfile, standardize: bool = False
) -> dict[tuple[str, Fraction], Fraction]:
    """
    Converts profile to dictionary with keys = scores and
    values = corresponding total weights.

    Args:
        profile (PreferenceProfile): Profile to convert.
        standardize (bool, optional): If True, divides the weight of each ballot by the total
            weight. Defaults to False.

    Returns:
        dict[tuple[str, Fraction], Fraction]:
            A dictionary with scores (keys) and corresponding total weights (values).

    Raises:
        ValueError: Profile must contain scores.
    """
    if not profile.contains_scores:
        raise ValueError(
            (
                "You are trying to convert a profile that contains no scores to a "
                "scores_dict."
            )
        )

    tot_weight = profile.total_ballot_wt
    di: dict = {}
    for ballot in profile.ballots:
        scores = tuple(ballot.scores.items()) if ballot.scores else None
        weight = ballot.weight
        if standardize:
            weight /= tot_weight

        if scores not in di.keys():
            di[scores] = weight
        else:
            di[scores] += weight
    return di

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
