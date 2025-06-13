from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pref_profile import PreferenceProfile

from ..ballot import Ballot
from typing import Optional
import pandas as pd
from functools import partial
from .profile_error import ProfileError


def _convert_ranking_cols_to_ranking(
    row: pd.Series, max_ranking_length: int
) -> Optional[tuple[frozenset, ...]]:
    """
    Convert the ranking cols to a ranking tuple in profile.df.

    Args:
        row (pd.Series): Row of a profile.df.
        max_ranking_length (int, optional): The maximum length of a ranking.

    Returns:
        Optional[tuple[frozenset, ...]]: Ranking of ballot.

    Raises:
        ValueError: NaN values can only trail on a ranking.

    """
    ranking_cols_idxs = [f"Ranking_{i+1}" for i in range(max_ranking_length)]

    if any(idx not in row.index for idx in ranking_cols_idxs):
        raise ValueError(f"Row has improper ranking columns: {row.index}.")

    if any(
        row[col_idx] == frozenset({"~"})
        and not all(row[idx] == frozenset({"~"}) for idx in ranking_cols_idxs[i:])
        for i, col_idx in enumerate(ranking_cols_idxs)
    ):
        raise ValueError(
            f"Row {row} has '~' between valid ranking positions. "
            "'~' values can only trail on a ranking."
        )

    ranking = [
        row[col_idx] for col_idx in ranking_cols_idxs if row[col_idx] != frozenset("~")
    ]

    return tuple(ranking) if len(ranking) > 0 else None


def convert_row_to_ballot(
    row: pd.Series, candidates: tuple[str, ...], max_ranking_length: int = 0
) -> Ballot:
    """
    Convert a row of a properly formatted profile.df to a Ballot.

    Args:
        row (pd.Series): Row of a profile.df.
        candidates (tuple[str,...]): The name of the candidates.
        max_ranking_length (int, optional): The maximum length of a ranking. Defaults to 0, which
            is used for ballots with no ranking.

    Returns:
        Ballot: Ballot corresponding to the row of the df.
    """
    ranking = None
    if max_ranking_length > 0:
        ranking = _convert_ranking_cols_to_ranking(row, max_ranking_length)
    scores = {c: row[c] for c in candidates if c in row and not pd.isna(row[c])}
    voter_set = row["Voter Set"]
    weight = row["Weight"]

    return Ballot(
        ranking=ranking,
        scores=scores if scores else None,
        weight=weight,
        voter_set=voter_set,
    )


def _df_to_ballot_tuple(
    df: pd.DataFrame, candidates: tuple[str, ...], max_ranking_length: int = 0
) -> tuple[Ballot, ...]:
    """
    Convert a properly formatted profile.df into a list of ballots.

    Args:
        df (pd.DataFrame): A profile.df.
        candidates (tuple[str,...]): The candidates.
        max_ranking_length (int, optional): The maximum length of a ranking. Defaults to 0, which
            is used for ballots with no ranking.

    Returns:
        tuple[Ballot]: The tuple of ballots.
    """
    if df.empty:
        return tuple()

    return tuple(
        df.apply(  # type: ignore[call-overload]
            partial(
                convert_row_to_ballot,
                candidates=candidates,
                max_ranking_length=max_ranking_length,
            ),
            axis="columns",
        )
    )


def profile_to_ballot_dict(
    profile: PreferenceProfile, standardize: bool = False
) -> dict[Ballot, float]:
    """
    Converts profile to dictionary with keys = ballots and
    values = corresponding total weights.

    Args:
        profile (PreferenceProfile): Profile to convert.
        standardize (bool, optional): If True, divides the weight of each ballot by the total
            weight. Defaults to False.

    Returns:
        dict[Ballot, float]:
            A dictionary with ballots (keys) and corresponding total weights (values).
    """
    tot_weight = profile.total_ballot_wt
    di: dict = {}
    for ballot in profile.ballots:
        print(ballot)
        weightless_ballot = Ballot(
            ranking=ballot.ranking,
            scores=ballot.scores,
            voter_set=ballot.voter_set,
        )
        weight = ballot.weight
        if standardize:
            weight /= tot_weight

        if weightless_ballot not in di.keys():
            di[weightless_ballot] = weight
        else:
            di[weightless_ballot] += weight
        print()
    return di


def profile_to_ranking_dict(
    profile: PreferenceProfile, standardize: bool = False
) -> dict[tuple[frozenset[str], ...], float]:
    """
    Converts profile to dictionary with keys = rankings and
    values = corresponding total weights.

    Args:
        profile (PreferenceProfile): Profile to convert.
        standardize (bool, optional): If True, divides the weight of each ballot by the total
            weight. Defaults to False.

    Returns:
        dict[tuple[frozenset[str],...], float]:
            A dictionary with rankings (keys) and corresponding total weights (values).

    Raises:
        ProfileError: Profile must contain rankings.
    """

    if not profile.contains_rankings:
        raise ProfileError(
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
        di[ranking] = di.get(ranking, 0) + weight

    return di


def profile_to_scores_dict(
    profile: PreferenceProfile, standardize: bool = False
) -> dict[tuple[str, float], float]:
    """
    Converts profile to dictionary with keys = scores and
    values = corresponding total weights.

    Args:
        profile (PreferenceProfile): Profile to convert.
        standardize (bool, optional): If True, divides the weight of each ballot by the total
            weight. Defaults to False.

    Returns:
        dict[tuple[str, float], float]:
            A dictionary with scores (keys) and corresponding total weights (values).

    Raises:
        ProfileError: Profile must contain scores.
    """
    if not profile.contains_scores:
        raise ProfileError(
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

        di[scores] = di.get(scores, 0) + weight
    return di


def profile_df_head(
    profile: PreferenceProfile,
    n: int,
    sort_by_weight: Optional[bool] = True,
    percents: Optional[bool] = False,
    totals: Optional[bool] = False,
    n_decimals: int = 1,
) -> pd.DataFrame:
    """
    Returns a pd.DataFrame with the top-n ballots in profile.

    Args:
        n (int): Number of ballots to view.
        sort_by_weight (bool, optional): If True, rank ballot from most to least votes.
            If sorting by weight, index resets. Defaults to True.
        percents (bool, optional): If True, show voter share for a given ballot.
            Defaults to False.
        totals (bool, optional): If True, show total values for Percent and Weight.
            Defaults to False.
        n_decimals (int, optional): Number of decimals to round to. Defaults to 1.

    Returns:
        pandas.DataFrame: A dataframe with top-n ballots.

    Raises:
        ZeroDivisionError: Profile has 0 total ballot weight; cannot show percentages.
    """

    if sort_by_weight:
        df = profile.df.sort_values(by="Weight", ascending=False).head(n).copy()

    else:
        df = profile.df.head(n).copy()

    df_col_num = len(df.columns)
    if percents:
        if profile.total_ballot_wt == 0:
            raise ZeroDivisionError(
                "Profile has 0 total ballot weight; cannot show percentages."
            )
        df["Percent"] = df["Weight"] / float(profile.total_ballot_wt)

    if totals:
        total_row = [""] * (df_col_num - 1) + [df["Weight"].sum()]
        if percents:
            total_row += [df["Percent"].sum()]
        df.loc["Total"] = total_row

    if percents:
        df["Percent"] = df["Percent"].apply(lambda x: f"{float(x):.{n_decimals}%}")

    return df


def profile_df_tail(
    profile: PreferenceProfile,
    n: int,
    sort_by_weight: Optional[bool] = True,
    percents: Optional[bool] = False,
    totals: Optional[bool] = False,
    n_decimals: int = 1,
) -> pd.DataFrame:
    """
    Returns a pd.DataFrame with the bottom-n ballots in profile.

    Args:
        n (int): Number of ballots to view.
        sort_by_weight (bool, optional): If True, rank ballot from least to most votes.
            Defaults to True.
        percents (bool, optional): If True, show voter share for a given ballot.
            Defaults to False.
        totals (bool, optional): If True, show total values for Percent and Weight.
            Defaults to False.
        n_decimals (int, optional): Number of decimals to round to. Defaults to 1.

    Returns:
        pandas.DataFrame: A data frame with bottom-n ballots.

    Raises:
        ZeroDivisionError: Profile has 0 total ballot weight; cannot show percentages.
    """
    if sort_by_weight:
        df = profile.df.sort_values(by="Weight", ascending=False).tail(n).copy()
    else:
        df = profile.df.tail(n).copy()

    df_col_num = len(df.columns)
    if percents:
        if profile.total_ballot_wt == 0:
            raise ZeroDivisionError(
                "Profile has 0 total ballot weight; cannot show percentages."
            )
        df["Percent"] = df["Weight"] / float(profile.total_ballot_wt)

    if totals:
        total_row = [""] * (df_col_num - 1) + [df["Weight"].sum()]
        if percents:
            total_row += [df["Percent"].sum()]
        df.loc["Total"] = total_row

    if percents:
        df["Percent"] = df["Percent"].apply(lambda x: f"{float(x):.{n_decimals}%}")

    return df
