import pandas as pd
from typing import Optional, Union
import os
from pathlib import Path
import numpy as np
from votekit.pref_profile import RankProfile
import warnings


def __validate_df_contains_column_idxs(
    df: pd.DataFrame, col_idxs: list[int] | int, *, error_label: str
):
    """
    Validate that the dataframe contains the provided column indices.

    Args:
        df (pd.DataFrame): Dataframe to check.
        col_idxs (list[int] | int): The column indices to check. Can be a list or singleton.
        error_label (str): The type of column being validated.

    Raises:
        ValueError: If the column index is not in the dataframe.

    """
    if not isinstance(col_idxs, list):
        col_idxs = [col_idxs]

    if any(idx < 0 or idx > len(df.columns) - 1 for idx in col_idxs):
        for idx in col_idxs:
            if idx < 0 or idx > len(df.columns) - 1:
                raise ValueError(
                    f"{error_label} column index {idx} must be in [0, {len(df.columns) -1}] "
                    "because Python is 0-indexed."
                )


def __validate_weight_col_values(df: pd.DataFrame, weight_col: int):
    """
    Validate that the weight column has no nan values and all values can be cast to float.

    Args:
        df (pd.DataFrame): The dataframe to validate.
        weight_col (int): The index of the weight column.


    Raises:
        ValueError: If the weight column contains a nan value.
        ValueError: If the weight column contains a value that cannot be cast to float.

    """
    if df[weight_col].isna().any():
        for idx, weight in df[weight_col].items():
            if np.isnan(weight):
                raise ValueError(f"No weight provided in row {idx}.")

    try:
        df[weight_col].astype(float)
    except ValueError:

        for idx, weight in df[weight_col].items():
            try:
                float(weight)
            except ValueError:
                raise ValueError(
                    f"Weight {weight} in row {idx} must be able to be cast to float."
                )


def __validate_distinct_cols(
    rank_cols: list[int], id_col: Optional[int], weight_col: Optional[int]
):
    """
    Validate that the column indices are distinct.

    Args:
        rank_cols (list[int]): The list of ranking column indices.
        id_col (Optional[int]): The index of the Voter ID column.
        weight_col (Optional[int]): The index of the Weight column.

    Raises:
        ValueError: If the id column index is in rank_cols.
        ValueError: If the weight column index is in rank_cols.

    """

    if id_col is not None and id_col in rank_cols:
        raise ValueError(
            f"ID column {id_col} must not be a ranking column {rank_cols}."
        )

    if weight_col is not None and weight_col in rank_cols:
        raise ValueError(
            f"Weight column {weight_col} must not be a ranking column {rank_cols}."
        )


def __validate_columns(
    df: pd.DataFrame,
    rank_cols: list[int],
    id_col: Optional[int],
    weight_col: Optional[int],
):
    """
    Validate that the columns of the csv are correctly formatted.

    Args:
        df (pd.DataFrame): The dataframe of the csv.
        rank_cols (list[int]): The list of ranking column indices.
        id_col (Optional[int]): The index of the Voter ID column.

    Raises:
        ValueError: If a column index is not in the dataframe.
        ValueError: If both the weight and id column are provided.
        ValueError: If the id column index is in rank_cols.
        ValueError: If the weight column index is in rank_cols.
        ValueError: If the weight column contains a nan value.
        ValueError: If the weight column contains a value that cannot be cast to float.

    """

    if weight_col and id_col:
        raise ValueError(
            "Only one of weight_col and id_col can be provided; you cannot have an ID"
            " column if the weight of each ballot is anything but 1."
        )

    __validate_distinct_cols(rank_cols, id_col, weight_col)

    __validate_df_contains_column_idxs(df, rank_cols, error_label="Ranking")

    if id_col is not None:
        __validate_df_contains_column_idxs(df, id_col, error_label="ID")

    if weight_col is not None:
        __validate_df_contains_column_idxs(df, weight_col, error_label="Weight")
        __validate_weight_col_values(df, weight_col)

    return rank_cols, id_col, weight_col


def __format_ranking_cols(
    mutated_df: pd.DataFrame, rank_cols: list[str]
) -> pd.DataFrame:
    """
    Formats the datatype of the ranking columns.

    Args:
        mutated_df (pd.DataFrame): The dataframe of the csv.
        rank_cols (list[str]): The list of ranking column names.

    Returns:
        pd.DataFrame: The mutated dataframe.

    """
    mutated_df[rank_cols] = mutated_df[rank_cols].astype(
        object
    )  # ensure object dtype for sets

    def _format_row(row: pd.Series) -> pd.Series:
        vals = row.to_list()
        out: list[frozenset[str]] = []
        for i, candidate in enumerate(vals):
            if isinstance(candidate, str):
                out.append(frozenset({candidate}))
            elif any(isinstance(c, str) for c in vals[i + 1 :]):
                out.append(frozenset())  # empty set
            else:
                out.append(frozenset({"~"}))  # explicit tilde marker
        return pd.Series(out, index=row.index)

    mutated_df[rank_cols] = mutated_df[rank_cols].apply(_format_row, axis=1)
    return mutated_df


def __format_df(
    mutated_df: pd.DataFrame,
    rank_cols: list[int],
    id_col: Optional[int],
    weight_col: Optional[int],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Formats the column names and datatypes.

    Args:
        mutated_df (pd.DataFrame): The dataframe of the csv.
        rank_cols (list[int]): The list of ranking column indices.
        id_col (int): The index of the ID column.
        weight_col (int): The index of the weight column.


    Returns:
        tuple[pd.DataFrame, list[str]]: The mutated dataframe and new ranking column names.

    """
    mutated_df = mutated_df.copy()
    renamed_columns = {r_col: f"Ranking_{i+1}" for i, r_col in enumerate(rank_cols)}

    if weight_col is not None:
        renamed_columns.update({weight_col: "Weight"})
    else:
        mutated_df["Weight"] = 1

    if id_col is not None:
        renamed_columns.update({id_col: "Voter Set"})
    else:
        mutated_df["Voter Set"] = [set() for _ in range(len(mutated_df))]

    mutated_df.rename(columns=renamed_columns, inplace=True)
    mutated_df.index.name = "Ballot Index"

    str_rank_cols = [c for c in mutated_df.columns if str(c).startswith("Ranking_")]
    mutated_df = mutated_df[str_rank_cols + ["Voter Set", "Weight"]]
    mutated_df = __format_ranking_cols(mutated_df, str_rank_cols)

    mutated_df["Weight"] = mutated_df["Weight"].astype(float)

    mutated_df["Voter Set"] = mutated_df["Voter Set"].map(
        lambda x: {x} if not isinstance(x, set) else x
    )
    return mutated_df, str_rank_cols


def __find_and_validate_cands(
    df: pd.DataFrame, rank_cols: list[str], candidates: Optional[list[str]]
) -> list[str]:
    """
    Finds the candidates in the csv and validates that they match the provided list.

    Args:
        df (pd.DataFrame): The dataframe of the csv.
        rank_cols (list[str]): The list of ranking column labels.
        candidates (Optional[list[str]]): The list of candidates.


    Returns:
        list[str]: The list of candidates found.

    """
    candidates_found: set[str] = set()

    sets = df[rank_cols].to_numpy().ravel()
    candidates_found |= set().union(*sets)

    candidates_found.discard("~")

    if candidates is None:
        candidates = list(candidates_found)

    else:
        if any(c not in candidates_found for c in candidates):
            for c in candidates:
                if c not in candidates_found:
                    raise ValueError(
                        f"Candidate {c} was provided in candidates {candidates} but "
                        "not found in the csv."
                    )

        if any(c not in candidates for c in candidates_found):
            for c in candidates_found:
                if c not in candidates:
                    raise ValueError(
                        f"Candidate {c} was found in the csv but not provided in "
                        f"candidates {candidates}."
                    )

    return candidates


def load_ranking_csv(
    path_or_url: Union[str, os.PathLike, Path],
    rank_cols: list[int],
    *,
    weight_col: Optional[int] = None,
    id_col: Optional[int] = None,
    candidates: Optional[list[str]] = None,
    delimiter: str = ",",
    header_row: Optional[int] = None,
    print_profile: bool = True,
) -> RankProfile:
    """
    Given a file path or url, loads ranked cast vote record (cvr) with ranks as columns and
    voters as rows.

    Args:
        path_or_url (Union[str, os.PathLike, pathlib.Path]): Path or url to cvr file.
        rank_cols (list[int]): List of column indices that contain rankings. Column indexing
            starts from 0, in order from top to bottom rank.
        weight_col (Optional[int]): The column position for ballot weights. Defaults to None, which
            implies each row has weight 1. Cannot be provided if ``id_col`` is also provided.
        id_col (Optional[int]): Index for the column with voter ids. Defaults to None.
            Cannot be provided if ``weight_col`` is also provided.
        candidates (Optional[list[str]]): List of candidate names. Defaults to None, in which case
            names are inferred from the CVR.
        delimiter (Optional[str]): The character that separates entries. Defaults to a comma.
        header_row (Optional[int]): The row containing the column names, below which the data
            begins. Defaults to None, in which case row 0 is considered to be the first ballot.
        print_profile (bool): Whether or not to print the loaded profile. Defaults to True. Useful
            for debugging.


    Raises:
        FileNotFoundError: CSV cannot be found. Raised by ``pandas.read_csv``.
        URLError: Invalid url. Raised by ``pandas.read_csv``.
        HTTPError: URL is valid but other failure occurs. Raised by ``pandas.read_csv``.
        ParserError: Pandas fails to read the csv. Raised by ``pandas.read_csv``.
        UnicodeDecodeError: Bad encoding. Raised by ``pandas.read_csv``.
        ValueError: Candidates provided but they do not exist in the CSV.
        ValueError: Candidates provided but extra candidates are found in the CSV.
        ValueError: Only one of weight_col or id_col can be provided.
        ValueError: weight_col or id_col are not distinct from rank_cols.
        ValueError: weight_col, id_col, and each entry of rank_cols must be non-negative and
            within the number of columns of the csv.
        ValueError: If weight_col is provided, weights must be non-empty and convertible to
            float.
        ValueError: Header must be non-negative.


    Returns:
        RankProfile: A ``RankProfile`` that represents all the ballots in the csv.
    """
    path_or_url = str(path_or_url)

    if header_row is not None and header_row < 0:
        raise ValueError(f"Header row {header_row} must be non-negative.")

    df = pd.read_csv(
        path_or_url,
        on_bad_lines="error",
        encoding="utf8",
        index_col=False,
        delimiter=delimiter,
        header=header_row,
    )

    df.columns = pd.Index(range(len(df.columns)))

    rank_cols, id_col, weight_col = __validate_columns(
        df, rank_cols, id_col, weight_col
    )

    df, str_rank_cols = __format_df(df, rank_cols, id_col, weight_col)

    candidates = __find_and_validate_cands(df, str_rank_cols, candidates)

    profile = RankProfile(
        df=df,
        max_ranking_length=len(str_rank_cols),
        candidates=tuple(candidates),
    )

    if print_profile:
        print(profile)

    return profile


def load_csv(
    path_or_url: str,
    rank_cols: list[int],
    *,
    weight_col: Optional[int] = None,
    id_col: Optional[int] = None,
    candidates: Optional[list[str]] = None,
    delimiter: str = ",",
    header_row: Optional[int] = None,
    print_profile: bool = True,
) -> RankProfile:
    """
    Given a file path or url, loads ranked cast vote record (cvr) with ranks as columns and
    voters as rows.

    Args:
        path_or_url (str): Path or url to cvr file.
        rank_cols (list[int]): List of column indices that contain rankings. Column indexing
            starts from 0, in order from top to bottom rank.
        weight_col (Optional[int]): The column position for ballot weights. Defaults to None, which
            implies each row has weight 1. Cannot be provided if ``id_col`` is also provided.
        id_col (Optional[int]): Index for the column with voter ids. Defaults to None.
            Cannot be provided if ``weight_col`` is also provided.
        candidates (Optional[list[str]]): List of candidate names. Defaults to None, in which case
            names are inferred from the CVR.
        delimiter (Optional[str]): The character that separates entries. Defaults to a comma.
        header_row (Optional[int]): The row containing the column names, below which the data
            begins. Defaults to None, in which case row 0 is considered to be the first ballot.
        print_profile (bool): Whether or not to print the loaded profile. Defaults to True. Useful
            for debugging.


    Raises:
        FileNotFoundError: CSV cannot be found. Raised by ``pandas.read_csv``.
        URLError: Invalid url. Raised by ``pandas.read_csv``.
        HTTPError: URL is valid but other failure occurs. Raised by ``pandas.read_csv``.
        ParserError: Pandas fails to read the csv. Raised by ``pandas.read_csv``.
        UnicodeDecodeError: Bad encoding. Raised by ``pandas.read_csv``.
        ValueError: Candidates provided but they do not exist in the CSV.
        ValueError: Candidates provided but extra candidates are found in the CSV.
        ValueError: Only one of weight_col or id_col can be provided.
        ValueError: weight_col or id_col are not distinct from rank_cols.
        ValueError: weight_col, id_col, and each entry of rank_cols must be non-negative and
            within the number of columns of the csv.
        ValueError: If weight_col is provided, weights must be non-empty and convertible to
            float.
        ValueError: Header must be non-negative.


    Returns:
        RankProfile: A ``RankProfile`` that represents all the ballots in the csv.
    """
    warnings.warn(
        "This function is being deprecated in March 2026. The correct function call is "
        "now load_ranking_csv.",
        DeprecationWarning,
    )

    return load_ranking_csv(
        path_or_url,
        rank_cols,
        weight_col=weight_col,
        id_col=id_col,
        candidates=candidates,
        delimiter=delimiter,
        header_row=header_row,
        print_profile=print_profile,
    )
