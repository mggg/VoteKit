import pandas as pd
from typing import Optional
import numpy as np
from ..pref_profile import PreferenceProfile


def _validate_in_range(df, col_idxs, label):
    if not isinstance(col_idxs, list):
        col_idxs = [col_idxs]

    if any(idx < 0 or idx > len(df.columns) - 1 for idx in col_idxs):
        for idx in col_idxs:
            if idx < 0 or idx > len(df.columns) - 1:
                raise ValueError(
                    f"{label} column index {idx} must be in [0, {len(df.columns) -1}] "
                    "because Python is 0-indexed."
                )


def _validate_rank_columns(df, rank_cols):
    _validate_in_range(df, rank_cols, "Ranking")


def _validate_id_col(df, id_col):
    if id_col is None:
        return
    _validate_in_range(df, id_col, "ID")


def _validate_numeric_weights(df, weight_col):
    try:
        df.iloc[:, weight_col].astype(float)
    except ValueError:

        for idx, weight in df.iloc[:, weight_col].items():
            try:
                float(weight)
            except ValueError:
                raise ValueError(
                    f"Weight {weight} in row {idx} must be able to be cast to float."
                )


def _validate_nonempty_weights(df, weight_col):
    if df.iloc[:, weight_col].isna().any():
        for idx, weight in df.iloc[:, weight_col].items():
            if np.isnan(weight):
                raise ValueError(f"No weight provided in row {idx}.")


def _validate_weight_col(df, weight_col):
    if weight_col is None:
        return

    _validate_in_range(df, weight_col, "Weight")

    _validate_nonempty_weights(df, weight_col)
    _validate_numeric_weights(df, weight_col)


def _validate_distinct_cols(rank_cols, id_col, weight_col):

    if id_col is not None and id_col in rank_cols:
        raise ValueError(
            f"ID column {id_col} must not be a ranking column {rank_cols}."
        )

    if weight_col is not None and weight_col in rank_cols:
        raise ValueError(
            f"Weight column {weight_col} must not be a ranking column {rank_cols}."
        )


def _validate_columns(df, rank_cols, id_col, weight_col):

    if weight_col and id_col:
        raise ValueError(
            "Only one of weight_col and id_col can be provided; you cannot have an ID"
            " column if the weight of each ballot is anything but 1."
        )

    _validate_distinct_cols(rank_cols, id_col, weight_col)

    _validate_rank_columns(df, rank_cols)
    _validate_id_col(df, id_col)
    _validate_weight_col(df, weight_col)

    return rank_cols, id_col, weight_col


def _format_ranking_cols(df, rank_cols):
    for row_idx, row in df.iterrows():
        for rank_idx, (rank_col, candidate) in enumerate(row[rank_cols].items()):
            if isinstance(candidate, str):
                df.at[row_idx, rank_col] = frozenset({candidate})
                continue

            if any(isinstance(c, str) for c in row[rank_cols][(rank_idx + 1) :]):
                df.at[row_idx, rank_col] = frozenset()
                continue

            df.at[row_idx, rank_col] = frozenset("~")

    return df


def _format_df(df, rank_cols, id_col, weight_col):
    renamed_columns = {r_col: f"Ranking_{i+1}" for i, r_col in enumerate(rank_cols)}
    if weight_col is not None:
        renamed_columns.update({weight_col: "Weight"})
    else:
        df["Weight"] = 1

    if id_col is not None:
        renamed_columns.update({id_col: "Voter Set"})
    else:
        df["Voter Set"] = [set() for _ in range(len(df))]

    df.columns = [renamed_columns.get(i, col) for i, col in enumerate(df.columns)]
    df.index.name = "Ballot Index"
    rank_cols = [c for c in df.columns if str(c).startswith("Ranking_")]

    df = df[rank_cols + ["Voter Set", "Weight"]]

    df = _format_ranking_cols(df, rank_cols)

    df["Weight"] = df["Weight"].astype(float)

    df["Voter Set"] = df["Voter Set"].map(
        lambda x: {x} if not isinstance(x, set) else x
    )
    return df, rank_cols


def _find_and_validate_cands(df, rank_cols, candidates):
    candidates_found: set[str] = set()

    sets = df.loc[:, rank_cols].to_numpy().ravel()
    candidates_found |= set().union(*sets)

    candidates_found.discard("~")

    if candidates is None:
        candidates = candidates_found

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


def load_csv(
    path_or_url: str,
    rank_cols: list[int],
    *,
    weight_col: Optional[int] = None,
    id_col: Optional[int] = None,
    candidates: Optional[list[str]] = None,
    delimiter: str = ",",
    header: Optional[int] = None,
) -> PreferenceProfile:
    """
    Given a file path or url, loads cast vote record (cvr) with ranks as columns and voters as rows.
    Does not currently support score based profiles.

    Args:
        path_or_url (str): Path or url to cvr file.
        rank_cols (list[int]): List of column indices that contain rankings. Column indexing
            starts from 0, in order from top to bottom rank.
        weight_col (int, optional): The column position for ballot weights. Defaults to None, which
            implies each row has weight 1.
        id_col (int, optional): Index for the column with voter ids. Defaults to None.
        candidates (list[str], optional): List of candidate names. Defaults to None, in which case
            names are inferred from the CVR.
        delimiter (str, optional): The character that separates entries. Defaults to a comma.
        header (int, optional): The row containing the column names, below which the datae begins.
            Defaults to None, in which case row 0 is considered to be the first ballot.


    Raises:
        FileNotFoundError: CSV cannot be found
        URLError: Invalid url.
        HTTPError: URL is valid but other failure occurs.
        ParserError: Pandas fails to read the csv.
        UnicodeDecodeError: Bad encoding.
        ValueError: Candidates provided but they do not exist in the CSV.
        ValueError: Candidates provided but extra candidates are found in the CSV.
        ValueError: Only one of weight_col or id_col can be provided.
        ValueError: weight_col or id_col are not distinct from rank_cols.
        ValueError: weight_col, id_col, and each entry of rank_cols must be non-negative and
            within the number of columns of the csv.
        ValueError: If weight_col is provided, weights must be non-empty and convertible to
            float.
        ValueError: If no rank_cols are provided, but weight or id col is, and the csv has only one
            column.
        ValueError: Header must be non-negative.


    Returns:
        PreferenceProfile: A ``PreferenceProfile`` that represents all the ballots in the csv.
    """
    if header is not None and header < 0:
        raise ValueError(f"Header {header} must be non-negative.")

    df = pd.read_csv(
        path_or_url,
        on_bad_lines="error",
        encoding="utf8",
        index_col=False,
        delimiter=delimiter,
        header=header,
    )

    rank_cols, id_col, weight_col = _validate_columns(df, rank_cols, id_col, weight_col)

    df, rank_cols = _format_df(df, rank_cols, id_col, weight_col)

    candidates = _find_and_validate_cands(df, rank_cols, candidates)

    return PreferenceProfile(
        df=df,
        max_ranking_length=len(rank_cols),
        contains_rankings=True,
        contains_scores=False,
        candidates=candidates,
    )
