import pandas as pd
from typing import Optional

from ..pref_profile import PreferenceProfile


def _validate_rank_columns(df, rank_cols):
    if any(r_col < 0 or r_col > len(df.columns) for r_col in rank_cols):
        for r_col in rank_cols:
            if r_col < 0:
                raise ValueError(f"Ranking column {r_col} must be non-negative.")
            elif r_col > len(df.columns):
                raise ValueError(
                    f"Ranking column {r_col} must be less than {len(df.columns)}, "
                    "the number of columns of the csv file."
                )


def _validate_id_col(df, id_col):
    if id_col is not None:
        if id_col < 0:
            raise ValueError(f"ID column {id_col} must be non-negative.")
        elif id_col > len(df.columns):
            raise ValueError(
                f"ID column {id_col} must be less than {len(df.columns)}, "
                "the number of columns of the csv file."
            )


def _validate_weight_col(df, weight_col):
    if weight_col is not None:
        if weight_col < 0:
            raise ValueError(f"Weight column {weight_col} must be non-negative.")
        elif weight_col > len(df.columns):
            raise ValueError(
                f"Weight column {weight_col} must be less than {len(df.columns)}, "
                "the number of columns of the csv file."
            )


def _validate_distinct_cols(df, rank_cols, id_col, weight_col):
    if id_col is not None and weight_col is not None:
        if id_col == weight_col:
            raise ValueError(
                f"ID column {id_col} and weight column {weight_col} must be distinct."
            )

    if id_col in rank_cols:
        raise ValueError(
            f"ID column {id_col} must not be a ranking column {rank_cols}."
        )

    if weight_col in rank_cols:
        raise ValueError(
            f"Weight column {weight_col} must not be a ranking column {rank_cols}."
        )


def _validate_columns(df, rank_cols, id_col, weight_col):

    _validate_rank_columns(df, rank_cols)
    _validate_id_col(df, id_col)
    _validate_weight_col(df, weight_col)
    _validate_distinct_cols(df, rank_cols, id_col, weight_col)


def _format_df(df, rank_cols, id_col, weight_col):
    renamed_columns = {r_col: f"Ranking_{i+1}" for i, r_col in enumerate(rank_cols)}
    if weight_col is not None:
        renamed_columns.update({weight_col: "Weight"})
    else:
        df["Weight"] = 1

    if id_col is not None:
        renamed_columns.update({id_col: "Voter Set"})
    else:
        df["Voter Set"] = set()

    df.columns = [renamed_columns.get(i, col) for i, col in enumerate(df.columns)]
    df.index.name = "Ballot Index"
    rank_cols = [c for c in df.columns if c.startswith("Ranking_")]

    df = df[rank_cols + ["Voter Set", "Weight"]]

    # TODO is there any way to deal with empty as skip versus empty as no ranking?
    # #no ranking should be a tilde, skip should be empty frozenset
    # but if you default to the tilde, then there are "invalid ballots" that are empty at the front
    for r_col in rank_cols:
        df[r_col] = df[r_col].map(
            lambda x: frozenset({x}) if isinstance(x, str) else frozenset()
        )  # handles nan from empty load

    df["Voter Set"] = df["Voter Set"].map(lambda x: {x})
    return df, rank_cols


def _find_and_validate_cands(df, rank_cols, candidates):
    # find candidates in csv
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
                        f"Candidate {c} was found in the csv but not but not provided in "
                        "candidates {candidates}."
                    )

    return candidates


def load_csv(
    path_or_url: str,
    rank_cols: Optional[list[int]] = None,
    *,
    weight_col: Optional[int] = None,
    id_col: Optional[int] = None,
    candidates: Optional[list[str]] = None,
    delimiter: str = ",",
) -> PreferenceProfile:
    """
    Given a file path or url, loads cast vote record (cvr) with ranks as columns and voters as rows.
    Does not currently support score based profiles.

    Args:
        path_or_url (str): Path or url to cvr file.
        rank_cols (list[int], optional): List of column indices that contain rankings. Indexing
            starts from 0, in order from top to bottom rank. Default is None, which implies
            that all columns contain rankings.
        weight_col (int, optional): The column position for ballot weights. Defaults to None, which
            implies each row has weight 1.
        id_col (int, optional): Index for the column with voter ids. Defaults to None.
        candidates (list[str], optional): List of candidate names. Defaults to None, in which case
            names are inferred from the CVR.
        delimiter (str, optional): The character that separates entries. Defaults to a comma.


    Raises:
        FileNotFoundError: If fpath is invalid.
        EmptyDataError: If dataset is empty.
        ValueError: If the voter id column has missing values.
        DataError: If the voter id column has duplicate values.

    Returns:
        PreferenceProfile: A ``PreferenceProfile`` that represents all the ballots in the csv.
    """
    # TODO what does pandas raise for a bad file path or url
    df = pd.read_csv(
        path_or_url,
        on_bad_lines="error",
        encoding="utf8",
        index_col=False,
        delimiter=delimiter,
    )

    if rank_cols is None:
        rank_cols = list(range(len(df.columns)))

    _validate_columns(df, rank_cols, id_col, weight_col)

    df, rank_cols = _format_df(df, rank_cols, id_col, weight_col)

    candidates = _find_and_validate_cands(df, rank_cols, candidates)

    # # TODO if weight is not 1, how do we handle voter id
    # # TODO check for empty voter id and weight entries if those columns were provided
    # TODO check that weight can be cast to float?
    # TODO call .ballots in a test b/c it will notice if there is a tilde in an invalid place

    return PreferenceProfile(
        df=df,
        max_ranking_length=len(rank_cols),
        contains_rankings=True,
        contains_scores=False,
        candidates=candidates,
    )
