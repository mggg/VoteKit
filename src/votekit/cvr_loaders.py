from fractions import Fraction
import os
import csv
import pandas as pd
from pandas.errors import EmptyDataError, DataError
import pathlib
from typing import Optional

from .pref_profile import PreferenceProfile
from .ballot import Ballot


def load_csv(
    fpath: str,
    rank_cols: list[int] = [],
    *,
    weight_col: Optional[int] = None,
    delimiter: Optional[str] = None,
    id_col: Optional[int] = None,
) -> PreferenceProfile:
    """
    Given a file path, loads cast vote record (cvr) with ranks as columns and voters as rows.
    Empty cells are treated as None.

    Args:
        fpath (str): Path to cvr file.
        rank_cols (list[int]): List of column indexes that contain rankings. Indexing starts from 0,
            in order from top to bottom rank. Default is empty list, which implies that all columns
            contain rankings.
        weight_col (int, optional): The column position for ballot weights. Defaults to None, which
            implies each row has weight 1.
        delimiter (str, optional): The character that breaks up rows. Defaults to None, which
            implies a carriage return.
        id_col (int, optional): Index for the column with voter ids. Defaults to None.

    Raises:
        FileNotFoundError: If fpath is invalid.
        EmptyDataError: If dataset is empty.
        ValueError: If the voter id column has missing values.
        DataError: If the voter id column has duplicate values.

    Returns:
        PreferenceProfile: A ``PreferenceProfile`` that represents all the ballots in the election.
    """
    if not os.path.isfile(fpath):
        raise FileNotFoundError(f"File with path {fpath} cannot be found")

    cvr_path = pathlib.Path(fpath)
    df = pd.read_csv(
        cvr_path,
        on_bad_lines="error",
        encoding="utf8",
        index_col=False,
        delimiter=delimiter,
    )

    if df.empty:
        raise EmptyDataError("Dataset cannot be empty")
    if id_col is not None and df.iloc[:, id_col].isnull().values.any():  # type: ignore
        raise ValueError(f"Missing value(s) in column at index {id_col}")
    if id_col is not None and not df.iloc[:, id_col].is_unique:
        raise DataError(f"Duplicate value(s) in column at index {id_col}")

    if rank_cols:
        if id_col is not None:
            df = df.iloc[:, rank_cols + [id_col]]
        else:
            df = df.iloc[:, rank_cols]

    ranks = list(df.columns)
    if id_col is not None:
        ranks.remove(df.columns[id_col])
    grouped = df.groupby(ranks, dropna=False)
    ballots = []

    for group, group_df in grouped:
        ranking = tuple(
            [frozenset({None}) if pd.isnull(c) else frozenset({c}) for c in group]
        )

        voter_set = None
        if id_col is not None:
            voter_set = set(group_df.iloc[:, id_col])
        weight = len(group_df)
        if weight_col is not None:
            weight = sum(group_df.iloc[:, weight_col])
        b = Ballot(ranking=ranking, weight=Fraction(weight), voter_set=voter_set)
        ballots.append(b)

    return PreferenceProfile(ballots=tuple(ballots))


def load_scottish(
    fpath: str,
) -> tuple[PreferenceProfile, int, list[str], dict[str, str], str]:
    """
    Given a file path, loads cast vote record from format used for Scottish election data
    in (this repo)[https://github.com/mggg/scot-elex].

    Args:
        fpath (str): Path to Scottish election csv file.

    Raises:
        FileNotFoundError: If fpath is invalid.
        EmptyDataError: If dataset is empty.
        DataError: If there is missing or incorrect metadata or candidate data.

    Returns:
        tuple:
            A tuple ``(PreferenceProfile, seats, cand_list, cand_to_party, ward)``
            representing the election, the number of seats in the election, the candidate
            names, a dictionary mapping candidates to their party, and the ward. The
            candidate names are also stored in the PreferenceProfile object.
    """

    if not os.path.isfile(fpath):
        raise FileNotFoundError(f"File with path {fpath} cannot be found")
    if os.path.getsize(fpath) == 0:
        raise EmptyDataError(f"CSV at {fpath} is empty.")

    # Convert the ballot rows to ints while leaving the candidates as strings
    def convert_row(row):
        return [int(item) if item.isdigit() else item for item in row]

    data = []
    with open(fpath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            # This just removes any empty strings that are hanging out since
            # we don't need to preserve columns
            filtered_row = list(filter(lambda x: x != "", row))

            # only save non-empty rows
            if len(filtered_row) > 0:
                data.append(convert_row(filtered_row))

    if len(data[0]) != 2:
        raise DataError(
            "The metadata in the first row should be number of \
                            candidates, seats."
        )

    cand_num, seats = data[0][0], data[0][1]
    ward = data[-1][0]

    num_to_cand = {}
    cand_to_party = {}

    data_cand_num = len([r for r in data if "Candidate" in str(r[0])])
    if data_cand_num != cand_num:
        raise DataError(
            "Incorrect number of candidates in either first row metadata \
                        or in candidate list at end of csv file."
        )

    # record candidate names, which are up until the final row
    for i, line in enumerate(data[len(data) - (cand_num + 1) : -1]):
        if "Candidate" not in line[0]:
            raise DataError(
                f"The number of candidates on line 1 is {cand_num}, which\
                            does not match the metadata."
            )
        cand = line[1]
        party = line[2]

        # candidates are 1 indexed
        num_to_cand[i + 1] = cand
        cand_to_party[cand] = party

    cand_list = list(cand_to_party.keys())

    ballots = [Ballot()] * len(data[1 : len(data) - (cand_num + 1)])

    for i, line in enumerate(data[1 : len(data) - (cand_num + 1)]):
        ballot_weight = Fraction(line[0])
        cand_ordering = line[1:]
        ranking = tuple([frozenset({num_to_cand[n]}) for n in cand_ordering])

        ballots[i] = Ballot(ranking=ranking, weight=ballot_weight)

    profile = PreferenceProfile(
        ballots=tuple(ballots), candidates=tuple(cand_list)
    ).condense_ballots()
    return (profile, seats, cand_list, cand_to_party, ward)
