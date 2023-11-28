from fractions import Fraction
import os
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
    Given a file path, loads cast vote records (cvr) with ranks as columns and voters as rows.
    Empty cells are treated as None.

    Args:
        fpath: Path to cvr file.
        rank_cols: List of column indexes that contain rankings. Indexing starts from 0,
                    in order from top to bottom rank.
                    Default implies that all columns contain rankings.
        weight_col: The column position for ballot weights.
        delimiter: The character that breaks up rows.
        id_col: Index for the column with voter ids.

    Raises:
        FileNotFoundError: If fpath is invalid.
        EmptyDataError: If dataset is empty.
        ValueError: If the voter id column has missing values.
        DataError: If the voter id column has duplicate values.

    Returns:
        A PreferenceProfile that represents all the ballots in the election.
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
            df = df.iloc[:, rank_cols+[id_col]]
        else:
            df = df.iloc[:, rank_cols]

    ranks = list(df.columns)
    if id_col is not None:
        ranks.remove(df.columns[id_col])
    grouped = df.groupby(ranks, dropna=False)
    ballots = []

    for group, group_df in grouped:
        ranking = [{None} if pd.isnull(c) else {c} for c in group]


        voter_set = None
        if id_col is not None:
            voter_set = set(group_df.iloc[:, id_col])
        weight = len(group_df)
        if weight_col is not None:
            weight = sum(group_df.iloc[:, weight_col])
        b = Ballot(ranking=ranking, weight=Fraction(weight), voter_set=voter_set)
        ballots.append(b)

    return PreferenceProfile(ballots=ballots)


def load_scottish(fpath: str) -> tuple[PreferenceProfile, int]:
    """
    Given a file path, loads cvr from format used for Scottish election data.

    Args:
        fpath: Path to cvr file.

    Raises:
        FileNotFoundError: If fpath is invalid.
        EmptyDataError: If dataset is empty.
        DataError: If there is missing or incorrect metadata or candidate data.

    Returns:
        (tuple): A tuple (PreferenceProfile, seats) representing the election and the 
                number of seats in the election.
    """
    ballots = []
    names = []
    name_map = {}
    numbers = True
    cands_included = False

    if not os.path.isfile(fpath):
        raise FileNotFoundError(f"File with path {fpath} cannot be found")
    if os.path.getsize(fpath) == 0:
        raise EmptyDataError("Dataset cannot be empty")

    with open(fpath, "r") as file:
        for i, line in enumerate(file):
            s = line.rstrip("\n").rstrip()
            if i == 0:
                # first number is number of candidates, second is number of seats to elect
                metadata = [int(data) for data in s.split(" ")]
                if len(metadata) != 2:
                    raise DataError(
                        "metadata (first line) should have two parameters"
                        " (number of candidates, number of seats)"
                    )
                seats = metadata[1]
            # read in ballots, cleaning out rankings labeled '0' (designating end of line)
            elif numbers:
                ballot = [int(vote) for vote in s.split(" ")]
                num_votes = ballot[0]
                # ballots terminate with a single row with the character '0'
                if num_votes == 0:
                    numbers = False
                else:
                    ranking = [rank for rank in list(ballot[1:]) if rank != 0]
                    b = (ranking, num_votes)
                    ballots.append(b)  # this is converted to the PP format later
            # read in candidates
            elif "(" in s:
                cands_included = True
                name_parts = s.strip('"').split(" ")
                first_name = " ".join(name_parts[:-2])
                last_name = name_parts[-2]
                party = name_parts[-1].strip("(").strip(")")
                names.append(str((first_name, last_name, party)))
            else:
                if len(names) != metadata[0]:
                    err_message = (
                        f"Number of candidates listed, {len(names)}," + f" differs from"
                        f"number of candidates recorded in metadata, {metadata[0]}"
                    )
                    raise DataError(err_message)
                # read in election location (do we need this?)
                # location = s.strip("\"")
                if not cands_included:
                    raise DataError("Candidates missing from file")
                # map candidate numbers onto their names and convert ballots to PP format
                for i, name in enumerate(names):
                    name_map[i + 1] = name
                clean_ballots = [
                    Ballot(
                        ranking=[{name_map[cand]} for cand in ballot[0]],
                        weight=Fraction(ballot[1]),
                    )
                    for ballot in ballots
                ]

        return PreferenceProfile(ballots=clean_ballots, candidates=names), seats
