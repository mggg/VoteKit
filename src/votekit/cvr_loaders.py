from fractions import Fraction
import os
import pandas as pd
from pandas.errors import EmptyDataError, DataError
import pathlib
from typing import Optional

from .pref_profile import PreferenceProfile
from .ballot import Ballot

# TODO: update documentation for function below


def rank_column_csv(
    fpath: str,
    *,
    weight_col: Optional[int] = None,
    delimiter: Optional[str] = None,
    id_col: Optional[int] = None,
) -> PreferenceProfile:
    """
    Takes a file path and loads a cast vote record (cvr) with cast votes as cols and voters as rows.
    Empty cells are treated as None. Currently, missing voter ids are not assigned

    :param fpath: :class:`str`: path to cvr file
    :param weight_col: optional :class:`int` the column position for ballot weights
    if parsing Scottish elections like cvrs
    :param delimiter: :class:`str`: the character that breaks up rows
    :param id_col: optional :class:`int`: index for the column with voter ids \n

    Raises:
        `FileNotFoundError`: if fpath is invalid \n
        `EmptyDataError`: if dataset is empty \n
        `ValueError`: if the voter id column has missing values \n
        `DataError`: if the voter id column has duplicate values

    :return: A preference profile object with all cast ballots from file
    :rtype: :class:`PreferenceProfile`
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

    ranks = list(df.columns)
    if id_col is not None:
        ranks.remove(df.columns[id_col])
    grouped = df.groupby(ranks, dropna=False)
    ballots = []

    for group, group_df in grouped:
        ranking = [{None} if pd.isnull(c) else {c} for c in group]
        voters = None
        if id_col is not None:
            voters = set(group_df.iloc[:, id_col])
        weight = len(group_df)
        if weight_col is not None:
            weight = sum(group_df.iloc[:, weight_col])
        b = Ballot(ranking=ranking, weight=Fraction(weight), voters=voters)
        ballots.append(b)

    return PreferenceProfile(ballots=ballots)


def blt(fpath: str) -> tuple[PreferenceProfile, int]:
    """
    Loads cast vote record from .blt file.
    blt is text-like format used for scottish election data
    the first line of the file is metadata recording the number of candidates and seats,
    followed by ballot data (first number in row is ballot weight),
    followed by candidate data (order corresponds to number in ballots),
    followed by election location
    :param fpath: :class:`str`: path to cvr file \n
    Raises:\n
    `FileNotFoundError`: if fpath is invalid \n
    `EmptyDataError`: if dataset is empty \n
    `DataError`: if there is missing or incorrect metadata or candidate data \n
    :return: preference profile with all cast ballots from file and number of seats in election
    :rtype: :class:`PreferenceProfile`  , :class:`int`

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
