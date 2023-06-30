import pandas as pd
import pathlib
import os
from .profile import PreferenceProfile
from .ballot import Ballot
from pandas.errors import EmptyDataError, DataError

def rank_column_csv(fpath: str, id_col: int = None) -> PreferenceProfile:
    """
    given a file path, loads cvr with ranks as columns and voters as rows
    (empty cells are treated as None)
    (if voter ids are missing, we're currently not assigning ids)
    Args:
        fpath (str): path to cvr file
        id_col (int, optional): index for the column with voter ids
    Raises:
        FileNotFoundError: if fpath is invalid
        EmptyDataError: if dataset is empty
        ValueError: if the voter id column has missing values
        DataError: if the voter id column has duplicate values
    Returns:
        PreferenceProfile: a preference schedule that represents all the ballots in the elction
    """
    if not os.path.isfile(fpath):
        raise FileNotFoundError(f'File with path {fpath} cannot be found')
    cvr_path = pathlib.Path(fpath)
    df = pd.read_csv(cvr_path, on_bad_lines='error', encoding="utf8")
    if df.empty:
        raise EmptyDataError('Dataset cannot be empty')
    if id_col and df.iloc[:, id_col].isnull().values.any():
        raise ValueError(f'Missing value(s) in column at index {id_col}')
    if id_col and not df.iloc[:, id_col].is_unique:
        raise DataError(f'Duplicate value(s) in column at index {id_col}')

    ranks = list(df.columns)
    if id_col is not None:
        ranks.remove(df.columns[id_col])
    grouped = df.groupby(ranks, dropna=False)
    ballots = []

    for group, group_df in grouped:
        ranking = [{None} if pd.isnull(c) else {c} for c in group]
        voters = None
        if id_col is not None:
            voters = list(group_df.iloc[:, id_col])
        weight = len(group_df)
        b = Ballot(ranking=ranking, weight=weight, voters=voters)
        ballots.append(b)
    
    return PreferenceProfile(ballots=ballots)
