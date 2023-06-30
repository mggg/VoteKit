import pandas as pd
import pathlib, os
from pydantic import BaseModel
from typing import Callable
from unnamed_rcv_thing.profile import PreferenceProfile
from unnamed_rcv_thing.ballot import Ballot
from pandas.errors import EmptyDataError, DataError

class CVRLoader(BaseModel):
    """
    load_func (Callable([str, PreferenceProfile])):
      given a file path, loads cvr into a preference profile
    """
    load_func: Callable[[str], PreferenceProfile]

    class Config:
        allow_mutation = False
    
    def load_cvr(self, fpath: str, id_col: int = None) -> PreferenceProfile:
        """
        checks for the following errors and calls load_func to parse cvr file
        Args:
            fpath (str): path to cvr file
            id_col (int, optional): index for the column with voter ids. Defaults to None.

        Raises:
            FileNotFoundError: if fpath is invalid
            EmptyDataError: if dataset is empty
            ValueError: if the voter id column has missing values
            DataError: if the voter id column has duplicate values

        Returns:
            PreferenceProfile: preference schedule that contains all the parsed ballots
        """
        if not os.path.isfile(fpath):
            raise FileNotFoundError(f'File with path {fpath} cannot be found')
        cvr_path = pathlib.Path(fpath)
        df = pd.read_csv(cvr_path, on_bad_lines='error', encoding="utf8")
        if df.empty:
            raise EmptyDataError('Dataset cannot be empty')
        if id_col != None and df.iloc[:, id_col].isnull().values.any():
            raise ValueError(f'Missing value(s) in column at index {id_col}')
        if id_col != None and not df.iloc[:, id_col].is_unique:
            raise DataError(f'Duplicate value(s) in column at index {id_col}')

        return self.load_func(fpath, id_col)

def rank_column_csv(fpath: str, id_col: int = None) -> PreferenceProfile:
    """
    given a file path, loads cvr with ranks as columns and voters as rows
    (empty cells are treated as None)
    (if voter ids are missing, we're currently not assigning ids)
    Args:
        fpath (str): path to cvr file
        id_col (int, optional): index for the column with voter ids
    Returns:
        PreferenceProfile: a preference schedule that represents all the ballots in the elction
    """

    cvr_path = pathlib.Path(fpath)
    df = pd.read_csv(cvr_path, on_bad_lines='error', encoding="utf8")

    ranks = list(df.columns)
    if id_col != None:
        ranks.remove(df.columns[id_col])
    grouped = df.groupby(ranks, dropna=False)
    ballots = []

    for group, group_df in grouped:
        group = [None if pd.isnull(c) else c for c in group]
        ranking = list({c} for c in list(group))
        voters = None
        if id_col != None:
            voters = list(group_df.iloc[:, id_col])
        weight = len(group_df)
        b = Ballot(ranking=ranking, weight=weight, voters=voters)
        ballots.append(b)
    
    return PreferenceProfile(ballots=ballots)

