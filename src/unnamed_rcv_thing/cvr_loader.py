import pandas as pd
import pathlib, os
from pydantic import BaseModel
from typing import Optional, Callable
from unnamed_rcv_thing.profile import PreferenceProfile
from unnamed_rcv_thing.ballot import Ballot
from pandas.errors import EmptyDataError

class CVRLoader(BaseModel):
    """
    load_func (Callable([str, PreferenceProfile])):
      given a file path, loads cvr into a preference profile
    """
    load_func: Callable[[str], PreferenceProfile]

    class Config:
        allow_mutation = False
    
    def load_cvr(self, fpath: str) -> PreferenceProfile:
        if not os.path.isfile(fpath):
            raise FileNotFoundError(f'File with path {fpath} cannot be found')
        cvr_path = pathlib.Path(fpath)
        df = pd.read_csv(cvr_path, encoding="utf8")
        if df.empty:
            raise EmptyDataError('Dataset cannot be empty')
        
        #TODO: add a missing header function

        return self.load_func(fpath)

def rank_column_csv(fpath: str) -> PreferenceProfile:
    """
    given a file path, loads cvr with ranks as columns and voters as rows
    Args:
        fpath (str): path to cvr file
    Returns:
        PreferenceProfile: a preference schedule that represents all the ballots in the elction
    """

    cvr_path = pathlib.Path(fpath)
    df = pd.read_csv(cvr_path, encoding="utf8")
    grouped = df.groupby(list(df.columns[1:]), dropna=False)
    ballots = []

    for group, group_df in grouped:
        # print(group)
        # print(group_df)
        ranking = list(group)
        voters = list(group_df.iloc[:, 0])
        weight = len(group_df)
        b = Ballot(ranking=ranking, weight=weight, voters=voters)
        ballots.append(b)
    
    return PreferenceProfile(ballots=ballots)

if __name__ == '__main__':
    p = CVRLoader(load_func=rank_column_csv)
    prof = p.load_cvr("/Users/jenniferwang/Projects/unnamed_rcv_thing/tests/data/undervote.csv")
    # print(prof)
    # # example of what testing for an error looks like
    # with pytest.raises(EmptyDataError):
    #     p.parse_csv(DATA_DIR / "empty.csv")
