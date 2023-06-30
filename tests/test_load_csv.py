from unnamed_rcv_thing.cvr_loader import CVRLoader, rank_column_csv
from unnamed_rcv_thing.ballot import Ballot
from unnamed_rcv_thing.profile import PreferenceProfile
import numpy as np
from pathlib import Path
import pytest
from pandas.errors import EmptyDataError

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

def test_empty_csv():
    p = CVRLoader(load_func=rank_column_csv)
    # example of what testing for an error looks like
    with pytest.raises(EmptyDataError):
        p.load_cvr(DATA_DIR / "empty.csv")

def test_undervote():
    p = CVRLoader(load_func=rank_column_csv)
    prof = p.load_cvr(DATA_DIR / "undervote.csv")
    a = Ballot(id=None, ranking=['c', np.nan, np.nan], weight=1.0, voters=['a'])
    correct_prof = PreferenceProfile(ballots={a})
    print(correct_prof)
    print(prof)
    #TODO: why does this fail?
    # assert correct_prof.ballots == prof.ballots
    assert correct_prof == prof

def test_only_cols():
    p = CVRLoader(load_func=rank_column_csv)
    with pytest.raises(EmptyDataError):
        p.load_cvr(DATA_DIR / "only_cols.csv")

def test_invalid_path():
    p = CVRLoader(load_func=rank_column_csv)
    with pytest.raises(FileNotFoundError):
        p.load_cvr('fake_path.csv')

# def test_duplicates_candidates():
#     p = CVRLoader(load_func=rank_column_csv)
#     prof = p.load_cvr(DATA_DIR / "dup_cands.csv")
#     # assert len(prof.ballots) == 3
#     abe = Ballot(ranking=['b', 'c', 'c'], weight=1, voters=['abe'])
#     don = Ballot(ranking=['a', 'c', 'c'], weight=1, voters=['don'])
#     carrie = Ballot(ranking=['c', 'c', 'c'], weight=1, voters=['carrie'])
#     correct_prof = PreferenceProfile(ballots=[abe, don, carrie])
#     print(prof)
#     print(correct_prof)
#     assert prof.ballots == correct_prof.ballots
#     # assert prof == correct_prof

# def test_single_row():
#     p = CVRLoader(load_func=rank_column_csv)
#     prof = p.load_cvr(DATA_DIR / "single_row.csv")
#     a = Ballot(ranking=['b', 'c', 'd'], weight=1, voters=['a'])
#     correct_prof = PreferenceProfile(ballots=[a])
#     assert prof == correct_prof

# def test_multiple_undervotes():
#     p = CVRLoader(load_func=rank_column_csv)
#     prof = p.load_cvr(DATA_DIR / "mult_undervote.csv")
#     abc = Ballot(ranking=['c,', '', ''], weight=3, voters=['abe', 'ben', 'carl'])
#     dave = Ballot(ranking=['a', '', ''], weight=1, voters=['dave'])
#     correct_prof = PreferenceProfile(ballots=[abc, dave])
#     assert prof == correct_prof

def test_duplicate_ballots():
    p = CVRLoader(load_func=rank_column_csv)

def test_combo():
    p = CVRLoader(load_func=rank_column_csv)

def test_diff_candidates():
    p = CVRLoader(load_func=rank_column_csv)

def test_same_candidates():
    p = CVRLoader(load_func=rank_column_csv)

def test_special_char():
    p = CVRLoader(load_func=rank_column_csv)

def wrong_type():
    p = CVRLoader(load_func=rank_column_csv)

def unnamed_ballot():
    p = CVRLoader(load_func=rank_column_csv)
    # TODO: this should fail right?
