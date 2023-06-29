from unnamed_rcv_thing.cvr_loader import CVRLoader, rank_column_csv
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
    # TODO: these tests intentionally fail, fix them
    assert prof


# def test_only_cols():
#     p = CVRLoader(load_func=rank_column_csv)
#     p.load_cvr(DATA_DIR / "only_cols.csv")
#     assert p.candidate_ranking
