from unnamed_rcv_thing.csv_parser import CSVParser
from pathlib import Path
import pytest
from pandas.errors import EmptyDataError

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def test_empty_csv():
    p = CSVParser()
    # example of what testing for an error looks like
    with pytest.raises(EmptyDataError):
        p.parse_csv(DATA_DIR / "empty.csv")


def test_undervote():
    p = CSVParser()
    p.parse_csv(DATA_DIR / "undervote.csv")
    # TODO: these tests intentionally fail, fix them
    assert p.candidate_ranking


def test_only_cols():
    p = CSVParser()
    p.parse_csv(DATA_DIR / "only_cols.csv")
    assert p.candidate_ranking
