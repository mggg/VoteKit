from unnamed_rcv_thing.cvr_loaders import rank_column_csv
from unnamed_rcv_thing.ballot import Ballot
from unnamed_rcv_thing.profile import PreferenceProfile
from pathlib import Path
import pytest
from pandas.errors import EmptyDataError, DataError

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def is_equal(b1: list[Ballot], b2: list[Ballot]) -> bool:
    if len(b1) != len(b2):
        return False
    for b in b1:
        if b not in b2:
            return False
    return True


def test_empty_csv():
    with pytest.raises(EmptyDataError):
        rank_column_csv(DATA_DIR / "empty.csv", id_col=0)


def test_undervote():
    prof = rank_column_csv(DATA_DIR / "undervote.csv", id_col=0)
    a = Ballot(id=None, ranking=[{"c"}, {None}, {None}], weight=1.0, voters={"a"})
    correct_prof = PreferenceProfile(ballots=[a])
    assert correct_prof == prof
    # assert correct_prof.ballots[0].ranking == prof.ballots[0].ranking


def test_only_cols():
    with pytest.raises(EmptyDataError):
        rank_column_csv(DATA_DIR / "only_cols.csv", id_col=0)


def test_invalid_path():
    with pytest.raises(FileNotFoundError):
        rank_column_csv("fake_path.csv", id_col=0)


def test_duplicates_candidates():
    prof = rank_column_csv(DATA_DIR / "dup_cands.csv", id_col=0)
    # assert len(prof.ballots) == 3
    abe = Ballot(ranking=[{"b"}, {"c"}, {"c"}], weight=1, voters={"abe"})
    don = Ballot(ranking=[{"a"}, {"c"}, {"c"}], weight=1, voters={"don"})
    carrie = Ballot(ranking=[{"c"}, {"c"}, {"c"}], weight=1, voters={"carrie"})
    correct_prof = PreferenceProfile(ballots=[abe, don, carrie])
    assert is_equal(prof.ballots, correct_prof.ballots)


def test_single_row():
    prof = rank_column_csv(DATA_DIR / "single_row.csv", id_col=0)
    a = Ballot(ranking=[{"b"}, {"c"}, {"d"}], weight=1, voters={"a"})
    correct_prof = PreferenceProfile(ballots=[a])
    assert is_equal(prof.ballots, correct_prof.ballots)


def test_multiple_undervotes():
    prof = rank_column_csv(DATA_DIR / "mult_undervote.csv", id_col=0)
    abc = Ballot(
        ranking=[{"c"}, {None}, {None}], weight=3, voters={"abe", "ben", "carl"}
    )
    dave = Ballot(ranking=[{None}, {"a"}, {None}], weight=1, voters={"dave"})
    correct_prof = PreferenceProfile(ballots=[abc, dave])
    assert is_equal(correct_prof.ballots, prof.ballots)


def test_different_undervotes():
    prof = rank_column_csv(DATA_DIR / "diff_undervote.csv", id_col=0)
    a = Ballot(ranking=[{"c"}, {None}, {"b"}], weight=1, voters={"a"})
    b = Ballot(ranking=[{None}, {"d"}, {None}], weight=1, voters={"b"})
    c = Ballot(ranking=[{"e"}, {None}, {"e"}], weight=1, voters={"c"})
    correct_prof = PreferenceProfile(ballots=[a, b, c])
    assert is_equal(correct_prof.ballots, prof.ballots)


def test_duplicate_ballots():
    prof = rank_column_csv(DATA_DIR / "dup_ballots.csv", id_col=0)
    a = Ballot(ranking=[{"b"}, {"c"}, {"c"}], weight=1, voters={"abe"})
    dc = Ballot(ranking=[{"c"}, {"c"}, {"c"}], weight=2, voters={"don", "carrie"})
    correct_prof = PreferenceProfile(ballots=[a, dc])
    assert is_equal(correct_prof.ballots, prof.ballots)


def test_combo():
    prof = rank_column_csv(DATA_DIR / "combo.csv", id_col=0)
    abc = Ballot(
        ranking=[{"b"}, {"c"}, {"c"}], weight=3, voters={"abe", "ben", "carrie"}
    )
    de = Ballot(ranking=[{"c"}, {None}, {None}], weight=2, voters={"don", "ed"})
    correct_prof = PreferenceProfile(ballots=[abc, de])
    assert is_equal(correct_prof.ballots, prof.ballots)


def test_diff_candidates():
    prof = rank_column_csv(DATA_DIR / "diff_cands.csv", id_col=0)
    abe = Ballot(ranking=[{"a"}, {"b"}, {"c"}], voters={"abe"}, weight=1)
    don = Ballot(ranking=[{"d"}, {"e"}, {"f"}], weight=1, voters={"don"})
    carrie = Ballot(ranking=[{"g"}, {"h"}, {"i"}], weight=1, voters={"carrie"})
    correct_prof = PreferenceProfile(ballots=[abe, don, carrie])
    assert is_equal(correct_prof.ballots, prof.ballots)


def test_same_candidates():
    prof = rank_column_csv(DATA_DIR / "same_cands.csv", id_col=0)
    abe = Ballot(ranking=[{"a"}, {"b"}, {"c"}], voters={"abe"}, weight=1)
    don = Ballot(ranking=[{"c"}, {"b"}, {"a"}], weight=1, voters={"don"})
    carrie = Ballot(ranking=[{"a"}, {"c"}, {"b"}], weight=1, voters={"carrie"})
    correct_prof = PreferenceProfile(ballots=[abe, don, carrie])
    assert is_equal(correct_prof.ballots, prof.ballots)


def test_special_char():
    prof = rank_column_csv(DATA_DIR / "special_char.csv", id_col=0)
    a1 = Ballot(ranking=[{"b@#"}, {"@#$"}, {"c"}], weight=2, voters={"a@#", "1@#"})
    d = Ballot(ranking=[{"!23"}, {"c"}, {"c"}], weight=1, voters={"d#$"})
    correct_prof = PreferenceProfile(ballots=[a1, d])
    assert is_equal(correct_prof.ballots, prof.ballots)


def test_unnamed_ballot():
    with pytest.raises(ValueError):
        rank_column_csv(DATA_DIR / "unnamed.csv", id_col=0)


def test_same_name():
    with pytest.raises(DataError):
        rank_column_csv(DATA_DIR / "same_name.csv", id_col=0)


# def malformed_rows():
#     p = CVRLoader(load_func=rank_column_csv)
#     # p.load_cvr(DATA_DIR / "malformed.csv")
#     # print(p)
