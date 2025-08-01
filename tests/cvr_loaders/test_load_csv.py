from pandas.errors import EmptyDataError

# from urllib.error import URLError
from pathlib import Path
import pytest
import re
from votekit.cvr_loaders import load_csv


BASE_DIR = Path(__file__).resolve().parent.parent
CSV_DIR = BASE_DIR / "data/csv/load_csv"


def test_empty_csv():
    with pytest.raises(EmptyDataError):
        load_csv(CSV_DIR / "empty.csv")


def test_invalid_path():
    with pytest.raises(FileNotFoundError):
        load_csv("fake_path.csv")


# occasionally stalls the test suite
# def test_invalid_url():
#     with pytest.raises(URLError):
#         load_csv("http://example.com/bad.csv")


# TODO bad encoding
# TODO parser error
# TODO HTTPError


def test_one_column_error():
    with pytest.raises(
        ValueError,
        match="CSV has only one column but one of weight_col or id_col is provided.",
    ):
        load_csv(CSV_DIR / "one_column_error.csv", weight_col=0)
    with pytest.raises(
        ValueError,
        match="CSV has only one column but one of weight_col or id_col is provided.",
    ):
        load_csv(CSV_DIR / "one_column_error.csv", id_col=0)


def test_candidate_errors():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Candidate Chris was provided in candidates ['Chris', 'Peter'] but "
            "not found in the csv."
        ),
    ):
        load_csv(CSV_DIR / "missing_candidate_error.csv", candidates=["Chris", "Peter"])

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Candidate Chris was found in the csv but not provided in candidates ['Peter']."
        ),
    ):
        load_csv(CSV_DIR / "extra_candidate_error.csv", candidates=["Peter"])


def test_id_and_weight_error():
    with pytest.raises(
        ValueError, match="Only one of weight_col and id_col can be provided"
    ):
        load_csv(CSV_DIR / "valid_cvr.csv", weight_col=3, id_col=4)


def test_distinct_col_error():
    with pytest.raises(
        ValueError,
        match=re.escape(f"ID column {1} must not be a ranking column {[0,1,2]}."),
    ):
        load_csv(CSV_DIR / "valid_cvr.csv", rank_cols=[0, 1, 2], id_col=1)

    with pytest.raises(
        ValueError,
        match=re.escape(f"Weight column {1} must not be a ranking column {[0,1,2]}."),
    ):
        load_csv(CSV_DIR / "valid_cvr.csv", rank_cols=[0, 1, 2], weight_col=1)


def test_columns_out_of_range_error():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "column index -1 must be in [0, 4] because Python is 0-indexed."
        ),
    ):
        load_csv(CSV_DIR / "valid_cvr.csv", rank_cols=[-1])
    with pytest.raises(
        ValueError,
        match=re.escape(
            "column index -1 must be in [0, 4] because Python is 0-indexed."
        ),
    ):
        load_csv(CSV_DIR / "valid_cvr.csv", weight_col=-1)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "column index -1 must be in [0, 4] because Python is 0-indexed."
        ),
    ):
        load_csv(CSV_DIR / "valid_cvr.csv", id_col=-1)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "column index 5 must be in [0, 4] because Python is 0-indexed."
        ),
    ):
        load_csv(CSV_DIR / "valid_cvr.csv", rank_cols=[5])

    with pytest.raises(
        ValueError,
        match=re.escape(
            "column index 5 must be in [0, 4] because Python is 0-indexed."
        ),
    ):
        load_csv(CSV_DIR / "valid_cvr.csv", id_col=5)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "column index 5 must be in [0, 4] because Python is 0-indexed."
        ),
    ):
        load_csv(CSV_DIR / "valid_cvr.csv", weight_col=5)


def test_numerical_weights_error():
    with pytest.raises(
        ValueError, match=f"Weight {'a'} in row {0} must be able to be cast to float."
    ):
        load_csv(CSV_DIR / "non_numeric_weight.csv", rank_cols=[0, 1, 2], weight_col=3)

    with pytest.raises(ValueError, match=f"No weight provided in row {1}."):
        load_csv(CSV_DIR / "no_weight_provided.csv", rank_cols=[0, 1, 2], weight_col=3)


def test_header_error():
    with pytest.raises(ValueError, match="Header -1 must be non-negative."):
        load_csv(CSV_DIR / "valid_cvr.csv", header=-1)


################# old tests
# def test_undervote():
#     prof = load_csv(CSV_DIR / "undervote.csv", id_col=0)
#     a = Ballot(ranking=[{"c"}, frozenset(), frozenset()], weight=1, voter_set={"a"})
#     correct_prof = PreferenceProfile(ballots=[a])
#     assert correct_prof == prof


# def test_only_cols():
#     with pytest.raises(EmptyDataError):
#         load_csv(CSV_DIR / "only_cols.csv", id_col=0)


# def test_duplicates_candidates():
#     prof = load_csv(CSV_DIR / "dup_cands.csv", id_col=0)
#     # assert len(prof.ballots) == 3
#     abe = Ballot(ranking=[{"b"}, {"c"}, {"c"}], weight=1, voter_set={"abe"})
#     don = Ballot(ranking=[{"a"}, {"c"}, {"c"}], weight=1, voter_set={"don"})
#     carrie = Ballot(ranking=[{"c"}, {"c"}, {"c"}], weight=1, voter_set={"carrie"})
#     correct_prof = PreferenceProfile(ballots=[abe, don, carrie])
#     assert prof == correct_prof


# def test_single_row():
#     prof = load_csv(CSV_DIR / "single_row.csv", id_col=0)
#     a = Ballot(ranking=[{"b"}, {"c"}, {"d"}], weight=1, voter_set={"a"})
#     correct_prof = PreferenceProfile(ballots=[a])
#     assert prof == correct_prof


# def test_multiple_undervotes():
#     prof = load_csv(CSV_DIR / "mult_undervote.csv", id_col=0)
#     abc = Ballot(
#         ranking=[{"c"}, frozenset(), frozenset()],
#         weight=3,
#         voter_set={"abe", "ben", "carl"},
#     )
#     dave = Ballot(
#         ranking=[frozenset(), {"a"}, frozenset()],
#         weight=1,
#         voter_set={"dave"},
#     )
#     correct_prof = PreferenceProfile(ballots=[abc, dave])
#     assert correct_prof == prof


# def test_different_undervotes():
#     prof = load_csv(CSV_DIR / "diff_undervote.csv", id_col=0)
#     a = Ballot(ranking=[{"c"}, frozenset(), {"b"}], weight=1, voter_set={"a"})
#     b = Ballot(ranking=[frozenset(), {"d"}, frozenset()], weight=1, voter_set={"b"})
#     c = Ballot(ranking=[{"e"}, frozenset(), {"e"}], weight=1, voter_set={"c"})
#     correct_prof = PreferenceProfile(ballots=[a, b, c])
#     assert correct_prof == prof


# def test_duplicate_ballots():
#     prof = load_csv(CSV_DIR / "dup_ballots.csv", id_col=0)
#     a = Ballot(ranking=[{"b"}, {"c"}, {"c"}], weight=1, voter_set={"abe"})
#     dc = Ballot(ranking=[{"c"}, {"c"}, {"c"}], weight=2, voter_set={"don", "carrie"})
#     correct_prof = PreferenceProfile(ballots=[a, dc])
#     assert correct_prof == prof


# def test_combo():
#     prof = load_csv(CSV_DIR / "combo.csv", id_col=0)
#     abc = Ballot(
#         ranking=[{"b"}, {"c"}, {"c"}],
#         weight=3,
#         voter_set={"abe", "ben", "carrie"},
#     )
#     de = Ballot(
#         ranking=[{"c"}, frozenset(), frozenset()],
#         weight=2,
#         voter_set={"don", "ed"},
#     )
#     correct_prof = PreferenceProfile(ballots=[abc, de])
#     assert correct_prof == prof


# def test_diff_candidates():
#     prof = load_csv(CSV_DIR / "diff_cands.csv", id_col=0)
#     abe = Ballot(ranking=[{"a"}, {"b"}, {"c"}], voter_set={"abe"}, weight=1)
#     don = Ballot(ranking=[{"d"}, {"e"}, {"f"}], weight=1, voter_set={"don"})
#     carrie = Ballot(ranking=[{"g"}, {"h"}, {"i"}], weight=1, voter_set={"carrie"})
#     correct_prof = PreferenceProfile(ballots=[abe, don, carrie])
#     assert correct_prof == prof


# def test_same_candidates():
#     prof = load_csv(CSV_DIR / "same_cands.csv", id_col=0)
#     abe = Ballot(ranking=[{"a"}, {"b"}, {"c"}], voter_set={"abe"}, weight=1)
#     don = Ballot(ranking=[{"c"}, {"b"}, {"a"}], weight=1, voter_set={"don"})
#     carrie = Ballot(ranking=[{"a"}, {"c"}, {"b"}], weight=1, voter_set={"carrie"})
#     correct_prof = PreferenceProfile(ballots=[abe, don, carrie])
#     assert correct_prof == prof


# def test_special_char():
#     prof = load_csv(CSV_DIR / "special_char.csv", id_col=0)
#     a1 = Ballot(ranking=[{"b@#"}, {"@#$"}, {"c"}], weight=2, voter_set={"a@#", "1@#"})
#     d = Ballot(ranking=[{"!23"}, {"c"}, {"c"}], weight=1, voter_set={"d#$"})
#     correct_prof = PreferenceProfile(ballots=[a1, d])
#     assert correct_prof == prof


# def test_unnamed_ballot():
#     with pytest.raises(ValueError):
#         load_csv(CSV_DIR / "unnamed.csv", id_col=0)


# def test_same_name():
#     with pytest.raises(DataError):
#         load_csv(CSV_DIR / "same_name.csv", id_col=0)


# def malformed_rows():
#     p = CVRLoader(load_func=rank_column_csv)
#     # p.load_cvr(DATA_DIR / "malformed.csv")
#     # print(p)
